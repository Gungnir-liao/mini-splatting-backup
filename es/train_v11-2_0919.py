import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_es, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False
from utils.sh_utils import SH2RGB


# ==================== 行为画像裁剪参数（无退火版） ====================
# 从哪一轮开始进行基于画像的裁剪 (建议在梯度致密化开始一段时间后)
PROFILE_PRUNE_START_ITER = 15000
# 画像裁剪持续的迭代数（统计窗口长度）
PRUNE_WINDOW_ITERS = 10000
# 目标：窗口内总计裁掉占比（相对第一次裁剪时的点数 N0）
PRUNE_TARGET_FRAC = 0.95
# 固定分位阈值（候选筛选用，不退火）
PRUNE_Q = 0.05
# ================================================================

def _to_indices(vis):
    """把 visibility_filter 统一转成 1D long 索引列表"""
    if vis.dtype == torch.bool:
        return torch.nonzero(vis, as_tuple=False).squeeze(-1).long()
    return vis.reshape(-1).long()

def _alloc_behavior_stats(N: int, device="cuda"):
    return {
        "sum_w":         torch.zeros(N, device=device, dtype=torch.float32),   # ∑(α·T)
        "covered_px":    torch.zeros(N, device=device, dtype=torch.float32),   # 覆盖像素总数（像素级）
        "top1_px":       torch.zeros(N, device=device, dtype=torch.float32),   # top-1 像素数（像素级）
        "seen_views":    torch.zeros(N, device=device, dtype=torch.float32),   # 被看到的帧数（帧级）
        "used_views":    torch.zeros(N, device=device, dtype=torch.float32),   # 真参与渲染的帧数（帧级）
        "faint_hits":    torch.zeros(N, device=device, dtype=torch.float32),   # α过低被跳过的像素命中数（像素级）
        "occ_alpha_sum": torch.zeros(N, device=device, dtype=torch.float32),   # 早停未收录 α 之和（像素/累和）
        "px_max":        torch.zeros(N, device=device, dtype=torch.float32),   # 最大覆盖像素值（像素级）
    }

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # ==================== 初始化行为画像统计 & 调度 ====================
    pipe.collect_stats = True  # 强制打开统计开关
    rounds_per_prune = getattr(pipe, "rounds_per_densify", 1)  # 每 N 轮裁剪一次
    rounds_seen = 0
    eps = 1e-8

    cycle_stats = _alloc_behavior_stats(gaussians.get_xyz.shape[0])

    # 计算本窗口会发生的“回合末裁剪”次数，并均摊 p_cap
    num_train_cams = len(scene.getTrainCameras())
    window_rounds = max(1, PRUNE_WINDOW_ITERS // max(1, num_train_cams))
    effective_rounds = max(1, window_rounds // max(1, rounds_per_prune))
    base_p_cap = round(PRUNE_TARGET_FRAC / effective_rounds, 4)
    base_p_cap = float(min(0.95, max(0.000, base_p_cap)))
    prune_sched = {
        "N0": None,                # 首次裁剪时的基准点数
        "removed_so_far": 0,       # 已累计裁掉的个数（口径：相对 N0）
        "round_idx": 0,            # 第几次裁剪（从 1 开始）
        "total_rounds": effective_rounds,
        "base_cap": base_p_cap,
        "target": PRUNE_TARGET_FRAC
    }
    print(f"[PRUNE SCHED] cams={num_train_cams}, window_rounds={window_rounds}, "
          f"effective_rounds={effective_rounds}, base_p_cap={base_p_cap}, target={PRUNE_TARGET_FRAC:.0%}")
    # ================================================================

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_es(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if iteration<PROFILE_PRUNE_START_ITER+PRUNE_WINDOW_ITERS:
            gaussians.update_learning_rate(iteration)
        else:
            gaussians.update_learning_rate(iteration-(PROFILE_PRUNE_START_ITER+PRUNE_WINDOW_ITERS)+5000)

        if iteration % 1000 == 0 and iteration>PROFILE_PRUNE_START_ITER+PRUNE_WINDOW_ITERS:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_es(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render_es, (pipe, background))
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ==================== 1) 行为画像数据收集（在 densify/prune 之前） ====================
            in_window = (PROFILE_PRUNE_START_ITER < iteration <=
                         PROFILE_PRUNE_START_ITER + PRUNE_WINDOW_ITERS)

            if in_window:
                # 若点数变化，重置缓冲大小
                if cycle_stats["seen_views"].shape[0] != gaussians.get_xyz.shape[0]:
                    cycle_stats = _alloc_behavior_stats(gaussians.get_xyz.shape[0])

                # 统一索引形式
                vis_idx = _to_indices(visibility_filter)

                # a) seen_views（帧级）
                if vis_idx.numel() > 0:
                    cycle_stats["seen_views"].index_add_(0, vis_idx,
                        torch.ones_like(vis_idx, dtype=torch.float32, device=vis_idx.device))

                # b) 像素/能量统计
                if "accum_weights" in render_pkg:
                    wframe = render_pkg["accum_weights"].to(torch.float32)
                    cycle_stats["sum_w"] += wframe
                else:
                    wframe = None

                if "area_proj" in render_pkg:
                    covered_px_frame = render_pkg["area_proj"].to(torch.float32)
                    cycle_stats["covered_px"] += covered_px_frame
                    cycle_stats["px_max"] = torch.maximum(cycle_stats["px_max"], covered_px_frame)
                    # used：简单稳妥 —— 只要覆盖>0就算一次（如需更严可加分位门槛）
                    cycle_stats["used_views"] += (covered_px_frame > 0).float()

                if "area_max" in render_pkg:
                    cycle_stats["top1_px"] += render_pkg["area_max"].to(torch.float32)
                if "faint_count" in render_pkg:
                    cycle_stats["faint_hits"] += render_pkg["faint_count"].to(torch.float32)
                if "occluded_alpha" in render_pkg:
                    cycle_stats["occ_alpha_sum"] += render_pkg["occluded_alpha"].to(torch.float32)
                
                # 统计梯度
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # ==================== 2) 标准梯度致密化（原版逻辑） ====================
            if iteration <= PROFILE_PRUNE_START_ITER:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.002,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ==================== 3) 回合末：按均摊 p_cap 做 OR 裁剪 ====================
            if in_window:
                end_of_round = (len(viewpoint_stack) == 0)
                if end_of_round:
                    rounds_seen += 1
                    viewpoint_stack = scene.getTrainCameras().copy()

                do_profile_prune = end_of_round and (rounds_seen % max(1, rounds_per_prune) == 0)

                if do_profile_prune and cycle_stats["seen_views"].sum() > 0:
                    low_mask = gaussians.collect_and_get_lowgrad_mask(
                        max_grad=opt.densify_grad_threshold/num_train_cams,
                        include_unseen=False,
                        reset_buffers=True
                    )
                    # 首次裁剪时记录基准 N0（后续总目标按它累计）
                    if prune_sched["N0"] is None:
                        prune_sched["N0"] = int(gaussians.get_xyz.shape[0])

                    prune_sched["round_idx"] += 1
                    rounds_left = max(1, prune_sched["total_rounds"] - prune_sched["round_idx"] + 1)

                    # 还需完成的目标占比（相对 N0）
                    remaining_target = max(
                        0.0,
                        prune_sched["target"] - prune_sched["removed_so_far"] / prune_sched["N0"]
                    )

                    # 本轮 cap：基础 cap 与“把剩余目标均分到余下轮数”的较小者；保留 4 位；夹紧
                    p_cap_now = min(prune_sched["base_cap"], round(remaining_target / rounds_left, 4))
                    p_cap_now = float(min(0.95, max(0.0, p_cap_now)))

                    eps0 = 0.0
                    never_used_mask = (cycle_stats["used_views"] <= eps0)
                    never_seen_mask = (cycle_stats["seen_views"] <= eps0)
                    seen_but_not_used_mask = (~never_seen_mask) & never_used_mask

                    never_used_cnt = int(never_used_mask.sum().item())
                    never_seen_cnt = int(never_seen_mask.sum().item())
                    seen_but_not_used_cnt = int(seen_but_not_used_mask.sum().item())

                    # 取出统计，计算四指标
                    num_gaussians = gaussians.get_xyz.shape[0]
                    seen   = cycle_stats["seen_views"]
                    px     = cycle_stats["covered_px"]
                    used   = cycle_stats["used_views"]
                    top1   = cycle_stats["top1_px"]
                    sum_w  = cycle_stats["sum_w"]
                    px_max = cycle_stats["px_max"]

                    eps = 1e-8
                    mean_w_px = sum_w / (px + eps)
                    top1_rate = top1 / (px + eps)
                    used_safe = torch.clamp(used, min=1.0)
                    px_per_used_view = px / used_safe

                    # 分位阈值（固定 q）
                    q = PRUNE_Q
                    th_w    = torch.quantile(mean_w_px, q)             # 低贡献
                    th_top1 = torch.quantile(top1_rate, q)             # 低主导
                    th_pxM  = torch.quantile(px_max, q)                # 小尺寸
                    th_pv   = torch.quantile(px_per_used_view, 1 - 3*q)# 大块（更极端）

                    # OR 候选（是否叠加低梯度门由你控制；此处与原逻辑保持一致）
                    c1 = (mean_w_px <= th_w)
                    c2 = (top1_rate <= th_top1)
                    c3 = (px_max <= th_pxM)
                    c4 = (px_per_used_view >= th_pv)
                    cands = c1
                    cand_idx = torch.where(cands)[0]

                    # ====== 关键改动：按 N0 计算“绝对个数”上限，并与剩余目标夹紧 ======
                    N0 = prune_sched["N0"]
                    target_total = int(round(prune_sched["target"] * N0))
                    remaining_needed = max(0, target_total - prune_sched["removed_so_far"])

                    k_cap_abs = int(round(p_cap_now * N0))
                    k_cap_abs = min(k_cap_abs, remaining_needed)
                    k_cap_abs = max(0, k_cap_abs)

                    # 候选可能不足，再与候选数夹一次得到实际要删的个数
                    k_cap = min(k_cap_abs, cand_idx.numel())

                    # ======================= 新增：四指标 → 重要性分数 =======================
                    @torch.no_grad()
                    def robust_norm(x: torch.Tensor, q_lo: float, q_hi: float, invert: bool=False, eps: float=1e-8):
                        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                        lo = torch.quantile(x, q_lo)
                        hi = torch.quantile(x, q_hi)
                        if (hi - lo) < eps:
                            out = torch.zeros_like(x)
                        else:
                            out = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
                        return (1.0 - out) if invert else out

                    # 四项分数（映射到 [0,1]，越大越该删）
                    s_w   = robust_norm(mean_w_px,        q_lo=q,   q_hi=1-q,   invert=True)   # 低贡献高分
                    s_t1  = robust_norm(top1_rate,        q_lo=q,   q_hi=1-q,   invert=True)   # 低主导高分
                    s_pxM = robust_norm(px_max,           q_lo=q,   q_hi=1-q,   invert=True)   # 小尺寸高分
                    s_pv  = robust_norm(px_per_used_view, q_lo=3*q, q_hi=1-3*q, invert=False)  # 大块高分

                    # 权重（可调）
                    w_w, w_t1, w_pxM, w_pv = 0.00, 0.40, 0.25, 0.35

                    prune_score_all =  (w_w*s_w + w_t1*s_t1 + w_pxM*s_pxM + w_pv*s_pv)
                    # 只在候选里排序
                    scores_sub = prune_score_all[cand_idx]

                    if k_cap > 0 and cand_idx.numel() > 0:
                        order = torch.argsort(scores_sub, descending=True)  # 分数越大越优先删
                        pick = cand_idx[order[:k_cap]]

                        prune_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=mean_w_px.device)
                        prune_mask[pick] = True

                        before_N = int(gaussians.get_xyz.shape[0])
                        gaussians.prune_points(prune_mask)
                        pruned_count = int(prune_mask.sum().item())
                        after_N = int(gaussians.get_xyz.shape[0])

                        prune_sched["removed_so_far"] += pruned_count
                        cum_frac = prune_sched["removed_so_far"] / N0

                        # 统计信息（各条件命中与候选规模）
                        c1n, c2n, c3n, c4n = int(c1.sum()), int(c2.sum()), int(c3.sum()), int(c4.sum())
                        print(
                            f"\n[ITER {iteration}] PRUNE r{prune_sched['round_idx']}/{prune_sched['total_rounds']}: "
                            f"c1={c1n}, c2={c2n}, c3={c3n}, c4={c4n}, "
                            f"low_mask={low_mask.sum().item()}, "
                            f"cands={cand_idx.numel()}, "
                            f"cap_abs={k_cap_abs} (~{(k_cap_abs / N0):.4%} of N0, remain={remaining_needed}), "
                            f"pruned={pruned_count}, N: {before_N} -> {after_N}, cum={cum_frac:.4%} of N0={N0} "
                            f"never_used_cnt={never_used_cnt}, never_seen_cnt={never_seen_cnt}, seen_but_not_used_cnt={seen_but_not_used_cnt} "
                        )
                    else:
                        c1n, c2n, c3n, c4n = int(c1.sum()), int(c2.sum()), int(c3.sum()), int(c4.sum())
                        print(
                            f"\n [ITER {iteration}] PRUNE r{prune_sched['round_idx']}/{prune_sched['total_rounds']}: "
                            f"c1={c1n}, c2={c2n}, c3={c3n}, c4={c4n}, "
                            f"cands={cand_idx.numel()}, cap_abs={k_cap_abs} (~{(k_cap_abs / N0 if N0>0 else 0):.2%} of N0), pruned=0 "
                            f"never_used_cnt={never_used_cnt}, never_seen_cnt={never_seen_cnt}, seen_but_not_used_cnt={seen_but_not_used_cnt}"
                        )


                    # 重置统计缓冲，开始下一轮
                    cycle_stats = _alloc_behavior_stats(gaussians.get_xyz.shape[0])

                '''
                if iteration % (PROFILE_PRUNE_START_ITER + PRUNE_WINDOW_ITERS) == 0 :
                    gaussians.max_sh_degree=dataset.sh_degree
                    gaussians.reinitial_pts(gaussians._xyz, 
                                        SH2RGB(gaussians._features_dc+0)[:,0])
                    
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                '''
                
            
            '''
            if iteration > (PROFILE_PRUNE_START_ITER + PRUNE_WINDOW_ITERS):
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % 25000 == 0:                    
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    size_threshold = 20 
                    gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent)
                    prune_mask = (gaussians.get_opacity < 0.002).squeeze()
                    if size_threshold:
                        big_points_vs = gaussians.max_radii2D > size_threshold
                        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                    gaussians.prune_points(prune_mask)
            '''
                        # ================== 新增：仅检测&统计“大点”，不处理 ==================
            if iteration >  PROFILE_PRUNE_START_ITER and iteration % 100 == 0:
                # 像素空间阈值（与 densify 步的习惯一致：过了 opacity reset 再启用 20px；你也可做成超参）
                screen_thr = 20 if iteration > PROFILE_PRUNE_START_ITER else None
                # 世界空间阈值（最大主轴 σ > frac * 场景尺度）
                big_ws_frac = getattr(opt, "big_ws_frac", 0.1)   # 可在 args 里加 --big_ws_frac 调
                extent = scene.cameras_extent

                # 世界空间大点：max σ 是否过大
                max_sigma = gaussians.get_scaling.max(dim=1).values     # 已激活的物理尺度 σ
                big_points_ws = (max_sigma > (big_ws_frac * extent))

                # 像素空间大点：跨视角的最大投影半径是否过大（gaussians.max_radii2D 持续取 max）
                if screen_thr is not None:
                    big_points_vs = (gaussians.max_radii2D > float(screen_thr))
                else:
                    big_points_vs = torch.zeros_like(big_points_ws, dtype=torch.bool)

                # 计数（总体 & 候选交集）
                big_vs_cnt       = int(big_points_vs.sum().item())
                big_ws_cnt       = int(big_points_ws.sum().item())

                print(
                    f"\n[ITER {iteration}] BIG POINTS: "
                    f"big_vs_cnt={big_vs_cnt}, big_ws_cnt={big_ws_cnt}"
                )


            # ==================== Optimizer step ====================
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 100 == 0:
                print(f"\n [ITER {iteration}] Current number of Gaussians: {gaussians.get_xyz.shape[0]}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--rounds_per_densify", type=int, default=1,
                        help="Run profile-based pruning once every N full training-view rounds")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt     = op.extract(args)
    pipe    = pp.extract(args)
    pipe.rounds_per_densify = args.rounds_per_densify

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
