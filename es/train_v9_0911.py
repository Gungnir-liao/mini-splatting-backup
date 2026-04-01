import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))
import math
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

# ==================== 行为画像裁剪参数（无退火版） ====================
# 从哪一轮开始进行基于画像的裁剪 (建议在梯度致密化开始一段时间后)
PROFILE_PRUNE_START_ITER = 15000
# 画像裁剪持续的迭代数（统计窗口长度）
PRUNE_WINDOW_ITERS = 10000
# 目标：窗口内总计裁掉占比（相对第一次裁剪时的点数 N0）
PRUNE_TARGET_FRAC = 0.85
# 固定分位阈值（候选筛选用，不退火）
PRUNE_Q = 0.12
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

    num_train_cams = len(scene.getTrainCameras())
    window_rounds = max(1, PRUNE_WINDOW_ITERS // max(1, num_train_cams))
    rounds_per_prune = getattr(pipe, "rounds_per_densify", 1)
    effective_rounds = max(1, window_rounds // max(1, rounds_per_prune))

    # 形状参数：越大越“前重后轻”。0=均匀，1≈线性，1.5~2 更激进
    PRUNE_POLY_GAMMA = 1.5

    # 预计算权重：w_r ∝ (剩余轮数)^gamma，r=1..R；并归一化到和为 1
    R = effective_rounds
    w = torch.arange(R, 0, -1, dtype=torch.float64) ** PRUNE_POLY_GAMMA  # R, R-1, ..., 1
    w = (w / w.sum()).tolist()  # -> list[float]，长度 R

    prune_sched = {
        "N0": None,                # 首次裁剪时的基准点数
        "removed_so_far": 0,       # 已累计裁掉的绝对个数（口径：相对 N0）
        "round_idx": 0,            # 第几次裁剪（从 1 开始）
        "total_rounds": R,
        "weights": w,              # 归一权重表
        "target": PRUNE_TARGET_FRAC # 例如 0.85
    }

    preview = [f"{wi:.3f}" for wi in w[:min(5, R)]]
    print(f"[PRUNE SCHED] cams={num_train_cams}, window_rounds={window_rounds}, "
        f"effective_rounds={R}, gamma={PRUNE_POLY_GAMMA}, "
        f"weights_head={preview} (sum≈{sum(w):.3f}), target={PRUNE_TARGET_FRAC:.0%}")
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
            in_window = (PROFILE_PRUNE_START_ITER <= iteration <
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

            # ==================== 2) 标准梯度致密化（原版逻辑） ====================
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ==================== 3) 回合末：按均摊 p_cap 做 OR 裁剪 ====================
            end_of_round = (len(viewpoint_stack) == 0)
            if end_of_round:
                rounds_seen += 1
                viewpoint_stack = scene.getTrainCameras().copy()

            do_profile_prune = in_window and end_of_round and (rounds_seen % max(1, rounds_per_prune) == 0)

            if do_profile_prune and cycle_stats["seen_views"].sum() > 0:
                # 1) 首次记录 N0（后续都按它的绝对个数口径）
                if prune_sched["N0"] is None:
                    prune_sched["N0"] = int(gaussians.get_xyz.shape[0])
                N0 = prune_sched["N0"]

                # 2) 推进轮次
                prune_sched["round_idx"] += 1
                r = prune_sched["round_idx"]                # 当前第 r 次（1-based）
                R = prune_sched["total_rounds"]
                r = min(r, R)                               # 防御

                # 3) 剩余目标（绝对个数，口径 N0）
                target_total = int(round(prune_sched["target"] * N0))      # 例如 0.85 * N0
                remaining_needed = max(0, target_total - prune_sched["removed_so_far"])
                if remaining_needed == 0:
                    print(f"[ITER {iteration}] PRUNE r{r}/{R}: done (cum=100% of target, N0={N0})")
                    cycle_stats = _alloc_behavior_stats(gaussians.get_xyz.shape[0])
                    # 重置统计后直接结束本次裁剪
                    # （如果你想继续训练逻辑，用 'continue' 跳过下面部分）
                else:
                    # 4) 计算本轮权重占比，并把“剩余目标”按比例切给本轮
                    w_all = prune_sched["weights"]          # list[float], sum=1
                    w_this = w_all[r-1]
                    w_left_sum = sum(w_all[r-1:])           # 当前及未来轮次的权重和（>0）

                    # 本轮“绝对个数配额”（向上取整，确保能吃干净尾差）
                    k_cap_abs = math.ceil(remaining_needed * (w_this / w_left_sum))

                    # ====== 你的候选生成（保持原逻辑） ======
                    num_gaussians = gaussians.get_xyz.shape[0]
                    seen   = cycle_stats["seen_views"]
                    px     = cycle_stats["covered_px"]
                    used   = cycle_stats["used_views"]
                    top1   = cycle_stats["top1_px"]
                    sum_w  = cycle_stats["sum_w"]
                    px_max = cycle_stats["px_max"]

                    mean_w_px = sum_w / (px + eps)
                    top1_rate = top1 / (px + eps)
                    used_safe = torch.clamp(used, min=1.0)
                    px_per_used_view = px / used_safe

                    q = PRUNE_Q
                    th_w    = torch.quantile(mean_w_px, q)             # 低贡献
                    th_top1 = torch.quantile(top1_rate, q)             # 低主导
                    th_pxM  = torch.quantile(px_max, q)                # 小尺寸
                    th_pv   = torch.quantile(px_per_used_view, 1 - q)  # 大块

                    c1 = (mean_w_px <= th_w)
                    c2 = (top1_rate <= th_top1)
                    c3 = (px_max   <= th_pxM)
                    c4 = (px_per_used_view >= th_pv)
                    cands = c1 | c2 | c3 | c4
                    cand_idx = torch.where(cands)[0]

                    # 5) 候选与配额夹紧，挑样本
                    k_cap = min(k_cap_abs, cand_idx.numel())
                    if k_cap > 0:
                        order = torch.argsort(mean_w_px[cand_idx], descending=False)  # 贡献更低优先
                        pick = cand_idx[order[:k_cap]]

                        prune_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=mean_w_px.device)
                        prune_mask[pick] = True

                        before_N = int(gaussians.get_xyz.shape[0])
                        gaussians.prune_points(prune_mask)
                        pruned_count = int(prune_mask.sum().item())
                        after_N = int(gaussians.get_xyz.shape[0])

                        prune_sched["removed_so_far"] += pruned_count
                        cum_frac = prune_sched["removed_so_far"] / N0

                        print(
                            f"\n[ITER {iteration}] PRUNE r{r}/{R}: "
                            f"w_this={w_this:.4f}, remain_need={remaining_needed}, "
                            f"cands={cand_idx.numel()}, cap_abs={k_cap_abs}, pruned={pruned_count}, "
                            f"N: {before_N} -> {after_N}, cum={cum_frac:.4%} of N0={N0}"
                        )
                    else:
                        print(
                            f"\n[ITER {iteration}] PRUNE r{r}/{R}: "
                            f"w_this={w_this:.4f}, remain_need={remaining_needed}, "
                            f"cands={cand_idx.numel()}, cap_abs={k_cap_abs}, pruned=0"
                        )

                    # 6) 重置统计缓冲，开始下一轮
                    cycle_stats = _alloc_behavior_stats(gaussians.get_xyz.shape[0])


            # ==================== Optimizer step ====================
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 100 == 0:
                print(f"[ITER {iteration}] Current number of Gaussians: {gaussians.get_xyz.shape[0]}")

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
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
