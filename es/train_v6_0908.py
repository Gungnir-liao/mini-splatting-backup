import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# ==== [MOD]：render 需要支持在 pipe.collect_stats=True 时返回三项统计（accum_weights/area_proj/area_max） ====
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
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # =====================[MOD] 轮次&阶段调度 + Warmup =====================
    # 阶段：densify（原版致密化）/ prune（贡献度裁剪）
    phase = "densify"
    rounds_in_phase = 0                 # 当前阶段内已完成的轮数（完整遍历一次训练相机记 1 轮）
    total_rounds_seen = 0               # 训练至今完成的总轮数（用于 Warmup 计数）

    warmup_rounds = getattr(pipe, "warmup_rounds", 0)   # 前 N 轮仅致密化
    rounds_densify = getattr(pipe, "rounds_densify", 1) # Warmup 后：X 轮致密化
    rounds_prune   = getattr(pipe, "rounds_prune",   1) # Warmup 后：Y 轮裁剪

    # 统计收集相关（仅在 prune 阶段开启统计；Warmup 阶段不收集）
    pipe.collect_stats = getattr(pipe, "collect_stats", True)
    eps = 1e-8

    def _alloc_cycle_stats(num_pts: int, device="cuda"):
        buf = torch.zeros(num_pts, device=device, dtype=torch.float32)
        return buf.clone(), buf.clone(), buf.clone()  # sum_w, hits, top1

    cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])

    # 裁剪参数（可从 pipe 传入或走 CLI 默认）
    prune_q_w   = getattr(pipe, "prune_q_w", 0.05)   # 低贡献分位阈
    prune_q_dom = getattr(pipe, "prune_q_dom", 0.05) # 低主导分位阈
    prune_cap   = getattr(pipe, "prune_cap", 0.10)   # 单轮最多删比例（避免过激）
    hard_prune_min = getattr(pipe, "hard_prune_min", 100)  # 小规模数据时避免 0 删

    def _reset_stats_if_topology_changed():
        nonlocal cycle_sum_w, cycle_hits, cycle_top1
        if cycle_sum_w.shape[0] != gaussians._xyz.shape[0]:
            cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])

    def _switch_phase(new_phase: str):
        nonlocal phase, rounds_in_phase, cycle_sum_w, cycle_hits, cycle_top1
        phase = new_phase
        rounds_in_phase = 0
        # 切换阶段时清空统计，避免跨阶段污染
        cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])
        print(f"\n[PHASE] Switch to '{phase.upper()}'")

    # =====================[MOD-END]================================

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
        gaussians.update_learning_rate(iteration)

        # SH 提升
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 抽取一个训练相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render_es(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志/保存
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render_es, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # =====================[MOD] 阶段化逻辑 + Warmup =====================
            end_of_round = (len(viewpoint_stack) == 0)

            # —— DENSIFY 阶段（Warmup 或常规致密化阶段）——
            if phase == "densify":
                # 原版致密化：每个 iteration 都维护必要统计
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 轮结束：计数并决定是否切换
                if end_of_round:
                    # 致密化触发频率：你上个版本是每 iter 都 densify（> densify_from_iter）
                    # opt.densify_from_iter = 500
                    if iteration > opt.densify_from_iter and total_rounds_seen < 50:
                        size_threshold = 20 if total_rounds_seen > 25 else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    # 周期性重置不透明度
                    if total_rounds_seen % 25 == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                    total_rounds_seen += 1
                    rounds_in_phase += 1
                    viewpoint_stack = scene.getTrainCameras().copy()

                    # —— Warmup 期间：始终停留在 densify，不收集统计，不做裁剪
                    if total_rounds_seen < warmup_rounds:
                        # 重置阶段内轮计数，让 warmup 不消耗后续 X 轮配额
                        rounds_in_phase = 0
                        print(f"[WARMUP] finished round {total_rounds_seen}/{warmup_rounds}")
                    else:
                        # Warmup 已完成，进入常规 X/Y 交替
                        if rounds_in_phase >= rounds_densify:
                            _switch_phase("prune")

            # —— PRUNE 阶段（仅在 Warmup 结束后才会进入）——
            else:  # phase == "prune"
                # 只在 prune 阶段统计贡献度（Warmup 已过）
                if getattr(pipe, "collect_stats", False) and \
                   ("accum_weights" in render_pkg and "area_proj" in render_pkg and "area_max" in render_pkg):
                    _reset_stats_if_topology_changed()
                    cycle_sum_w += render_pkg["accum_weights"]
                    cycle_hits  += render_pkg["area_proj"].to(torch.float32)
                    cycle_top1  += render_pkg["area_max"].to(torch.float32)

                if end_of_round:
                    # 一轮完成：用本轮统计做一次温和裁剪
                    if getattr(pipe, "collect_stats", False) and (cycle_hits.sum() > 0):
                        mean_w    = cycle_sum_w / (cycle_hits + eps)
                        dom_share = cycle_top1  / (cycle_hits + eps)

                        th_w   = torch.quantile(mean_w, prune_q_w)
                        th_dom = torch.quantile(dom_share, prune_q_dom)
                        prune_mask = (mean_w <= th_w) & (dom_share <= th_dom)

                        if prune_mask.any():
                            frac = prune_mask.float().mean().item()
                            # 上限保护
                            cap = max(prune_cap, (hard_prune_min / max(1, prune_mask.numel())))
                            if frac > cap:
                                k = max(1, int(cap * prune_mask.numel()))
                                cand_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(1)
                                _, order = torch.sort(mean_w[cand_idx], descending=False)
                                keep_idx = cand_idx[order[:k]]
                                new_mask = torch.zeros_like(prune_mask)
                                new_mask[keep_idx] = True
                                prune_mask = new_mask

                            gaussians.prune_points(prune_mask)
                            print(f"[PRUNE] removed {prune_mask.float().sum().item():.0f} / {prune_mask.numel()} "
                                  f"({100.0 * prune_mask.float().mean().item():.2f}%)")

                        # 清空统计，下一轮重新累计
                        cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])

                    # 轮计数并决定是否切回 densify
                    rounds_in_phase += 1
                    viewpoint_stack = scene.getTrainCameras().copy()
                    if rounds_in_phase >= rounds_prune and total_rounds_seen < 50:
                        _switch_phase("densify")
                        # 切回 densify 前可选做一次 reset_opacity，帮助新点尽快参与
                        gaussians.reset_opacity()

            # =====================[MOD-END]==============================

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 100 == 0:
                print(f"[ITER {iteration}] Warmup={total_rounds_seen < warmup_rounds} "
                      f"| Phase={phase} | RoundsSeen={total_rounds_seen} | #Gaussians: {gaussians.get_xyz.shape[0]}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    ...
    # 其余与原版相同，保持不变
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
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
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
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
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # ===================[MOD] CLI：Warmup + X/Y 轮调度 & 裁剪阈===================
    parser.add_argument("--warmup_rounds", type=int, default=0,
                        help="前 N 轮仅致密化（不收集统计、不裁剪）")
    parser.add_argument("--rounds_densify", type=int, default=1,
                        help="Warmup 后：做 X 轮完整训练视角的致密化")
    parser.add_argument("--rounds_prune", type=int, default=1,
                        help="Warmup 后：连续做 Y 轮（每轮末尾裁剪一次）")
    parser.add_argument("--prune_q_w", type=float, default=0.05,
                        help="按 ∑(α·T)/hits 的分位数阈，低于该分位视作低贡献")
    parser.add_argument("--prune_q_dom", type=float, default=0.05,
                        help="按 top1/hits 的分位数阈，低于该分位视作低主导")
    parser.add_argument("--prune_cap", type=float, default=0.10,
                        help="单轮最多删除比例上限，避免过激")
    # ===================[MOD-END]=======================================

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # ==== [MOD] 将关键超参挂到 pipe，便于在 training 内使用 ====
    dataset = lp.extract(args)
    opt     = op.extract(args)
    pipe    = pp.extract(args)

    pipe.warmup_rounds = args.warmup_rounds
    pipe.rounds_densify = args.rounds_densify
    pipe.rounds_prune   = args.rounds_prune
    pipe.prune_q_w      = args.prune_q_w
    pipe.prune_q_dom    = args.prune_q_dom
    pipe.prune_cap      = args.prune_cap
    pipe.collect_stats  = True  # 渲染器已支持统计则设 True

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
