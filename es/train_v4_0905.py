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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # ==== [MOD]：新增——跨轮统计缓冲与“每 N 轮致密化”的控制 ====
    # 若渲染器支持统计，打开开关并初始化累加缓冲；否则保持为 False，不影响原逻辑
    pipe.collect_stats = getattr(pipe, "collect_stats", True)
    rounds_per_densify = getattr(pipe, "rounds_per_densify", 1)  # 每 N 轮触发一次致密化
    rounds_seen = 0
    eps = 1e-8

    def _alloc_cycle_stats(num_pts: int, device="cuda"):
        buf = torch.zeros(num_pts, device=device, dtype=torch.float32)
        # 依次返回：∑(α·T)、命中/覆盖计数、最大贡献者计数
        return buf.clone(), buf.clone(), buf.clone()

    cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])
    # ==== [MOD-END] ===========================================================

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

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_es(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_es, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
 
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            if iteration > opt.densify_until_iter:
                # ==== [MOD]：把本视角的三项统计累加到“跨轮累计缓冲”里（若渲染器已返回这些键） ====        
                if getattr(pipe, "collect_stats", False) and \
                   ("accum_weights" in render_pkg and "area_proj" in render_pkg and "area_max" in render_pkg):
                    # 若点数发生变化（例如上次 densify 后），重置缓冲大小
                    if cycle_sum_w.shape[0] != gaussians._xyz.shape[0]:
                        cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])
                    cycle_sum_w += render_pkg["accum_weights"]               # ∑(α·T)
                    cycle_hits  += render_pkg["area_proj"].to(torch.float32) # 命中像素计数 / 覆盖
                    cycle_top1  += render_pkg["area_max"].to(torch.float32)  # 最大贡献者像素计数
                    #print("Data collected")
                # ==== [MOD-END] =======================================================
        
                # 当前轮是否结束（用尽 viewpoint_stack）
                end_of_round = (len(viewpoint_stack) == 0)

                if end_of_round:
                    rounds_seen += 1  # 累加已完成的轮数

                do_round_densify = (
                    end_of_round 
                    and iteration < opt.densify_until_iter+15000
                    and rounds_seen > 0
                    and (rounds_seen % max(1, rounds_per_densify) == 0)
                )

                if do_round_densify:
                    # ——（可选）在调用原版 densify 之前做一个非常温和的全局删点，只用三项统计
                    if getattr(pipe, "collect_stats", False) and cycle_hits.sum() > 0:
                        mean_w    = cycle_sum_w / (cycle_hits + eps)
                        dom_share = cycle_top1  / (cycle_hits + eps)

                        # —— 按比例设置分位数；例如删掉强度处于底部 5% 且主导处于底部 5% 的点
                        q_w   = 0.05   # 低强度的比例阈
                        q_dom = 0.05   # 低主导的比例阈
                        th_w   = torch.quantile(mean_w, q_w)
                        th_dom = torch.quantile(dom_share, q_dom)

                        prune_mask = (mean_w <= th_w) & (dom_share <= th_dom)

                        # —— 可选：设置单次最多删掉的比例上限，避免过激（例如最多删 2%）
                        p_cap = 0.1
                        if prune_mask.any():
                            if prune_mask.float().mean() > p_cap:
                                k = max(1, int(p_cap * prune_mask.numel()))
                                # 只从候选里选出最弱的前 k 个（按 mean_w 升序）
                                cand_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(1)
                                _, order = torch.sort(mean_w[cand_idx], descending=False)
                                keep_idx = cand_idx[order[:k]]
                                new_mask = torch.zeros_like(prune_mask)
                                new_mask[keep_idx] = True
                                prune_mask = new_mask
                            gaussians.prune_points(prune_mask)

                    # 触发后清零“跨轮统计缓冲”，开始下一批轮次的累计
                    cycle_sum_w, cycle_hits, cycle_top1 = _alloc_cycle_stats(gaussians._xyz.shape[0])

                # 轮结束但未到第 N 轮时：不致密化，也不清零累计；仅重装视角栈
                if end_of_round:
                    viewpoint_stack = scene.getTrainCameras().copy()
                # ==== [MOD-END] ===============================================

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 100 == 0:  # 每100轮打印一次
                print(f"[ITER {iteration}] Current number of Gaussians: {gaussians.get_xyz.shape[0]}")



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
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

    # Report test and samples of training set
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # ==== [MOD]：新增 CLI —— 每 N 轮触发一次致密化 ====
    parser.add_argument("--rounds_per_densify", type=int, default=1,
                        help="Run densify/prune once every N full training-view rounds")
    # ==== [MOD-END] ==========================================================

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # ==== [MOD]：先提取 dataset/opt/pipe，再把 N 与 collect_stats 挂到 pipe 上（不改 training 接口） ====
    dataset = lp.extract(args)
    opt     = op.extract(args)
    pipe    = pp.extract(args)
    pipe.rounds_per_densify = args.rounds_per_densify
    # 可按需切换统计开关（默认 True）；若底层尚未实现统计返回，可设为 False
    pipe.collect_stats = True
    # ==== [MOD-END] ==========================================================

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

