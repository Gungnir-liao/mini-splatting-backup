#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, torch, csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from gaussian_renderer import render, render_test, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration


class generateFrames:
    def __init__(self, args, dataset, pipe):
        self.args = args
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None
        self.num_gaussians = 0

    def initMediaServer(self):
        load_iteration = self.args.load_iteration
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        model_path = self.args.model_path
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))
        self.gaussians.load_ply(os.path.join(model_path,
                                             "point_cloud",
                                             f"iteration_{loaded_iter}",
                                             "point_cloud.ply"))
        bg_color = [1,1,1] if self.dataset.white_background else [0,0,0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f"当前模型中高斯点数量: {self.num_gaussians}")

    def generateImage(self, data: str, use_render_test=True):
        remotedata = json.loads(data)
        width, height = remotedata["resolution_x"], remotedata["resolution_y"]
        if width == 0 or height == 0: return ""
        world_view_transform = torch.tensor(remotedata["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        world_view_transform[:,1] *= -1; world_view_transform[:,2] *= -1
        full_proj_transform = torch.tensor(remotedata["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        full_proj_transform[:,1] *= -1
        custom_cam = MiniCam(width, height,
                             remotedata["fov_y"], remotedata["fov_x"],
                             remotedata["z_near"], remotedata["z_far"],
                             world_view_transform, full_proj_transform)
        scaling_modifier = remotedata["scaling_modifier"]
        if use_render_test:
            render_pkg = render_test(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier)
        else:
            render_pkg = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier)
        return render_pkg["render"]


def measure_renderer(gen: generateFrames, json_str: str, use_render_test: bool,
                     frames: int=100, trials: int=10):
    times, fps, vram = [], [], []
    for t in range(trials):
        torch.cuda.synchronize(); torch.cuda.reset_max_memory_allocated()
        start = time.time()
        for _ in range(frames):
            gen.generateImage(json_str, use_render_test=use_render_test)
            torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_time_ms = (elapsed/frames)*1000.0
        max_fps = int(1000.0/avg_time_ms) if avg_time_ms>0 else 0
        max_vram = torch.cuda.max_memory_allocated()/(1024**2)
        times.append(avg_time_ms); fps.append(max_fps); vram.append(max_vram)
    return np.mean(times), np.mean(fps), np.mean(vram)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Render benchmark")
    lp = ModelParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--models_root", type=str, default="output")
    args = parser.parse_args()
    safe_state(args.quiet)

    json_template = '''{
      "yaw": 0,
      "pitch": 0,
      "radius": 2.8293918840000014,
      "train": 1,
      "shs_python": 0,
      "rot_scale_python": 0,
      "scaling_modifier": 1,
      "resolution_x": 3840,
      "resolution_y": 2160,
      "fov_y": 1.0471975511965976,
      "fov_x": 1.5968513788836174,
      "z_far": 1100,
      "z_near": 0.009,
      "keep_alive": 1,
      "view_matrix": [
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, -2.8293918840000014, 1
      ],
      "view_projection_matrix": [
        0.9742785792574936, 0, 0, 0,
        0, -1.7320508075688774, 0, 0,
        0, 0, 1.000016363770249, 1,
        0, 0, 2.8114380362448035, 2.8293918840000014
      ],
      "extrinsicMatrix": [
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, -2.8293918840000014,
        0, 0, 0, 1
      ]
    }'''

    model_dirs = [os.path.join(args.models_root,d) for d in os.listdir(args.models_root)
                  if os.path.isdir(os.path.join(args.models_root,d))]
    os.makedirs("test", exist_ok=True)
    out_csv = os.path.join("test","render_results.csv")

    with open(out_csv,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow([
            "Model","Num Gaussians",
            "Render AvgTime(ms)","Render MaxFPS","Model VRAM(MB)","Render MaxVRAM(MB)",
            "RenderTest AvgTime(ms)","RenderTest MaxFPS","RenderTest MaxVRAM(MB)",
            "TimeRelChange(test_vs_render)","FPSRelChange(test_vs_render)","VRAMRelChange(test_vs_render)"
        ])

    resolutions=[(1920,1080),(3840,2160)]
    for model_path in model_dirs:
        model_name=os.path.basename(model_path)
        print(f"\n=== 渲染模型 {model_name} ===")
        args.model_path=model_path
        gen=generateFrames(args, lp.extract(args), pp.extract(args))
        gen.initMediaServer()
        model_mem_mb=torch.cuda.memory_allocated()/(1024**2)
        for rx,ry in resolutions:
            data = json.loads(json_template)
            data["resolution_x"], data["resolution_y"] = rx, ry
            jstr = json.dumps(data)
            print(f">>> 分辨率 {rx}x{ry}")
            # 測 render
            r_time,r_fps,r_vram=measure_renderer(gen,jstr,use_render_test=False)
            # 測 render_test
            rt_time,rt_fps,rt_vram=measure_renderer(gen,jstr,use_render_test=True)
            # 比率
            t_ratio=(rt_time-r_time)/r_time if r_time>0 else float("inf")
            f_ratio=(rt_fps-r_fps)/r_fps if r_fps>0 else float("inf")
            v_ratio=(rt_vram-r_vram)/r_vram if r_vram>0 else float("inf")
            with open(out_csv,"a",newline="") as f:
                writer=csv.writer(f)
                writer.writerow([
                    f"{model_name}_{rx}x{ry}",gen.num_gaussians,
                    f"{r_time:.2f}",f"{r_fps:.2f}",f"{model_mem_mb:.2f}",f"{r_vram:.2f}",
                    f"{rt_time:.2f}",f"{rt_fps:.2f}",f"{rt_vram:.2f}",
                    f"{t_ratio:.4f}",f"{f_ratio:.4f}",f"{v_ratio:.4f}"
                ])
