#!/usr/bin/env python3
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))
import os
import signal
import time
import json
import argparse
import cv2
import sys
import torch
import torchvision
import numpy as np
import csv

from gaussian_renderer import render_test, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration

# class for generating images based on viewinfo
class generateFrames:

    def __init__(self, args, dataset, pipe):
        self.args = args
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None
        self.count = 0
        self.num_gaussians = 0

    def initMediaServer(self):
        load_iteration = self.args.load_iteration
        self.gaussians = GaussianModel(self.dataset.sh_degree)

        model_path = self.args.model_path

        if load_iteration:
            if load_iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            else:
                loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(loaded_iter))
        else:
            print('''You should set the "--load_iteration" flag for evaluation!''')

        if loaded_iter:
            self.gaussians.load_ply(os.path.join(model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(loaded_iter),
                                                 "point_cloud.ply"))
        else:
            print('''You should set the "--load_iteration" flag for evaluation!''')

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f"当前模型中高斯点数量: {self.num_gaussians}")

    def generateImage(self, data):
        frameByteArray = ""

        remotedata = json.loads(data)
        width = remotedata["resolution_x"]
        height = remotedata["resolution_y"]

        if width != 0 and height != 0:
            try:
                yaw = remotedata["yaw"]
                pitch = remotedata["pitch"]
                radius = remotedata["radius"]
                do_training = bool(remotedata["train"])
                fovy = remotedata["fov_y"]
                fovx = remotedata["fov_x"]
                znear = remotedata["z_near"]
                zfar = remotedata["z_far"]
                self.pipe.convert_SHs_python = bool(remotedata["shs_python"])
                self.pipe.compute_cov3D_python = bool(remotedata["rot_scale_python"])
                keep_alive = bool(remotedata["keep_alive"])
                scaling_modifier = remotedata["scaling_modifier"]

                world_view_transform = torch.reshape(torch.tensor(remotedata["view_matrix"], dtype=torch.float32), (4, 4)).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]

                full_proj_transform = torch.reshape(torch.tensor(remotedata["view_projection_matrix"], dtype=torch.float32), (4, 4)).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]

                extrinsicMatrix = torch.reshape(torch.tensor(remotedata["extrinsicMatrix"], dtype=torch.float32), (4, 4)).cuda()

                custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)

                if custom_cam is not None:
                    render_pkg = render_test(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier)
                    net_image = render_pkg["render"]
                    net_image_bytes = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

                    self.count += 1
                    frameByteArray = net_image_bytes

            except Exception as e:
                print(e)

        return frameByteArray


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPU VRAM and rendering framerate test.')
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--models_root", type=str, default="output", help="Root directory containing model folders")

    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path if hasattr(args, "model_path") else "No model_path set")
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    json_str = '''{
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

    model_dirs = [os.path.join(args.models_root, d) for d in os.listdir(args.models_root)
                  if os.path.isdir(os.path.join(args.models_root, d))]

    os.makedirs("test", exist_ok=True)
    output_csv = os.path.join("test", "render_results.csv")

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 新增了两列：Model VRAM 和 Max Render VRAM
        writer.writerow(["Model", "Num Gaussians", "Avg Rendering Time (ms)", "Max FPS", "Model VRAM (MB)", "Max Render VRAM (MB)"])

    resolutions = [
        (1920, 1080),  # 2K
        (3840, 2160),  # 4K
    ]

    for model_path in model_dirs:
        model_name = os.path.basename(model_path)
        print(f"\n========== Rendering model: {model_name} ==========")
        args.model_path = model_path

        try:
            generate_frames = generateFrames(args, lp.extract(args), pp.extract(args))
            generate_frames.initMediaServer()

            torch.cuda.synchronize()
            model_mem_bytes = torch.cuda.memory_allocated()
            model_mem_mb = model_mem_bytes / (1024 ** 2)
            print(f"模型加载后显存占用: {model_mem_mb:.2f} MB")

            for res_x, res_y in resolutions:
                print(f"\n>>> 测试分辨率: {res_x}x{res_y}")
                res_data = json.loads(json_str)
                res_data["resolution_x"] = res_x
                res_data["resolution_y"] = res_y
                res_json_str = json.dumps(res_data)

                for trial in range(10):
                    print(f"第 {trial + 1} 次实验:")

                    torch.cuda.synchronize()
                    torch.cuda.reset_max_memory_allocated()
                    start_time = time.time()

                    for _ in range(100):  # 30帧测试
                        generate_frames.generateImage(res_json_str)
                        torch.cuda.synchronize()

                    end_time = time.time()
                    total_time = end_time - start_time
                    avg_time_ms = (total_time / 100.0) * 1000.0
                    max_fps = int(1000.0 / avg_time_ms)

                    max_memory_bytes = torch.cuda.max_memory_allocated()
                    max_memory_mb = max_memory_bytes / (1024 ** 2)

                    print(f"平均每帧: {avg_time_ms:.2f}ms，最大帧率: {max_fps} FPS，最大显存占用: {max_memory_mb:.2f} MB")

                    with open(output_csv, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([f"{model_name}_{res_x}x{res_y}", generate_frames.num_gaussians,
                                         f"{avg_time_ms:.2f}", max_fps, f"{model_mem_mb:.2f}", f"{max_memory_mb:.2f}"])

        except Exception as e:
            print(f"渲染模型 {model_name} 失败: {e}")

