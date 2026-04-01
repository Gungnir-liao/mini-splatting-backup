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


from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration

# class for generating images based on viewinfo
class generateFrames:
    
    def __init__(self,args,dataset,pipe):
        self.args = args  
        self.dataset = dataset 
        self.pipe = pipe 
        self.gaussians = None
        self.background = None
        self.count = 0
    
    def initMediaServer(self):
        load_iteration=self.args.load_iteration
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
        # 打印高斯点数量
        num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f"当前模型中高斯点数量: {num_gaussians}")

    def generateImage(self,data):
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
                world_view_transform[:,1] = -world_view_transform[:,1]
                world_view_transform[:,2] = -world_view_transform[:,2]
                full_proj_transform = torch.reshape(torch.tensor(remotedata["view_projection_matrix"], dtype=torch.float32), (4, 4)).cuda()
                full_proj_transform[:,1] = -full_proj_transform[:,1]
                #print("remotedata[extrinsicMatrix]:",remotedata["extrinsicMatrix"])
                extrinsicMatrix = torch.reshape(torch.tensor(remotedata["extrinsicMatrix"], dtype=torch.float32), (4, 4)).cuda()
                #print("extrinsicMatrix:",extrinsicMatrix)

                custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        
                net_image_bytes = None

                if custom_cam != None:

                    render_pkg = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier)
                    net_image = render_pkg["render"]

                    net_image_bytes = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    model_name = os.path.basename(os.path.normpath(self.args.model_path))
                    filename = f'test/{model_name}_{self.count}.png'
                    torchvision.utils.save_image(net_image, filename)

                    self.count += 1
                    #print("net_image_bytes:",net_image_bytes.shape)
                    frameByteArray = net_image_bytes
                  
            except Exception as e:
                print(e)

        return frameByteArray


if __name__ == "__main__":

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='GPU VRAM and rendering framerate test.')
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # 添加参数
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--models_root", type=str, default="output", help="Root directory containing model folders")
    
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 固定视角
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


    # 遍历所有模型文件夹
    model_dirs = [os.path.join(args.models_root, d) for d in os.listdir(args.models_root)
                  if os.path.isdir(os.path.join(args.models_root, d))]

    os.makedirs("test", exist_ok=True)

    for model_path in model_dirs:
        print(f"\n Rendering model: {model_path}")
        args.model_path = model_path  # 动态设置模型路径

        try:
            generate_frames = generateFrames(args, lp.extract(args), pp.extract(args))
            generate_frames.initMediaServer()
            generate_frames.generateImage(json_str)
        except Exception as e:
            print(f"Error rendering {model_path}: {e}")


