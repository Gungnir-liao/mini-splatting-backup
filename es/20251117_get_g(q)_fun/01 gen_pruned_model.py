#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, csv, copy
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from gaussian_renderer import render_imp
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration


# =========================================================
#  Indoor / Outdoor 自动判定
# =========================================================
def guess_scene_type(model_name):
    model_name = model_name.lower()

    mipnerf360_outdoor = ["bicycle", "flowers", "garden", "stump", "treehill"]
    mipnerf360_indoor  = ["room", "counter", "kitchen", "bonsai"]
    tanks_outdoor      = ["truck", "train"]
    deepblend_indoor   = ["drjohnson", "playroom"]

    outdoor_scenes = set(mipnerf360_outdoor + tanks_outdoor)
    indoor_scenes  = set(mipnerf360_indoor + deepblend_indoor)

    for key in outdoor_scenes:
        if key in model_name:
            return "outdoor"

    for key in indoor_scenes:
        if key in model_name:
            return "indoor"

    return "indoor"  # 默认 indoor


# =========================================================
# Importance computation (multiple viewports)
# =========================================================
def compute_importance_from_viewports(gaussians, viewports, pipe, background, metric):
    torch.cuda.empty_cache()
    device = background.device
    N = gaussians._xyz.shape[0]

    imp_score = torch.zeros(N, device=device)
    accum_area_max = torch.zeros(N, device=device)

    for vp in tqdm(viewports, desc="Computing importance"):
        w, h = vp["resolution_x"], vp["resolution_y"]
        if w == 0 or h == 0:
            continue

        wv = torch.tensor(vp["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        wv[:,1] *= -1
        wv[:,2] *= -1
        proj = torch.tensor(vp["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        proj[:,1] *= -1

        cam = MiniCam(w, h, vp["fov_y"], vp["fov_x"], vp["z_near"], vp["z_far"], wv, proj)
        with torch.no_grad():
            pkg = render_imp(cam, gaussians, pipe, background)

        accum_weights = pkg["accum_weights"]
        area_proj     = pkg["area_proj"]
        area_max      = pkg["area_max"]

        accum_area_max += area_max
        mask = area_max != 0

        if metric == "outdoor":
            safe_area_proj = area_proj.clone()
            safe_area_proj[safe_area_proj == 0] = 1.0
            imp_score[mask] += (accum_weights[mask] / safe_area_proj[mask])
        else:  # default indoor metric
            imp_score += accum_weights

    imp_score[accum_area_max == 0] = 0
    return imp_score


# =========================================================
#  Top-p% pruning mask
# =========================================================
def get_mask_by_percentage(imp_score, keep_ratio):
    N = imp_score.shape[0]
    K = int(N * keep_ratio)
    if K <= 0:
        return torch.zeros(N, dtype=torch.bool, device=imp_score.device)
    if K >= N:
        return torch.ones(N, dtype=torch.bool, device=imp_score.device)

    # top K indices
    _, idx = torch.topk(imp_score, K, largest=True)
    mask = torch.zeros(N, dtype=torch.bool, device=imp_score.device)
    mask[idx] = True
    return mask


def build_pruned_gaussians(gaussians, mask_keep):
    """Create a standalone pruned GaussianModel compatible with the current codebase."""
    pruned = GaussianModel(gaussians.max_sh_degree)
    pruned.active_sh_degree = gaussians.active_sh_degree

    mask_keep = mask_keep.bool()

    pruned._xyz = nn.Parameter(gaussians._xyz[mask_keep].detach().clone().requires_grad_(True))
    pruned._features_dc = nn.Parameter(
        gaussians._features_dc[mask_keep].detach().clone().requires_grad_(True)
    )
    pruned._features_rest = nn.Parameter(
        gaussians._features_rest[mask_keep].detach().clone().requires_grad_(True)
    )
    pruned._scaling = nn.Parameter(
        gaussians._scaling[mask_keep].detach().clone().requires_grad_(True)
    )
    pruned._rotation = nn.Parameter(
        gaussians._rotation[mask_keep].detach().clone().requires_grad_(True)
    )
    pruned._opacity = nn.Parameter(
        gaussians._opacity[mask_keep].detach().clone().requires_grad_(True)
    )
    if gaussians.max_radii2D.numel() == mask_keep.shape[0]:
        pruned.max_radii2D = gaussians.max_radii2D[mask_keep].detach().clone()
    else:
        pruned.max_radii2D = torch.zeros((mask_keep.sum().item(),), device=mask_keep.device)
    return pruned


# =========================================================
#  Prune + render
# =========================================================
def prune_copy_and_render(gaussians, mask_keep, render_fn, cam, pipe, background):

    torch.cuda.synchronize()
    t0 = time.time()

    pruned = build_pruned_gaussians(gaussians, mask_keep)

    with torch.no_grad():
        out = render_fn(cam, pruned, pipe, background)

    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0), pruned, out


# =========================================================
#  Renderer class
# =========================================================
class BatchRenderer:
    def __init__(self, args, model_params, pipeline_params):
        self.args = args
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.pipe = pipeline_params
        self.gaussians = None
        self.background = None

    def load_model(self, model_root):
        self.gaussians = GaussianModel(self.model_params.sh_degree)

        iter_id = searchForMaxIteration(os.path.join(model_root, "point_cloud"))
        ply_path = os.path.join(model_root, "point_cloud", f"iteration_{iter_id}", "point_cloud.ply")

        print(f"Load model {model_root}, iter={iter_id}")
        self.gaussians.load_ply(ply_path)

        bg = [1,1,1] if self.model_params.white_background else [0,0,0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    def cam_from_vp(self, vp):
        w, h = vp["resolution_x"], vp["resolution_y"]
        wv = torch.tensor(vp["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        wv[:,1] *= -1
        wv[:,2] *= -1
        proj = torch.tensor(vp["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        proj[:,1] *= -1
        return MiniCam(w, h, vp["fov_y"], vp["fov_x"], vp["z_near"], vp["z_far"], wv, proj)


# =========================================================
#  Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--viewports_json", type=str, default="viewports.json")
    parser.add_argument("--csv_out", type=str, default="render_times.csv")
    parser.add_argument("--q_list", type=float, nargs="+", default=[50,55,60,65,70,75,80,85,90,95,100])
    parser.add_argument("--scenes", nargs="+", default=None, help="Optional subset of scene directory names to process.")

    args = parser.parse_args()
    safe_state(False)

    # load viewports
    viewports = json.load(open(args.viewports_json, "r"))

    # list models
    model_dirs = [os.path.join(args.models_root, d)
                  for d in os.listdir(args.models_root)
                  if os.path.isdir(os.path.join(args.models_root, d))]
    if args.scenes:
        selected = set(args.scenes)
        model_dirs = [d for d in model_dirs if os.path.basename(d) in selected]
        print(f"Selected scenes: {[os.path.basename(d) for d in model_dirs]}")

    renderer = BatchRenderer(args, lp.extract(args), pp.extract(args))

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "view_index", "q", "time_s", "remaining"])

        for model_root in model_dirs:
            model_name = os.path.basename(model_root)
            print(f"\n=== {model_name} ===")

            renderer.load_model(model_root)

            # --- 自动判断 indoor/outdoor
            scene_type = guess_scene_type(model_name)
            metric = "outdoor" if scene_type == "outdoor" else "indoor"
            print(f"Scene type detected: {scene_type} → using metric: {metric}")

            cache_stem = Path(args.viewports_json).stem
            imp_cache_path = os.path.join(model_root, f"importance_cache_{cache_stem}.pt")
            if os.path.exists(imp_cache_path):
                print(f"Loading cached importance from {imp_cache_path}")
                imp_score = torch.load(imp_cache_path, map_location="cuda")
            else:
                imp_score = compute_importance_from_viewports(
                    renderer.gaussians, viewports, renderer.pipe, renderer.background, metric
                )
                torch.save(imp_score.detach().cpu(), imp_cache_path)
                print(f"Saved importance cache to {imp_cache_path}")
                imp_score = imp_score.to(renderer.background.device)

            for q in args.q_list:
                torch.cuda.synchronize()
                t0 = time.time()
                mask = get_mask_by_percentage(imp_score, q/100)
                pruned = build_pruned_gaussians(renderer.gaussians, mask)
                torch.cuda.synchronize()
                t1 = time.time()
                t = t1 - t0
                print(f"Time taken to get mask and prune model: {t:.6f} seconds")   

                # --- 保存 prune 后模型
                save_path = os.path.join(model_root, f"point_cloud_pruned_{q}p.ply")
                pruned.save_ply(save_path)
                print(f"Saved pruned model {q}% to {save_path}")

if __name__ == "__main__":
    main()
