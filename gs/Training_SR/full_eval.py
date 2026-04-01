import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../..')))

from argparse import ArgumentParser

# 数据集场景定义
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval")

# 将所有数据集参数设为可选 (required=False, 默认为 None)
parser.add_argument('--mipnerf360', "-m360", required=False, type=str, default=None)
parser.add_argument("--tanksandtemples", "-tat", required=False, type=str, default=None)
parser.add_argument("--deepblending", "-db", required=False, type=str, default=None)

args = parser.parse_args()

# 动态记录被激活的场景及其源数据路径
active_scenes = []
active_sources = []

if args.mipnerf360:
    for scene in mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes:
        active_scenes.append(scene)
        active_sources.append(os.path.join(args.mipnerf360, scene))

if args.tanksandtemples:
    for scene in tanks_and_temples_scenes:
        active_scenes.append(scene)
        active_sources.append(os.path.join(args.tanksandtemples, scene))

if args.deepblending:
    for scene in deep_blending_scenes:
        active_scenes.append(scene)
        active_sources.append(os.path.join(args.deepblending, scene))

# 如果没有提供任何数据集路径，给出提示
if not active_scenes and not (args.skip_training and args.skip_rendering and args.skip_metrics):
    print("Warning: No dataset paths provided. Please specify at least one of --mipnerf360, --tanksandtemples, or --deepblending.")

# ================= 1. Training =================
if not args.skip_training:
    # 使用这版代码中最基础的共同参数
    common_args = " --quiet --eval --test_iterations -1 "
    
    # 训练 MipNeRF360
    if args.mipnerf360:
        for scene in mipnerf360_outdoor_scenes:
            source = os.path.join(args.mipnerf360, scene)
            os.system(f"python train.py -s {source} -i images_4 -m {args.output_path}/{scene}{common_args}")    
        for scene in mipnerf360_indoor_scenes:
            source = os.path.join(args.mipnerf360, scene)
            os.system(f"python train.py -s {source} -i images_2 -m {args.output_path}/{scene}{common_args}")

    # 训练 Tanks and Temples
    if args.tanksandtemples:
        for scene in tanks_and_temples_scenes:
            source = os.path.join(args.tanksandtemples, scene)
            os.system(f"python train.py -s {source} -m {args.output_path}/{scene}{common_args}")
    
    # 训练 Deep Blending
    if args.deepblending:
        for scene in deep_blending_scenes:
            source = os.path.join(args.deepblending, scene)
            os.system(f"python train.py -s {source} -m {args.output_path}/{scene}{common_args}")

# ================= 2. Rendering =================
if not args.skip_rendering:
    render_common_args = " --quiet --eval --skip_train"
    # 动态遍历有效场景，并分别执行 7000 和 30000 步的渲染
    for scene, source in zip(active_scenes, active_sources):
        os.system(f"python render.py --iteration 7000 -s {source} -m {args.output_path}/{scene}{render_common_args}")
        os.system(f"python render.py --iteration 30000 -s {source} -m {args.output_path}/{scene}{render_common_args}")

# ================= 3. Metrics =================
if not args.skip_metrics:
    if active_scenes:
        scenes_string = " ".join([f'"{args.output_path}/{scene}"' for scene in active_scenes])
        os.system(f"python metrics.py -m {scenes_string}")
    else:
        print("Skip metrics: No scenes were processed.")