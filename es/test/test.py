import cv2
import numpy as np
import os
import shutil
from moviepy.editor import ImageSequenceClip

# --- 1. 配置参数 ---
INPUT_VIDEO = 'true1.mp4'
OUTPUT_VIDEO = 'true1_transparent.webm'
TEMP_DIR = 'temp_frames'
# 要移除的颜色：纯白色 (BGR 格式，OpenCV 默认使用 BGR)
COLOR_TO_REMOVE = [255, 255, 255] 
# 假设原始视频的帧率是 30 FPS，请根据实际情况调整
FPS = 30 
# --------------------

def prepare_directory(directory):
    """清理并创建用于存储临时帧的目录"""
    if os.path.exists(directory):
        print(f"清理临时目录: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"创建临时目录: {directory}")

def extract_and_key(input_path, output_dir, color_to_remove):
    """从视频中抽帧，并将指定颜色抠图（转换为透明）"""
    if not os.path.exists(input_path):
        print(f"错误：找不到输入视频文件 '{input_path}'")
        return 0

    vidcap = cv2.VideoCapture(input_path)
    frame_count = 0
    
    print("--- 步骤 1/2: 开始抽帧和抠像 ---")

    while True:
        success, frame = vidcap.read()
        if not success:
            break

        # 1. 转换颜色空间：将 BGR 帧转换为 BGRA（BGR + Alpha）
        # 初始时，Alpha 通道都是 255 (不透明)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        # 2. 识别要移除的纯白色像素
        # np.all(frame == color_to_remove, axis=2) 创建一个布尔掩码，
        # 识别出所有 R, G, B 三个通道都匹配目标颜色的像素
        mask = np.all(frame == color_to_remove, axis=2)
        
        # 3. 应用透明度：将白色像素的 Alpha 通道设置为 0 (完全透明)
        # frame_bgra[mask, 3] 选取了所有在 mask 中为 True 的像素的第 3 个通道（Alpha 通道）
        frame_bgra[mask, 3] = 0

        # 4. 保存为带 Alpha 通道的 PNG 文件
        output_path = os.path.join(output_dir, f'frame_{frame_count:05d}.png')
        cv2.imwrite(output_path, frame_bgra)
        
        frame_count += 1

    vidcap.release()
    print(f"抽帧和抠像完成，共处理 {frame_count} 帧。")
    return frame_count

def encode_webm(frame_dir, output_path, fps):
    """将带有透明度的 PNG 帧序列重新编码为 WebM 视频"""
    print("--- 步骤 2/2: 开始编码为 WebM ---")
    
    # 1. 获取所有处理好的帧文件路径并排序
    png_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])

    if not png_files:
        print("错误：没有找到处理好的 PNG 帧，无法编码。")
        return

    # 2. 创建视频剪辑
    clip = ImageSequenceClip(png_files, fps=fps)

    # 3. 导出为 WebM (使用 VP9 编码器，并启用 Alpha 通道)
    # 'libvpx-vp9' 是 WebM 兼容且支持 Alpha 的推荐编码器
    # '-pix_fmt yuva420p' 确保像素格式支持 YUV + Alpha (A)
    try:
        clip.write_videofile(
            output_path, 
            codec='libvpx-vp9', 
            fps=fps, 
            preset='veryfast',
            ffmpeg_params=['-pix_fmt', 'yuva420p'],
            logger=None # 避免打印过多日志
        )
        print(f"\nWebM 视频 '{output_path}' 生成成功！")
        
    except Exception as e:
        print(f"\n错误：WebM 编码失败。请检查 moviepy/ffmpeg 设置。错误信息: {e}")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    
    prepare_directory(TEMP_DIR)
    
    # 执行抠像和抽帧
    frames_processed = extract_and_key(INPUT_VIDEO, TEMP_DIR, COLOR_TO_REMOVE)
    
    if frames_processed > 0:
        # 执行编码
        encode_webm(TEMP_DIR, OUTPUT_VIDEO, FPS)
    
    # 清理临时文件 (可选，但推荐)
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"临时目录 '{TEMP_DIR}' 已清理。")
    except Exception as e:
        print(f"清理临时目录时发生错误: {e}")