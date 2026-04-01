import json
import csv
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 全局配置
VIEWPORTS_JSON_PATH = "viewports_20260105.json"          # 包含坐标 x,y,z (所有模型共用)
MODELS_ROOT = "../../ms-eval1119"                          # 模型文件夹的根目录，请修改为您实际的路径

# 文件名模式
# 假设每个模型目录下都有这两个文件：
# 1. render_times_{model_name}.csv
# 2. fit_params_per_view.csv
FITTED_PARAMS_FILENAME = "fit_params_per_view.csv" 

# 参数配置
WARMUP_SKIP = 3      # 跳过前3次渲染
BASE_QUALITY = 100   # 基准质量 q=100
PREFERRED_METHOD = "Polynomial" # 仿真器首选的拟合函数类型 (保持数学形式统一)
# ===========================================

def load_viewports_map():
    """加载 viewports.json 并构建索引到坐标的映射"""
    print("1. Loading Viewports Data (Coordinates)...")
    try:
        with open(VIEWPORTS_JSON_PATH, 'r', encoding='utf-8') as f:
            viewports = json.load(f)
    except FileNotFoundError:
        print(f"Error: {VIEWPORTS_JSON_PATH} not found.")
        return None

    # 建立 view_index -> position 映射
    idx_to_pos = {}
    for idx, vp in enumerate(viewports):
        pos = vp.get("position")
        if pos:
            idx_to_pos[idx] = pos
            
    print(f"   Loaded {len(idx_to_pos)} coordinates.")
    return idx_to_pos

def process_single_model(model_name, model_dir, idx_to_pos):
    """处理单个模型的数据并导出 CSV"""
    print(f"\n=== Processing Model: {model_name} ===")
    
    # 构建该模型的文件路径
    render_csv_path = os.path.join(model_dir, f"render_times_{model_name}.csv")
    fitted_params_path = os.path.join(model_dir, FITTED_PARAMS_FILENAME)
    
    # 输出路径 (保存在当前脚本同一目录)
    output_sim_field_path = f"simulation_cost_field_{model_name}.csv"

    # --- 步骤 2: 计算基准耗时 (Base Cost at q=100) ---
    print(f"   [Step 2] Reading Render Times: {render_csv_path}")
    try:
        df_render = pd.read_csv(render_csv_path)
        # 过滤 Warm-up
        df_render = df_render[df_render['repeat_idx'] >= WARMUP_SKIP]
        
        # 检查是否存在基准质量数据
        if BASE_QUALITY not in df_render['q'].unique():
            print(f"   ⚠️ Warning: Base quality q={BASE_QUALITY} not found in {model_name}. Skipping...")
            return

        # 只保留基准质量 q=100 的数据，计算均值和方差
        df_base = df_render[df_render['q'] == BASE_QUALITY].groupby(['model_name', 'view_index']).agg({
            'render_time_s': ['mean', 'std']
        }).reset_index()
        
        # 展平列名
        df_base.columns = ['model_name', 'view_index', 'base_cost_mean', 'base_cost_std']
        
    except FileNotFoundError:
        print(f"   ❌ Error: Render CSV not found at {render_csv_path}")
        return

    # --- 步骤 3: 加载拟合参数 (G(q) Params) ---
    print(f"   [Step 3] Reading Fitted Params: {fitted_params_path}")
    try:
        df_params = pd.read_csv(fitted_params_path)
        
        # 过滤: 只保留我们想要的函数类型
        if PREFERRED_METHOD:
            df_params = df_params[df_params['method'] == PREFERRED_METHOD].copy()
            
        # 检查是否去重 (每个 view_index 只能有一条参数)
        # 如果有重复，取 R2 最高的
        df_params = df_params.sort_values('r2', ascending=False).drop_duplicates(subset=['model_name', 'view_index'])
        
    except FileNotFoundError:
        print(f"   ❌ Error: Fitted Params CSV not found at {fitted_params_path}")
        return

    # --- 步骤 4: 大合并 (Merge All) ---
    print("   [Step 4] Merging Data...")
    
    # 合并 参数 + 基准耗时
    if not df_base.empty:
        merged_df = pd.merge(df_params, df_base, on=['model_name', 'view_index'], how='left')
    else:
        print("   ❌ Error: Base cost data is empty.")
        return

    # 合并 坐标
    def get_xyz(idx):
        return pd.Series(idx_to_pos.get(idx, [None, None, None]))

    merged_df[['x', 'y', 'z']] = merged_df['view_index'].apply(get_xyz)
    
    # 清洗掉没有坐标的数据
    merged_df = merged_df.dropna(subset=['x', 'y', 'z'])

    # --- 步骤 5: 导出最终仿真用表 ---
    print(f"   [Step 5] Exporting to {output_sim_field_path}...")
    
    # 整理列顺序
    final_cols = [
        'model_name', 'view_index', 
        'x', 'y', 'z', 
        'base_cost_mean', 'base_cost_std',
        'method', 'param1', 'param2', 'param3', 
        'r2'
    ]
    
    # 确保列存在
    available_cols = [c for c in final_cols if c in merged_df.columns]
    final_df = merged_df[available_cols]
    
    # 排序
    final_df = final_df.sort_values('view_index')
    
    final_df.to_csv(output_sim_field_path, index=False)
    print(f"   ✅ Done! Rows: {len(final_df)}")

def main():
    # 1. 加载所有模型共用的坐标信息
    idx_to_pos = load_viewports_map()
    if not idx_to_pos:
        return

    # 2. 遍历模型目录
    if not os.path.exists(MODELS_ROOT):
        print(f"Error: Models root directory '{MODELS_ROOT}' does not exist.")
        print("Please edit 'MODELS_ROOT' in the script configuration.")
        return

    # 获取所有子目录作为模型列表
    model_dirs = [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))]
    
    if not model_dirs:
        print(f"No subdirectories found in {MODELS_ROOT}.")
        return

    print(f"\nFound {len(model_dirs)} models: {model_dirs}")

    # 3. 对每个模型执行处理
    for model_name in model_dirs:
        model_dir = os.path.join(MODELS_ROOT, model_name)
        process_single_model(model_name, model_dir, idx_to_pos)

if __name__ == "__main__":
    try:
        import pandas
        main()
    except ImportError:
        print("Error: pandas not installed.")