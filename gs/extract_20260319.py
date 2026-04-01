import pandas as pd
import argparse
import csv

def write_metric_to_csv(df, metric, method_name, csv_writer, decimal_places=2):
    """
    提取特定指标的数据，并将其写入到 csv_writer 中
    """
    render_col = f'render_{metric}'
    test_col = f'render_test_{metric}'
    
    # 确保 CSV 中包含这些列
    if render_col not in df.columns or test_col not in df.columns:
        print(f"警告: 输入文件中未找到 {render_col} 或 {test_col} 列。跳过 {metric}。\n")
        return

    # 提取需要的列并规范化场景名称（首字母大写）
    sub_df = df[['object', render_col, test_col]].copy()
    sub_df['object'] = sub_df['object'].str.capitalize()
    
    scenes = sub_df['object'].tolist()
    
    # 提取基础数据 (render)
    base_vals = sub_df[render_col].tolist()
    base_avg = sum(base_vals) / len(base_vals)
    
    # 提取加了 IO 的数据 (render_test)
    test_vals = sub_df[test_col].tolist()
    test_avg = sum(test_vals) / len(test_vals)
    
    # 计算 Diff (测试方法 - 基础方法)
    diff_val = test_avg - base_avg
    
    # 格式化字符串
    fmt = f"{{:.{decimal_places}f}}"
    
    # 构建表头
    diff_header = "Diff(dB)" if metric == "PSNR" else "Diff"
    header = ["Method"] + scenes + ["Avg.", diff_header]
    
    # 构建数据行
    base_row = [method_name] + [fmt.format(v) for v in base_vals] + [fmt.format(base_avg), "-"]
    test_row = [f"{method_name}+IO"] + [fmt.format(v) for v in test_vals] + [fmt.format(test_avg), fmt.format(diff_val)]
    
    # 写入 CSV 文件
    csv_writer.writerow([f"{metric} 评估表"]) # 写入指标标题
    csv_writer.writerow(header)               # 写入表头
    csv_writer.writerow(base_row)             # 写入基础方法数据
    csv_writer.writerow(test_row)             # 写入测试方法数据
    csv_writer.writerow([])                   # 写入空行，用于分隔不同指标表格

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="提取指标并直接导出到 CSV 文件。")
    parser.add_argument("--csv", type=str, required=True, help="输入的原始 CSV 文件路径")
    parser.add_argument("--name", type=str, required=True, help="基础方法的名字 (例如: 3DGS, MS)")
    parser.add_argument("--output", type=str, default="metrics_summary.csv", help="导出的目标 CSV 文件名 (默认: metrics_summary.csv)")
    
    args = parser.parse_args()
    
    try:
        # 读取原始 CSV 数据
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"读取输入 CSV 文件失败: {e}")
        exit(1)
        
    try:
        # 打开目标文件准备写入 (使用 utf-8-sig 防止 Excel 打开时中文乱码)
        with open(args.output, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 依次处理并写入三个指标
            write_metric_to_csv(df, 'PSNR', args.name, writer, decimal_places=2)
            write_metric_to_csv(df, 'SSIM', args.name, writer, decimal_places=4)
            write_metric_to_csv(df, 'LPIPS', args.name, writer, decimal_places=3)
            
        print(f"处理完成！数据已成功导出至: {args.output}")
        
    except Exception as e:
        print(f"写入目标 CSV 文件失败: {e}")