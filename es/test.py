import csv

def print_csv_samples(csv_path):
    # 读取所有行
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    total = len(reader)
    print(f"总行数: {total}")

    # 需要采样的位置比例
    ratios = [0.0, 0.25, 0.50, 0.75, 1.0]

    for r in ratios:
        idx = int(r * (total - 1))  # 计算目标行索引
        start = idx
        end = min(idx + 20, total)

        print("\n" + "=" * 40)
        print(f"采样位置 {int(r*100)}% （索引 {idx}）:")
        print("=" * 40)

        for row in reader[start:end]:
            print(row)

# 使用示例
print_csv_samples("bicycle.csv")
