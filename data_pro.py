import pandas as pd

# 读取 JSON 文件
file_path = 'data/2016.json'  # 替换为你的实际文件路径
df = pd.read_json(file_path)

# 筛选出 'Total' 列值为 0 的数据
filtered_df = df[df['Total'] == 0]

# 保存筛选结果到新的 JSON 文件
output_path = 'filtered_total_zero.json'  # 指定输出文件路径
filtered_df.to_json(output_path, orient='records', force_ascii=False)

print(f"筛选出的数据已保存到 {output_path}")
