"""
该脚本用于将期货合约的 unique_instrument_id 映射为 windcode 和 windcode_simp，并将映射关系保存到 CSV 文件中。

主要功能：
1. 从指定的 unique_instrument_ids.txt 文件中读取所有 unique_instrument_id。
2. 通过 unique_instrument_id_to_windcode 和 unique_instrument_id_to_windcode_simp 两个函数，将每个 unique_instrument_id 转换为 windcode 和 windcode_simp。
3. 将 unique_instrument_id、windcode 和 windcode_simp 的映射关系写入到 unique_instrument_ids_to_windcode.csv 文件中。
4. 如果映射文件已存在，则先删除，避免重复写入。

适用场景：
- 需要批量生成期货合约 ID 与 windcode 映射关系时使用。
- 便于后续数据处理和分析时查找合约代码对应关系。
"""

from DataMinkBasics import unique_instrument_id_to_windcode, unique_instrument_id_to_windcode_simp
import csv
import os

output_folder = './futures_data_mink'
unique_ids_file = os.path.join(output_folder, 'unique_instrument_ids.txt')

with open(unique_ids_file, 'r') as f:
    unique_ids = [line.strip() for line in f if line.strip()]

# 删除已有的映射文件以防重复写入
map_file_path = os.path.join(output_folder, 'unique_instrument_ids_to_windcode.csv')
if os.path.exists(map_file_path):
    os.remove(map_file_path)

for unique_id in unique_ids:
    # 将unique_instrument_id转换为windcode
    windcode = unique_instrument_id_to_windcode(unique_id)
    windcode_simp = unique_instrument_id_to_windcode_simp(unique_id)

    # 追加写入映射关系到csv文件
    write_header = not os.path.exists(map_file_path)
    with open(map_file_path, 'a', newline='') as map_file:
        writer = csv.writer(map_file)
        if write_header:
            writer.writerow(['unique_instrument_id', 'windcode', 'windcode_simp'])
        writer.writerow([unique_id, windcode, windcode_simp])

# End of file futures_data_mink/futures_data_mink_check_6.py