import h5py
import numpy as np
from collections import defaultdict

# 原始 HDF5 文件路径
test_File_Path = 'testset_cubic_fromULBD_label_by_extinction.h5'
# 保存选中的样本的新 HDF5 文件路径
output_File_Path = 'testset_cubic_fromULBD_label_by_extinction_100.h5'

# 读取 HDF5 文件
with h5py.File(test_File_Path, 'r') as ICSD_test:
    # 获取数据集
    stack_test = ICSD_test['data'][:]

    # 提取标签（假设标签在每行的最后一列）
    labels = stack_test[:, -1]
    unique_labels = np.unique(labels)

    # 根据标签分组样本
    samples_per_label = defaultdict(list)
    for sample in stack_test:
        label = sample[-1]
        samples_per_label[label].append(sample)

# 确定每个标签下选择的样本数量
num_samples_per_label = 100 // len(unique_labels)  # 确保总样本数为 100
if num_samples_per_label < 1:
    raise ValueError("无法从每个标签中选择样本，样本数量不足！")

# 随机选择每个标签的样本
sampled_data = []
for label in unique_labels:
    samples = samples_per_label[label]
    if len(samples) < num_samples_per_label:
        raise ValueError(f"标签 {label} 的样本数量不足 {num_samples_per_label} 条！")

    random_indices = np.random.choice(len(samples), num_samples_per_label, replace=False)
    sampled_data.extend(np.array(samples)[random_indices])

# 将选中的样本保存到新的 HDF5 文件
sampled_data = np.array(sampled_data)
with h5py.File(output_File_Path, 'w') as output_file:
    # 创建数据集并保存数据
    output_file.create_dataset('data', data=sampled_data)

print(f"成功保存 {len(sampled_data)} 条随机样本到 {output_File_Path}")