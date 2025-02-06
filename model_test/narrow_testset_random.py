import h5py
import numpy as np

# 原始 HDF5 文件路径
test_File_Path = 'testset_cubic_fromCDDD_label_by_extinction.h5'
# 保存选中的样本的新 HDF5 文件路径
output_File_Path = 'testset_cubic_fromCDDD_label_by_extinction_100.h5'

# 读取 HDF5 文件
with h5py.File(test_File_Path, 'r') as ICSD_test:
    # 获取数据集
    stack_test = ICSD_test['data'][:]

    # 随机选择 100 条样本
    num_samples = 100
    if stack_test.shape[0] < num_samples:
        raise ValueError("样本数量不足 100 条！")

    random_indices = np.random.choice(stack_test.shape[0], num_samples, replace=False)
    sampled_data = stack_test[random_indices]

# 将选中的样本保存到新的 HDF5 文件
with h5py.File(output_File_Path, 'w') as output_file:
    # 创建数据集并保存数据
    output_file.create_dataset('data', data=sampled_data)

print(f"成功保存 {num_samples} 条随机样本到 {output_File_Path}")