import h5py
import numpy as np
import tensorflow as tf

File_Path = "pyxtal_training_data_cctbx_1019.h5"

ICSD = h5py.File(File_Path, 'r')
stack = ICSD['data'][:, :]

data = stack[:, :-1]
labels = stack[:, -1] - 1

cleaned_indices = []

valid_indices = []
for i in range(len(data)):
    if not np.any(np.isnan(data[i])) and not np.any(np.isinf(data[i])):
        valid_indices.append(i)
    else:
        cleaned_indices.append(i)

cleaned_data = data[valid_indices]
cleaned_labels = labels[valid_indices]

x_data = cleaned_data.reshape(-1, cleaned_data.shape[1], 1)
y_data_sg = tf.keras.utils.to_categorical(cleaned_labels, num_classes=230)

output_file = "cleaned_pyxtal_training_data.h5"
with h5py.File(output_file, 'w') as f:
    f.create_dataset('data', data=np.hstack((x_data.reshape(x_data.shape[0], -1), cleaned_labels.reshape(-1, 1))))

cleaning_log_file = "cleaning_log.txt"
with open(cleaning_log_file, 'w') as f:
    f.write("Indices of removed data due to NaN or Inf values:\n")
    for index in cleaned_indices:
        f.write(f"{index}\n")

print(f"Cleaning has finished and cleaned data has been saved as: {output_file}")
print(f"Cleaning Log has saved in: {cleaning_log_file}")

