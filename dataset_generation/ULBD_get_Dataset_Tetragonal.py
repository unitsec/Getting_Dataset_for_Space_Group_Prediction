from cctbx import uctbx, sgtbx
from cctbx.array_family import flex
import math
import numpy as np
import pathlib
import h5py
import re
import sys
import os
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spacegroup_list import hm_to_space_group_number

def get_wanted_spacegroup(bravias):
    wanted_spacegroup = []
    for key, value in hm_to_space_group_number.items():
        if value[1] in bravias:
            wanted_spacegroup.append(key)
    return wanted_spacegroup

def get_lattice_range(data_path, sg_num):
    data_dir = pathlib.Path(data_path)
    data_files = list(data_dir.glob(f'*/{sg_num}/*.csv'))
    if len(data_files) == 0:
        return [],[]
    min_lattices = [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
    max_lattices = [0,0,0,0,0,0]
    for txt_file in data_files:
        filename = os.path.basename(txt_file)
        pattern = re.compile(r"(\d+\.\d+)_+(\d+\.\d+)_+(\d+\.\d+)_+(\d+\.\d+)_+(\d+\.\d+)_+(\d+\.\d+)")
        match = pattern.search(filename)
        if match:
            numbers = [float(match.group(i)) for i in range(1, 7)]
            numbers_array = np.array(numbers)  # 转换为 NumPy 数组
        else:
            print("No match found")
            continue
        min_lattices = list(map(min, min_lattices, numbers_array))
        max_lattices = list(map(max, max_lattices, numbers_array))
    return min_lattices, max_lattices

def save_h5(file_path,samples,labels):
    with h5py.File(file_path, 'w') as f:
        dataset = np.hstack((samples, np.expand_dims(labels, axis=1)))
        f.create_dataset('data', data=dataset)

    print(f"New file has been saved in {file_path}")

bravias_list = ['ST','BCT']
wanted_spacegroup = get_wanted_spacegroup(bravias_list)

def gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

two_theta_range = np.linspace(5, 90, 8192)

sigma = 0.1


print('start generating dataset......')
#samples = []
#labels = []
samples_train = []
labels_train = []
samples_validation = []
labels_validation = []
samples_test = []
labels_test = []
for sg_num in wanted_spacegroup:
    #if sg_num == '195':
    #    continue
    #min_lattices, max_lattices = get_lattice_range(mp_path,sg_num)
    #if len(min_lattices) == 0:
    #    continue
    #a_min = min_lattices[0] - 0.01
    #a_max = max_lattices[0] - 0.01
    a_min,a_max = 5.0,15.0
    sg_sym = hm_to_space_group_number[sg_num][0]
    print(f'generating data in space group {sg_num}: {sg_sym}')
    count = 0
    # ab_list = np.random.uniform(a_min,a_max,20)
    # c_list = np.random.uniform(a_min,a_max,20)
    for length1 in np.arange(a_min, a_max, 0.05):
        for length2 in np.arange(a_min, a_max, 0.05):
            if length2 == length1:
                continue
            a = length1
            b = length1
            c = length2
            alpha = 90
            beta = 90
            gamma = 90

            unit_cell = uctbx.unit_cell((a, b, c, alpha, beta, gamma))

            space_group_symbol = sg_sym
            space_group_info = sgtbx.space_group_info(symbol=space_group_symbol)
            space_group = space_group_info.group()

            miller_indices = flex.miller_index()
            for h in range(-10, 11):
                for k in range(-10, 11):
                    for l in range(-10, 11):
                        if (h, k, l) != (0, 0, 0):
                            miller_indices.append((h, k, l))

            wavelength = 1.5406
            max_2theta = 90
            theta_angles = []
            for hkl in miller_indices:
                if space_group.is_sys_absent(hkl):
                    continue
                d_spacing = unit_cell.d(hkl)
                if d_spacing > 0:
                    sin_theta_lambda = wavelength / (2 * d_spacing)
                    if -1 <= sin_theta_lambda <= 1:
                        theta = math.degrees(math.asin(sin_theta_lambda))
                        two_theta = 2 * theta
                        if two_theta <= max_2theta:
                            theta_angles.append(two_theta)

            rounded_angles = np.array(sorted(set(round(angle, 6) for angle in theta_angles)))

            samples_temp = []
            labels_temp = []
            for i in range(1,31):
                random_intensities = np.random.rand(len(rounded_angles))
                corrected_intensities = []
                for angle, random_intensity in zip(rounded_angles, random_intensities):
                    theta = angle / 2
                    lorentz_factor = 1 / (math.sin(math.radians(angle)) * math.sin(math.radians(theta)))
                    corrected_intensity = random_intensity * lorentz_factor
                    corrected_intensities.append(corrected_intensity)

                diffraction_pattern = np.zeros_like(two_theta_range)

                for angle, intensity in zip(rounded_angles, corrected_intensities):
                    diffraction_pattern += intensity * gaussian(two_theta_range, angle, sigma)

                norm_difpa = diffraction_pattern / np.max(diffraction_pattern)
                
                #samples.append(norm_difpa)
                #labels.append(float(sg_num))
                samples_temp.append(norm_difpa)
                labels_temp.append(float(sg_num))
                #plt.plot(two_theta_range,norm_difpa)
                #plt.show()
                count += 1
            samples_train.extend(samples_temp[0:20])
            labels_train.extend(labels_temp[0:20])
            samples_validation.extend(samples_temp[20:25])
            labels_validation.extend(labels_temp[20:25])
            samples_test.extend(samples_temp[25:30])
            labels_test.extend(labels_temp[25:30])
            

    print(f"{count} datas has been generated")

#save_h5('standard_dataset_tetragonal_random_30intlorentz.h5',samples,labels)
save_h5('standard_trainset_tetragonal_30intlorentz.h5',samples_train,labels_train)
save_h5('standard_validationset_tetragonal_30intlorentz.h5',samples_validation,labels_validation)
save_h5('standard_testset_tetragonal_30intlorentz.h5',samples_test,labels_test)
