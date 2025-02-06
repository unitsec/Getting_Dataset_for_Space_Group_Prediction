import random
import numpy as np
import pathlib
import os
import sys
import h5py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
from dataset_generation.spacegroup_list import hm_to_space_group_number
from cctbx import uctbx, sgtbx, miller, xray, crystal
from cctbx.array_family import flex
from iotbx import cif
from scipy.stats import norm
import pandas as pd

def get_wanted_spacegroup(bravias):
    wanted_spacegroup = []
    for key, value in hm_to_space_group_number.items():
        if value[1] in bravias:
            wanted_spacegroup.append(key)
    return wanted_spacegroup

def calculate_cell_volume(a, b, c, alpha, beta, gamma):
    # 将角度转换为弧度
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # 计算体积
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad) ** 2 - np.cos(beta_rad) ** 2 - np.cos(gamma_rad) ** 2 + 2 * np.cos(alpha_rad) * np.cos(
            beta_rad) * np.cos(gamma_rad))
    return volume

def load_cif(file_path):
    # 确保 file_path 是字符串类型
    file_path = str(file_path)
    cif_object = cif.reader(file_path=file_path).model()
    block = list(cif_object.values())[0]
    return block

def get_unit_cell(block):
    a = float(block['_cell_length_a'])
    b = float(block['_cell_length_b'])
    c = float(block['_cell_length_c'])
    alpha = float(block['_cell_angle_alpha'])
    beta = float(block['_cell_angle_beta'])
    gamma = float(block['_cell_angle_gamma'])
    return uctbx.unit_cell((a, b, c, alpha, beta, gamma))

def get_space_group(block):
    sg_symbol = block['_symmetry_space_group_name_H-M']
    return sgtbx.space_group_info(symbol=sg_symbol).group()

def get_atom_sites(block):
    atom_sites = []

    atom_labels = block['_atom_site_type_symbol']
    atom_types = block['_atom_site_label']

    atom_fracs = zip(block['_atom_site_fract_x'], block['_atom_site_fract_y'], block['_atom_site_fract_z'])
    for label, atom_type, (x, y, z) in zip(atom_labels, atom_types, atom_fracs):
        atom_sites.append((label, atom_type, float(x), float(y), float(z)))
    return atom_sites

def calculate_structure_factors(unit_cell, space_group, atom_sites, miller_indices):
    scatterers = flex.xray_scatterer()
    for label, atom_type, x, y, z in atom_sites:
        scatterers.append(xray.scatterer(scattering_type=atom_type, site=(x, y, z)))
    crystal_symmetry = crystal.symmetry(unit_cell=unit_cell, space_group=space_group)
    structure = xray.structure(crystal_symmetry=crystal_symmetry)
    structure.add_scatterers(scatterers)
    
    miller_set = miller.set(crystal_symmetry=crystal_symmetry, indices=miller_indices, anomalous_flag=False)
    f_calc = miller_set.structure_factors_from_scatterers(xray_structure=structure).f_calc()
    return f_calc


def generate_diffraction_pattern(unit_cell, space_group, atom_sites, wavelength=1.5406, max_2theta=90):
    miller_indices = flex.miller_index()
    for h in range(-10, 11):
        for k in range(-10, 11):
            for l in range(-10, 11):
                if (h, k, l) != (0, 0, 0):
                    miller_indices.append((h, k, l))

    f_calc = calculate_structure_factors(unit_cell, space_group, atom_sites, miller_indices)

    theta_angles = []
    intensities = []
    for hkl, f in zip(miller_indices, f_calc.data()):
        if space_group.is_sys_absent(hkl):
            continue
        d_spacing = unit_cell.d(hkl)
        if d_spacing > 0:
            sin_theta_lambda = wavelength / (2 * d_spacing)
            if -1 <= sin_theta_lambda <= 1:
                theta = math.degrees(math.asin(sin_theta_lambda))
                two_theta = 2 * theta
                if two_theta <= max_2theta:
                    lorentz_factor = 1 / (math.sin(math.radians(two_theta)) * math.sin(math.radians(theta)))
                    intensity = abs(f)**2 * lorentz_factor
                    theta_angles.append(two_theta)
                    intensities.append(intensity)

    unique_data = {}
    for angle, intensity in zip(theta_angles, intensities):
        rounded_angle = round(angle, 6)
        if rounded_angle not in unique_data:
            unique_data[rounded_angle] = intensity

    unique_angles = sorted(unique_data.keys())
    unique_intensities = [unique_data[angle] for angle in unique_angles]

    return unique_angles, unique_intensities

def gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

bravias_list = ['P','SM','BCM','SO','BCO','FCO','BCO (Body-Centered)','ST','BCT','R','STrig','SH','SC','FCC','BCC']

#bravias_list = ['SC','FCC','BCC']


wanted_spacegroup = get_wanted_spacegroup(bravias_list)

data_path = os.path.expanduser('~/cifAI/db/fixed_crystal_structures')


def get_data_files(data_path, wanted_spacegroup, file_num, split_ratio):
    all_data_files = []
    all_data_files_train = []
    all_data_files_test = []
    data_dir = pathlib.Path(data_path)
    for sg in wanted_spacegroup:
        data_files = list(data_dir.glob(f'*_sg{sg}_*.cif'))
        #data_files = data_files1 + data_files2
        random.shuffle(data_files)
        data_files = data_files[:file_num]
        all_data_files.extend(data_files)
        split_point = int(len(data_files) * split_ratio)
        part1 = data_files[:split_point+1]
        part2 = data_files[split_point+1:]
        #if len(data_files_cleaned) < file_num:
        #    continue
        print(f'training samples number of {sg}: {len(part1)}')
        print(f'testing samples number of {sg}: {len(part2)}')
        all_data_files_train.extend(part1)
        all_data_files_test.extend(part2)
    random.shuffle(all_data_files)
    random.shuffle(all_data_files_train)
    random.shuffle(all_data_files_test)
    return all_data_files, all_data_files_train, all_data_files_test

def generate_dataset(data_files):
    samples = []
    labels = []
    failed_files = []
    two_theta_range = np.linspace(5, 90, 8192)
    sigma = 0.1
    for txt_file in data_files:
        try:
            #print(txt_file)
            block = load_cif(txt_file)
            unit_cell = get_unit_cell(block)
            space_group = get_space_group(block)
            atom_sites = get_atom_sites(block)
            two_theta_angles, intensities = generate_diffraction_pattern(unit_cell, space_group, atom_sites)
            #print("Generated diffraction pattern angles (2θ) and intensities:")
            #for angle, intensity in zip(two_theta_angles, intensities):
            #    print(f"2θ: {angle}, Intensity: {intensity}")

            #line_spectrum = np.zeros_like(two_theta_range)

            #for angle, intensity in zip(two_theta_angles, intensities):
                #line_spectrum += intensity * gaussian(two_theta_range, angle, 0.01)  # sigma 很小，以产生尖峰

            diffraction_pattern = np.zeros_like(two_theta_range)

            diffraction_pattern = np.zeros_like(two_theta_range)

            for angle, intensity in zip(two_theta_angles, intensities):
                diffraction_pattern += intensity * gaussian(two_theta_range, angle, sigma)

            #plt.figure(figsize=(12, 6))

            #plt.subplot(1, 2, 1)
            #plt.plot(two_theta_range, line_spectrum, color=(20/255,54/255,95/255),linewidth=3)
            #plt.title('Line Spectrum Before Convolution')
            #plt.xlabel('2θ (degrees)')
            #plt.ylabel('Intensity')
            #plt.xlim([5, 90])
            #plt.grid(False)

            #plt.subplot(1, 1, 1)
            #plt.plot(two_theta_range, diffraction_pattern, color=(20/255,54/255,95/255),linewidth=2)
            #plt.title('Diffraction Pattern After Convolution')
            #plt.xlabel('2θ (degrees)')
            #plt.ylabel('Intensity')
            #plt.xlim([5, 90])
            #plt.grid(False)

            #plt.tight_layout()
            #plt.show()

            norm_difpa = diffraction_pattern / np.max(diffraction_pattern)

            samples.append(norm_difpa)

            sg_num = float(os.path.basename(txt_file).split('_')[1][2:])
            labels.append(sg_num)
            #labels.append(float(sg_num))

            #plt.plot(two_theta_range,norm_difpa)
            #plt.show()
        except:
            print(f'generate fiffraction pattern failed for {txt_file}')
            failed_files.append(txt_file)
            continue

    return samples,labels,failed_files

def save_string_list_to_csv(string_list, filename):
    if len(string_list)>0:
        df = pd.DataFrame(string_list, columns=['file_path'])

        df.to_csv(filename, index=False)


def statistic(labels,filename):
    counts = {}
    for num in labels:
        counts[num] = counts.get(num, 0) + 1

    df = pd.DataFrame(list(counts.items()), columns=['Labels', 'Count'])

    df.sort_values(by='Labels', inplace=True)

    df.to_csv(filename, index=False)


def save_h5(file_path,samples,labels):
    with h5py.File(file_path, 'w') as f:
        dataset = np.hstack((samples, np.expand_dims(labels, axis=1)))
        f.create_dataset('data', data=dataset)

    print(f"New file has been saved in {file_path}")

print('Start Loading.......')

#x = getd(get_data_files(data_path,wanted_spacegroup)[0])
#labels = ['1.0','2.0','2.0','3.0','3.0','3.0','4.0']
#statistic(labels,'pyxtal_cctbx_dataset_statistics.csv')
#exit()
files, files_train, files_test = get_data_files(data_path, wanted_spacegroup, 100000, 0.95)
samples, labels,failed_files = generate_dataset(files)
save_string_list_to_csv(failed_files,'failed_datapath_pyxtal_cctbx_1019.csv')
statistic(labels,'pyxtal_cctbx_dataset_statistics.csv')
train_samples, train_labels,_ = generate_dataset(files_train)
statistic(train_labels,'pyxtal_cctbx_trainset_statistics.csv')
test_samples, test_labels,_ = generate_dataset(files_test)
statistic(test_labels,'pyxtal_cctbx_testset_statistics.csv')
print(np.unique(train_labels),len(np.unique(train_labels)))
print(np.unique(test_labels),len(np.unique(test_labels)))
file_path = "pyxtal_all_data_cctbx_1019.h5"
train_file_path = "pyxtal_training_data_cctbx_1019.h5"
test_file_path = "pyxtal_testing_data_cctbx_1019.h5"
save_h5(file_path,samples,labels)
save_h5(train_file_path,train_samples,train_labels)
save_h5(test_file_path,test_samples,test_labels)

