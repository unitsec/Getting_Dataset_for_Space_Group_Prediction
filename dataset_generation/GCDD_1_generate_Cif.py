import os
import random
import time
from mpi4py import MPI
from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Directory to save the generated CIF files
output_dir = "crystal_structures"
os.makedirs(output_dir, exist_ok=True)

# List of common elements in nature
common_elements = [
    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
]

# common_elements = [
#     'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
#     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

# Number of structures to generate per space group
structures_per_space_group = 1000

# Range of space groups to consider (1 to 230 for 3D space groups)
space_group_range = range(1, 231)

crystal = pyxtal()

# Set to store unique structures
unique_structures = set()


# Function to generate a random crystal structure
def generate_structure(space_group, num_atoms):
    # Create a unique identifier for the structure
    structure_id = (space_group, tuple(sorted(zip(elements, num_atoms))))

    # Check if the structure is unique
    if structure_id in unique_structures:
        return None

    try:
        # print(elements, num_atoms)
        crystal.from_random(3, space_group, elements, num_atoms)

        # Convert to Pymatgen structure
        pmg_struc = crystal.to_pymatgen()

        # Use Pymatgen to analyze the space group
        analyzer = SpacegroupAnalyzer(pmg_struc)
        pmg_space_group = analyzer.get_space_group_number()

        # Check if the space group matches the intended space group
        if pmg_space_group != space_group:
            # print(f"Generated structure has higher symmetry space group {pmg_space_group} instead of {space_group}")
            return None

        unique_structures.add(structure_id)
        return crystal
    except Exception as e:
        # print(f"Failed to generate structure for space group {space_group}: {e}")
        return None


be = time.time()
num = 0
if rank != size-1 and int(len(space_group_range)*rank/size) != int(len(space_group_range)*(rank+1)/size):
    part_space_group_range = space_group_range[int(len(space_group_range)*rank/size):int(len(space_group_range)*(rank+1)/size)]
    for space_group in part_space_group_range:
        patience = 0
        generated_count = 0
        unique_structures = set()
        while generated_count < structures_per_space_group:
            # Adjust the number of elements and atoms
            num_elements = random.randint(1, 5)  # Limit to 1 or 2 elements
            elements = random.sample(common_elements, num_elements)
            num_atoms = [random.randint(1, 30) for _ in elements]  # Limit to 1 to 4 atoms per element

            structure = generate_structure(space_group, num_atoms)

            if structure:
                # Save the structure to a CIF file
                cif_filename = os.path.join(output_dir, f'structure_sg{space_group}_{generated_count + 1}.cif')
                structure.to_file(cif_filename)
                print(f"Saved structure to {cif_filename}")
                generated_count += 1
                num += 1
                patience = 0
            else:
                patience += 1
            if patience > 1000:
                print(f'generated space group {space_group} with {generated_count}, to next space group')
                break
    comm.send(num, dest = size-1, tag = 1)
elif rank == size-1:
    part_space_group_range = space_group_range[int(len(space_group_range)*rank/size):int(len(space_group_range)*(rank+1)/size)]
    for space_group in part_space_group_range:
        patience = 0
        generated_count = 0
        unique_structures = set()
        while generated_count < structures_per_space_group:
            # Adjust the number of elements and atoms
            num_elements = random.randint(1, 5)  # Limit to 1 or 2 elements
            elements = random.sample(common_elements, num_elements)
            num_atoms = [random.randint(1, 30) for _ in elements]  # Limit to 1 to 4 atoms per element

            structure = generate_structure(space_group, num_atoms)

            if structure:
                # Save the structure to a CIF file
                cif_filename = os.path.join(output_dir, f'structure_sg{space_group}_{generated_count + 1}.cif')
                structure.to_file(cif_filename)
                print(f"Saved structure to {cif_filename}")
                generated_count += 1
                num += 1
                patience = 0
            else:
                patience += 1
            if patience > 1000:
                print(f'generated space group {space_group} with {generated_count}, to next space group')
                break

    for j in range(size-1):
        t = comm.recv(source = j,tag = 1)
        print(j)
        num = num + t

    print("success num||expected num: ",num,len(space_group_range)*structures_per_space_group)
    print(time.time()-be)

print("Generation completed.")
