from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

API_KEY = "" # Get this from Materials Project
CIF_DIR = "cif_files"

with MPRester(API_KEY) as m:
    materials = m.materials.summary.search(fields=['material_id', "symmetry", 'structure'])
    for material in materials:
        try:
            material_id = material.material_id
            crystal_system = material.symmetry.crystal_system
            spacegroup_number = material.symmetry.number
            structure = material.structure

            if structure:
                sga = SpacegroupAnalyzer(structure)
                structure = sga.get_conventional_standard_structure()

                a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
                alpha, beta, gamma = structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma

                cif_writer = CifWriter(structure)
                dir_path = os.path.join(CIF_DIR, str(crystal_system), str(spacegroup_number))
                os.makedirs(dir_path, exist_ok=True)
                file_name = f"{a:.2f}_{b:.2f}_{c:.2f}_{alpha:.2f}_{beta:.2f}_{gamma:.2f}_{material_id}.cif"
                cif_path = os.path.join(dir_path, file_name)
                cif_writer.write_file(cif_path)
                print(f"Downloaded CIF for material: {material_id}")
        except Exception as e:
            print(f"Error processing material {material_id}: {e}")
