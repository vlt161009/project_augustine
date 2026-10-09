from pathlib import Path
import pandas as pd

## This script is used to process CSV files in a specified directory.

#---Configuration---#
PROTEIN_DIR = Path("C:\\augustine\\main_protease\\cleaned_files\\4YLU_cleaned.pdb")
LIGAND_DIR = Path("C:\\augustine\\ligands\\converted_ligands\\test_ligands")
OUTPUT_CSV = Path("C:\\augustine\\diffdock_csv\\test.csv") 

def build_csv(protein_dir, ligand_dir, output_csv):
    data = {
            'complex_name': [],
            'protein_path': [],
            'ligand_description': [],
            'protein_sequence': []
        }
    ligand_path = Path(ligand_dir)
    for ligand in ligand_path.iterdir():
        complex_name = f"{protein_dir.stem}_{ligand.stem}"
        data['complex_name'].append(complex_name)
        data['ligand_description'].append(str(ligand))
        data['protein_path'].append(str(protein_dir))
        data['protein_sequence'].append("")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

build_csv(PROTEIN_DIR, LIGAND_DIR, OUTPUT_CSV)

            
