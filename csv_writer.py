from pathlib import Path
import pandas as pd

## This script is used to process CSV files in a specified directory.

#---Configuration---#
PROTEIN_DIR = Path("C:\\augustine\\main_protease\\cleaned_files\\4YLU_cleaned.pdb")
LIGAND_DIR = Path("C:\\augustine\\ligands\\converted_ligands\\test_ligands")
OUTPUT_CSV = Path("C:\\augustine\\diffdock_csv")

def build_csv(protein_dir, ligand_dir, output_csv):
    data = {
            'Complex_Name': [],
            'Protein_Path': [],
            'Ligand_Path': []
        }
    
    for ligand in Path(ligand_dir).glob(".sdf"):
        complex_name = f"{protein_dir.stem}_{ligand.stem}"
        data['Complex_Name'].append(complex_name)
        data['Protein_Path'].append(str(protein_dir))
        data['Ligand_Path'].append(str(ligand))

    df = pd.DataFrame(data)
    print(df)

            
