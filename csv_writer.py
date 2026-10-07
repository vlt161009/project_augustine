import os
from pathlib import Path
import pandas as pd

## This script is used to process CSV files in a specified directory.

#---Configuration---#
PROTEIN_DIR = Path("C:\\augustine\\main_protease\\cleaned_files\\4YLU_cleaned.pdb")
LIGAND_DIR = Path("C:\\augustine\\ligands\\converted_ligands\\test_ligands")
OUTPUT_CSV = Path("C:\\augustine\\diffdock_csv")

def build_ligand_from_folder(LIGAND_DIR):
    data = []
    for ligand_file in LIGAND_DIR:
        data.append({
            ligand_file.name:
        })