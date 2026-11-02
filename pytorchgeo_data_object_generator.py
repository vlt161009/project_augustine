from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from pathlib import Path

LIG_FILE_PATH = "C:\\augustine\\data\\minimized\\minimized_lig"
POCK_FILE_PATH = "C:\\augustine\\data\\minimized\\truncated_pocket"

# Initialize ligand supplier
def lig_supplierfunc(file_path):
    for file in Path(file_path).glob("*.sdf"):   
        lig_supplier = Chem.SDMolSupplier(
                str(file),
                sanitize=True,
                removeHs=False,
                strictParsing=True
                )
        print(f"Found {len(lig_supplier)} molecules in {file}.")
        for mol in lig_supplier:
            if mol is not None:
                # Fallback to filename stem if _Name prop is missing
                ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else file.stem
                yield ligand_name, ligand_mol

# Initialize pocket supplier
def pocket_supplierfunc(file_path):
    for file in Path(file_path).glob("*.pdb"):
        pocket_supplier = Chem.rdmolfiles.PDBMolSupplier(
                str(file),
                sanitize=True,
                removeHs=False,
                strictParsing=True
                )
        print(f"Found {len(pocket_supplier)} molecules in {file}.")
        for mol in pocket_supplier:
            if mol is not None:
                # Fallback to filename stem if _Name prop is missing
                pocket_name = mol.GetProp('_Name') if mol.HasProp('_Name') else file.stem
                yield pocket_name, mol

def pocket_ligand_comb(pocket_supplier, lig_supplier):
    for pocket_name, pocket_mol in pocket_supplier:
        

