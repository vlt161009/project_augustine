from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import scipy.sparse as sp

LIG_FILE_PATH = "C:\\augustine\\data\\minimized\\minimized_lig"
POCKET_FILE_PATH = "C:\\augustine\\data\\minimized\\truncated_pocket"
DATA_OBJ_SAVE_DIR = "C:\\augustine\\data\\experimental_ligpro_graph_data_obj"

def combined_mol_generator():
    pocket_files = list(Path(POCKET_FILE_PATH).glob("*.pdb"))
    ligand_files = list(Path(LIG_FILE_PATH).glob("*.sdf"))

    print(f"Found {len(pocket_files)} PDB files and {len(ligand_files)} SDF files to process.")

    for pocket_file in pocket_files:
        pocket = Chem.MolFromPDBFile(
            str(pocket_file),
            sanitize=True,
            removeHs=False
        )
        if pocket is not None:
            pocket_name = pocket.GetProp('_Name') if pocket.HasProp('_Name') else pocket_file.stem
        for atom in pocket.GetAtoms():
            atom.SetBoolProp("is_ligand", False)

        for ligand_file in ligand_files:
            lig_supplier = Chem.SDMolSupplier(
                    str(ligand_file),
                    sanitize=True,
                    removeHs=False,
                    strictParsing=True
                    )
            for mol in lig_supplier:
                if mol is not None:
                    for atom in mol.GetAtoms():
                        atom.SetBoolProp("is_ligand", True)
                    ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ligand_file.stem
                    combined_mol = Chem.CombineMols(pocket, mol)
                    
                    yield pocket_name, combined_mol, ligand_name

