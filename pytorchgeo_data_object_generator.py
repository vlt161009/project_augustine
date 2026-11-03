from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from pathlib import Path
import numpy as np
import scipy.sparse as sp

LIG_FILE_PATH = "C:\\augustine\\data\\minimized\\minimized_lig"
POCKET_FILE_PATH = "C:\\augustine\\data\\minimized\\truncated_pocket"

atom_features_tensor = []
pos = []

# Initialize ligand supplier
def lig_supplierfunc(file_path):
    for file in Path(file_path).glob("*.sdf"):   
        lig_supplier = Chem.SDMolSupplier(
                str(file),
                sanitize=True,
                removeHs=False,
                strictParsing=True
                )
        
        ligand = lig_supplier[0]
        for atom in ligand.GetAtoms():
            atom.SetBoolProp("is_ligand", True)
        print(f"Found {len(lig_supplier)} molecules in {file}.")
        for mol in lig_supplier:
            if mol is not None:
                # Fallback to filename stem if _Name prop is missing
                ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else file.stem
                yield ligand_name, lig_supplier

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
        for ligand_name, ligand_mol in lig_supplier:
            pocket_mol_obj = Chem.CombineMols(pocket_mol, ligand_mol)
            yield pocket_name, pocket_mol_obj, ligand_name, ligand_mol

def generate_node_features_tensor_x(combined_mol):
    # Generate 3D coordinates if not present
    if combined_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(combined_mol)

    # Extract atom features
    for atom in combined_mol.GetAtoms():
        atom_features_tensor.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization(),
            atom.GetIsAromatic()
        ])
    
    x = torch.tensor(atom_features_tensor, dtype=torch.float)
    return x

def generate_3d_coord_tensor_pos(combined_mol):
    # Generate 3D coordinates if not present
    if combined_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(combined_mol)

    conf = combined_mol.GetConformer()
    for atom in combined_mol.GetAtoms():
        pos.append(list(conf.GetAtomPosition(atom.GetIdx())))
    
    pos_tensor = torch.tensor(pos, dtype=torch.float)
    return pos_tensor

# Draws the edge index tensor for the combined molecules, edges are drawn between atoms that are within 5 angstroms of each other
def generate_edge_index_tensor_edge_index(combined_mol):
    # Generate 3D coordinates if not present
    if combined_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(combined_mol)

    conf = combined_mol.GetConformer()

    edge_index_tensor = AllChem.Get3DDistanceMatrix(combined_mol, confId=conf.GetId())

    edge_index_tensor_torch = torch.tensor(edge_index_tensor, dtype=torch.long).t().contiguous()
    return edge_index_tensor_torch

# We generate the lowest 15 eigenvectors of the distance matrix & return them as a tensor, where each node has a 15-dimensional feature vector.
def generate_lowest_eigenvector_tensor(combined_mol):
    # Generate 3D coordinates if not present
    if combined_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(combined_mol)

    conf = combined_mol.GetConformer()
    distance_matrix = AllChem.Get3DDistanceMatrix(combined_mol, confId=conf.GetId())
    eigenvalues, eigenvectors = np.linalg.eigh(distance_matrix)
    lowest_eigenvectors = eigenvectors[:, :15]
    lowest_eigenvector_tensor = torch.tensor(lowest_eigenvectors, dtype=torch.float)

    return lowest_eigenvector_tensor

# Here we generate a tensor that is a boolean mask of each node, defining whether it is part of the ligand or the protein pocket. The ligand atoms are marked as 1, and the protein pocket atoms are marked as 0.
def generate_graph_mask_tensor(combined_mol):
    # Generate 3D coordinates if not present
    if combined_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(combined_mol)

    graph_mask = []

    for atom in combined_mol.GetAtoms():
        if atom.GetBoolProp("is_ligand"):
            graph_mask.append(1)
        else:
            graph_mask.append(0)

    graph_mask_tensor = torch.tensor(graph_mask, dtype=torch.bool)
    
    return graph_mask_tensor
