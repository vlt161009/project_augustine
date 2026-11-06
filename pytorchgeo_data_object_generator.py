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

atom_features_tensor = []

# Initialize ligand supplier
def lig_supplierfunc(file_path):
    for file in Path(file_path).glob("*.sdf"):   
        lig_supplier = Chem.SDMolSupplier(
                str(file),
                sanitize=True,
                removeHs=False,
                strictParsing=True
                )
        
        for mol in lig_supplier:
            if mol is not None:
                for atom in mol.GetAtoms():
                    atom.SetBoolProp("is_ligand", True)
                # Fallback to filename stem if _Name prop is missing
                ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else file.stem
                yield ligand_name, mol

# Initialize pocket supplier
def pocket_supplierfunc(file_path):
    for file in Path(file_path).glob("*.pdb"):
        pocket = Chem.MolFromPDBFile(
                str(file),
                sanitize=True,
                removeHs=False
                )
        if pocket is not None:
            # Fallback to filename stem if _Name prop is missing
            pocket_name = pocket.GetProp('_Name') if pocket.HasProp('_Name') else file.stem
            yield pocket_name, pocket

def pocket_ligand_comb(pocket_supplier):
    for pocket_name, pocket in pocket_supplier:
        lig_stream = lig_supplierfunc(LIG_FILE_PATH)
        for ligand_name, ligand_mol in lig_stream:
            pocket_mol_obj = Chem.CombineMols(pocket, ligand_mol)
            print(f"pocket_mol is object: {pocket_mol_obj}")
            print(f"DEBUG: combined is type: {pocket_mol_obj}")
            yield pocket_name, pocket_mol_obj, ligand_name

def generate_node_features_tensor_x(combined_mol):
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        obj = pocket_mol_obj
        print(f"DEBUG: pocket_mol_obj has {obj.GetNumAtoms()} atoms.")
        # Generate 3D coordinates if not present
        if obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(obj)

        # Extract atom features
        for atom in obj.GetAtoms():
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
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        # Generate 3D coordinates if not present
        if pocket_mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(pocket_mol_obj)

        pos = []
        conf = pocket_mol_obj.GetConformer()
        for atom in pocket_mol_obj.GetAtoms():
            pos.append(list(conf.GetAtomPosition(atom.GetIdx())))
        
        pos_tensor = torch.tensor(pos, dtype=torch.float)
        return pos_tensor

# Draws the edge index tensor for the combined molecules, edges are drawn between atoms that are within 5 angstroms of each other
def generate_edge_index_tensor_edge_index(combined_mol):
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        # Generate 3D coordinates if not present
        if pocket_mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(pocket_mol_obj)

        conf = pocket_mol_obj.GetConformer()

        edge_index_tensor = AllChem.Get3DDistanceMatrix(pocket_mol_obj, confId=conf.GetId())

        edge_index_tensor_torch = torch.tensor(edge_index_tensor, dtype=torch.long).t().contiguous()
        return edge_index_tensor_torch

# We generate the lowest 15 eigenvectors of the distance matrix & return them as a tensor, where each node has a 15-dimensional feature vector.
def generate_lowest_eigenvector_tensor(combined_mol):
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        # Generate 3D coordinates if not present
        if pocket_mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(pocket_mol_obj)

        conf = pocket_mol_obj.GetConformer()
        distance_matrix = AllChem.Get3DDistanceMatrix(pocket_mol_obj, confId=conf.GetId())
        eigenvalues, eigenvectors = np.linalg.eigh(distance_matrix)
        lowest_eigenvectors = eigenvectors[:, :15]
        lowest_eigenvector_tensor = torch.tensor(lowest_eigenvectors, dtype=torch.float)

        return lowest_eigenvector_tensor

# Here we generate a tensor that is a boolean mask of each node, defining whether it is part of the ligand or the protein pocket. The ligand atoms are marked as 1, and the protein pocket atoms are marked as 0.
def generate_graph_mask_tensor(combined_mol):
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        # Generate 3D coordinates if not present
        if pocket_mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(pocket_mol_obj)

        graph_mask = []

        for atom in pocket_mol_obj.GetAtoms():
            if atom.HasProp("is_ligand"):
                graph_mask.append(1)
            else:
                graph_mask.append(0)

        graph_mask_tensor = torch.tensor(graph_mask, dtype=torch.bool)
        
        return graph_mask_tensor

# This function will generate Geometric Scale Tensors, that hold the euclidean distances between each connected node in the graph.
def generate_geometric_scale_tensor(combined_mol):
    for pocket_name, pocket_mol_obj, ligand_name in combined_mol:
        # Generate 3D coordinates if not present
        if pocket_mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(pocket_mol_obj)

        conf = pocket_mol_obj.GetConformer()
        distance_matrix = AllChem.Get3DDistanceMatrix(pocket_mol_obj, confId=conf.GetId())
        
        # Create a sparse matrix to hold the distances
        sparse_distance_matrix = sp.csr_matrix(distance_matrix)
        
        # Convert the sparse matrix to a dense tensor
        geometric_scale_tensor = torch.tensor(sparse_distance_matrix.todense(), dtype=torch.float)
        
        return geometric_scale_tensor

# This tensor will hold the regression target values for each target pic50 score for each mprotease(4YLU, 6LU7, 8C9P, 8D4N, 8E25, 8H5P, 9AUK, 9G0H, 23MB).
def generate_regression_target_tensor(target_values):
    # This script is for the generation of regression target tensors for the pic50 scores of unknown ligands against the different protease mutations. Because there is no data on the experimental group, we will fill the missing data with NaN values. The tensor will be of shape (num_samples, 1).
    if len(target_values) == 0:
        regression_target_tensor = torch.tensor([float('nan')], dtype=torch.float)
    else:
        regression_target_tensor = torch.tensor(target_values, dtype=torch.float).unsqueeze(1)
    
    return regression_target_tensor

def main():
    # File-system pre-count (highly memory efficient alternative to list counting)
    num_sdf = sum(1 for _ in Path(LIG_FILE_PATH).glob("*.sdf"))
    num_pdb = sum(1 for _ in Path(POCKET_FILE_PATH).glob("*.pdb"))
    print(f"Found {num_pdb} PDB files and {num_sdf} SDF files to process.")

    # Initialize our end-to-end generator pipeline
    combined_stream = pocket_ligand_comb(pocket_supplierfunc(POCKET_FILE_PATH))
    print("Pipeline initialized. Processing streams...")

    # Process everything on-the-fly
    processed_count = 0
    for pocket_name, pocket_mol_obj, ligand_name in combined_stream:
        processed_count += 1
        
        print(f"Processing combination {processed_count}: Pocket: {pocket_name}, Ligand: {ligand_name}, object: {pocket_mol_obj}")
        x = generate_node_features_tensor_x([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"Node features tensor x generated: {x}, with shape: {x.shape}")
        pos = generate_3d_coord_tensor_pos([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"3D coordinates tensor pos generated: {pos}, with shape: {pos.shape}")
        edge_index = generate_edge_index_tensor_edge_index([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"Edge index tensor generated: {edge_index}, with shape: {edge_index.shape}")
        geometric_scale = generate_geometric_scale_tensor([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"Geometric scale tensor generated: {geometric_scale}, with shape: {geometric_scale.shape}")
        graph_mask = generate_graph_mask_tensor([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"Graph mask tensor generated: {graph_mask}, with shape: {graph_mask.shape}")
        lowest_eigenvectors = generate_lowest_eigenvector_tensor([(pocket_name, pocket_mol_obj, ligand_name)])
        print(f"Lowest eigenvectors tensor generated: {lowest_eigenvectors}, with shape: {lowest_eigenvectors.shape}")
        
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} combinations...")

    print(f"Finished! Total combinations processed: {processed_count}")
        


if __name__ == "__main__":
    main()
