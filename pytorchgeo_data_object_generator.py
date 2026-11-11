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

    # Load pocket & sets bool property "is_ligand" to 0 (false)
    for pocket_file in pocket_files:
        pocket = Chem.MolFromPDBFile(
            str(pocket_file),
            sanitize=True,
            removeHs=False
        )
        if pocket is not None:
            pocket_name = pocket.GetProp('_Name') if pocket.HasProp('_Name') else pocket_file.stem
        for atom in pocket.GetAtoms():
            atom.SetBoolProp("is_ligand", 0)
    # Load ligand & sets bool property "is_ligand" to 1 (true)
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
                        atom.SetBoolProp("is_ligand", 1)
                    ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ligand_file.stem
                    combined_mol = Chem.CombineMols(pocket, mol)
                    yield pocket_name, combined_mol, ligand_name

def generate_graph_data_obj(combined_mol, cutoff=5.0, num_eigenvectors=15):
    # Get number of atoms & conformer for combined_mol mol object
    num_atoms = combined_mol.GetNumAtoms()
    conf = combined_mol.GetConformer()

    x_atom_features = []
    graph_mask = []

    # Get values associated with atomic number, degree, formal charge, hybridization, and aromacity to create a atom features tensor (x)
    for atom in combined_mol.GetAtoms():
        atom_features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
        ]
        # Append Int Bool Property is_ligand to list graph_mask
        graph_mask.append(atom.HasProp("is_ligand"))
        x_atom_features.append(atom_features)
    
    # Retrieve coordinates of each atom into an numpy array, and then the array becomes pos_tensor
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    pos_tensor = torch.tensor(coords, dtype=torch.float)
    print(f"Position Matrix has been created ({pos_tensor.shape})")

    # create torch.tensor object for atom_features
    x_tensor = torch.tensor(x_atom_features, dtype=torch.float)
    print(f"Node Feature Matrix has been created ({x_tensor.shape})")
    # create torch.tensor object for graph mask feature ("is_ligand")
    graph_mask_tensor = torch.tensor(graph_mask, dtype=torch.bool)
    print(f"Graph Mask Matrix has been created ({graph_mask_tensor.shape})")

    dist_matrix = AllChem.Get3DDistanceMatrix(combined_mol, confId=conf.GetId())
    # Get the indices where the distance matrix is less than the 5 angstrom cutoff but greater than 0, to create edge index tensor
    adj_indices = np.where((dist_matrix < cutoff) & (dist_matrix > 0))
    edge_index_tensor = torch.tensor(adj_indices, dtype=torch.float)
    print(f"Edge Index Matrix has been created ({edge_index_tensor.shape})")

    # generate tensor for biophysical weights
    W = np.zeros((num_atoms, num_atoms), dtype=np.float32)


    ligand_np = np.array(graph_mask, dtype=bool)
    
    interface_mask = (ligand_np[:, None] != ligand_np[None, :]) & (dist_matrix < cutoff) & (dist_matrix > 0)
    W[interface_mask] = np.exp(-dist_matrix[interface_mask] / 2.0)

    for bond in combined_mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo = bond.GetBondTypeAsDouble()
        W[u, v] = bo
        W[v, u] = bo

    degrees = np.sum(W, axis=1)
    degrees[degrees == 0] = 1e-12
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(num_atoms) - D_inv_sqrt @ W @ D_inv_sqrt

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    geometric_scale = torch.tensor(eigenvalues[:num_eigenvectors], dtype=torch.float)
    print(f"Eigenvalue Matrix has been created ({geometric_scale.shape})")
    spectral_vectors = eigenvectors[:, :num_eigenvectors]

    spectral_tensor = torch.abs(torch.tensor(spectral_vectors, dtype=torch.float))
    print(f"Sign Invariant Eigenvalue Matrix has been created ({spectral_tensor.shape})")
    geometric_scale_edges = torch.tensor(dist_matrix[adj_indices], dtype=torch.float).unsqueeze(1)
    print(f"Geometric Edge Scale Matrix has been created ({geometric_scale_edges.shape})")
    return Data(
        x=x_tensor,
        pos=pos_tensor,
        edge_index=edge_index_tensor,
        edge_attr=geometric_scale_edges,
        is_ligand=graph_mask_tensor,
        spectral_tensor=spectral_tensor,
        geometric_scale=geometric_scale
    )

def main():
    stream = combined_mol_generator()
    processed_count = 0

    for pocket_name, combined_mol, ligand_name in stream:
        try:
            data_obj = generate_graph_data_obj(combined_mol=combined_mol)
            data_obj.y = torch.tensor([float('nan')], dtype=torch.float).unsqueeze(1)

            save_name = f"graph_{pocket_name}_{ligand_name}.pt"
            save_path = Path(DATA_OBJ_SAVE_DIR) / save_name

            torch.save(data_obj, str(save_path))
            processed_count +=1

            if processed_count % 10 == 0:
                print(f"Successfully constructed and saved {processed_count} data objects...")

        except Exception as e:
            print(f"Error executing graphs for {pocket_name} + {ligand_name}: {str(e)}")
            continue

    print(f"Execution complete. Total dataset objects saved: {processed_count}")

if __name__ == "__main__":
    main()
