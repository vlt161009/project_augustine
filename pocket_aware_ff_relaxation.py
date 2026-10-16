from rdkit import Chem
from rdkit.Chem import AllChem
import os
from pathlib import Path

ligand_dir = Path("C:\\augustine\\data\\output_diffdock_inferences\\highest_confidence")
protein_dir = Path("C:\\augustine\\data\\main_protease\\cleaned_files\\4YLU_cleaned.pdb")
output_dir = Path("C:\\augustine\\data\\minimized_ligands")

def supplierfunc(file_path):
    for file in Path(file_path).glob("*.sdf"):   
        supplier = Chem.SDMolSupplier(
                str(file),
                sanitize=True,
                removeHs=False,
                strictParsing=True
                )
        print(f"Found {len(supplier)} molecules in {file}.")
        for mol in supplier:
            if mol is not None:
                ligand_name = mol.GetProp('_Name')
                yield ligand_name, mol

def truncate_pocket(protein_dir, supplier_output, cutoff_distance = 6.0):
    protein = Chem.MolFromPDBFile(protein_dir, removeHs=False)
    if protein is None:
        print("Error: Failed to load protein PDB file.")
        return
    for ligand_name, mol in supplier_output:    
        print(f"Protein file located from {protein_dir}")


        ligand = mol

        if not ligand:
            print("Error: No valid ligand found in the SDF file.")
            return

        print("Ligand successfully loaded...")

        if ligand.GetNumConformers() == 0:
            AllChem.EmbedMolecule(ligand)
        
        print("Identifying residues to keep within cutoff distance...")
        
        
        residues = {}
        for atom in protein.GetAtoms():
            res_info = atom.GetPDBResidueInfo()
            if res_info is not None:
                res_id = (res_info.GetChainId(), res_info.GetResidueNumber(), res_info.GetResidueName())
                if res_id not in residues:
                    residues[res_id] = []
                residues[res_id].append(atom.GetIdx())

        residues_to_keep = set()
        for res_id, atom_indices in residues.items():
            keep_res = False
            for atom_idx in atom_indices:
                protein_pos = protein.GetConformer().GetAtomPosition(atom_idx)
                for ligand_atom in ligand.GetAtoms():
                    if ligand_atom.GetAtomicNum() == 1:
                        continue # skip hydrogens for distance check
                    ligand_pos = ligand.GetConformer().GetAtomPosition(ligand_atom.GetIdx())
                    if ligand_pos.Distance(protein_pos) <= cutoff_distance:
                        keep_res = True
                        break
                if keep_res:
                    break
            if keep_res:
                residues_to_keep.add(res_id)

        print(f"Identified {len(residues_to_keep)} residues within {cutoff_distance} Å of the ligand.")

        atoms_to_keep = set()
        for res_key in residues_to_keep:
            atoms_to_keep.update(residues[res_key])

        pocket_editable = Chem.EditableMol(protein)

        all_atom_indices = set(range(protein.GetNumAtoms()))
        atoms_to_delete = sorted(list(all_atom_indices - atoms_to_keep), reverse=True)

        for idx in atoms_to_delete:
            pocket_editable.RemoveAtom(idx)

        pocket_mol = pocket_editable.GetMol()
        
        orig_conf = protein.GetConformer()
        new_conf = Chem.Conformer(pocket_mol.GetNumAtoms())

        new_idx = 0
        for old_idx in range(protein.GetNumAtoms()):
            if old_idx in atoms_to_keep:
                new_conf.SetAtomPosition(new_idx, orig_conf.GetAtomPosition(old_idx))
                new_idx += 1

        pocket_mol.RemoveAllConformers()
        pocket_mol.AddConformer(new_conf)

        yield ligand, pocket_mol, ligand_name


def pocket_aware_ff_relaxation(truncated_pocket, max_iter=100):
    for ligand, pocket_mol, ligand_name in truncated_pocket:
        
        # 1. Ensure pocket has Hs too, so atom indexing remains perfectly predictable
        pocket_with_hs = Chem.AddHs(pocket_mol, addCoords=True)
        
        # 2. Track exactly how many atoms the ligand has (including its explicit Hs)
        num_ligand_atoms = ligand.GetNumAtoms() 
        
        # 3. Combine them. 
        # Ligand atoms are exactly indices [0] to [num_ligand_atoms - 1]
        # Pocket atoms are exactly indices [num_ligand_atoms] to the end
        complex_mol = Chem.CombineMols(ligand, pocket_with_hs)
        Chem.FastFindRings(complex_mol)
        complex_mol = Chem.AddHs(complex_mol)

        # Create the UFF force field
        ff = AllChem.UFFGetMoleculeForceField(complex_mol)
        if ff is None:
            print(f"Failed to load force field for {ligand_name}.")
            continue

        # 4. Freeze only the pocket atoms (everything past the ligand atoms)
        total_atoms = complex_mol.GetNumAtoms()
        for i in range(num_ligand_atoms, total_atoms):
            ff.UFFAddPositionConstraint(i, maxDispl=0.0, forceConstant=1e6)

        # Minimize the complex
        ff.Minimize(maxIts=max_iter)
        print(f"Minimized the energy of the complex for ligand: {ligand_name}")

        # 5. Extract minimized coordinates safely back to the original ligand
        complex_conf = complex_mol.GetConformer()
        ligand = Chem.AddHs(ligand)
        ligand_conf = ligand.GetConformer()
        # Update both heavy atoms and hydrogen positions identically
        for i in range(num_ligand_atoms):
            complex_pos = complex_conf.GetAtomPosition(i)
            ligand_conf.SetAtomPosition(i, complex_pos)

        yield ligand_name, ligand
        print(f"Minimized ligand: {ligand_name}")


def save_mol_to_file(file_path, minimized_ligands):
    os.makedirs(file_path, exist_ok=True)

    for ligand_name, ligand in minimized_ligands:
        safe_name = "".join(c for c in ligand_name if c.isalnum() or c in (' ', '.', '_')).rstrip() 
        file_name = f"{safe_name}_minimized.sdf"
        
        file_output = os.path.join(file_path, file_name)
        writer = Chem.SDWriter(file_output)
        writer.write(ligand)
        writer.close()

        print(f"Saved minimized ligand ({ligand_name}) to {file_output}")

def main():
    save_mol_to_file(output_dir, pocket_aware_ff_relaxation(truncate_pocket(protein_dir, supplierfunc(ligand_dir), cutoff_distance=6.0), max_iter=100))

if __name__ == "__main__":
    main()
