import os
from pathlib import Path
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, RDLogger

RDLogger.DisableLog('rdApp.*'
                    )

# main function
def main():
    #.smi file path
    filepath = "C:\\augustine\\data\\ligands\\raw_smiles_ligands\\test_ligands\\smiles_test_script.smi.txt"
    mol_from_file_path = supplierfunc(filepath)
    valid_mol = lipinskiverify(mol_from_file_path)
    converted_mol = conversionpipeline(valid_mol)
    writingmolecule(converted_mol)

    

def supplierfunc(file_path):
    supplier = Chem.SmilesMolSupplier(
            file_path,
            delimiter = " ",
            smilesColumn=0,
            nameColumn=1,
            titleLine=True,
            sanitize = True
            )
    for mol in supplier:
        if mol is not None:
            mol_obj = mol
            yield mol_obj

def process_batch(smiles_batch, output_dir, batch_sdf_path=None):

    saved_count = 0

    writer = Chem.SDWriter(str(batch_sdf_path)) if batch_sdf_path else None

    for smiles, zinc_id in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol.SetProp("_Name", zinc_id)
        mol.SetProp("zinc_id", zinc_id)

        # 1. Lipinski Rule of 5 Filter
        Violations = 0
        
        mw = Descriptors.MolWt(mol)
        hbd_amnt = Descriptors.NumHDonors(mol)
        hba_amnt = Descriptors.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)

        if mw >= 500: Violations +=1
        if hbd_amnt > 5: Violations +=1
        if hba_amnt > 10: Violations +=1
        if logp > 5: Violations +=1

        if Violations > 1:
            continue

        try:
            Chem.SanitizeMol(mol)      
            mol_obj_H = Chem.AddHs(mol)
            mol_obj_H.ClearComputedProps()
            #Implementation to fix future bug with ring finding during UFF minimization
            Chem.FastFindRings(mol_obj_H)
            
            status = AllChem.EmbedMolecule(mol_obj_H, AllChem.ETKDGv3())
            if status != 0:
                continue

            if writer:
                writer.write(mol_obj_H)
            else:
                out_path = os.path.join(output_dir, f"{zinc_id}.sdf")
                Chem.MoltoMolFile(mol_obj_H, out_path)
            saved_count += 1
        
        except Exception:
            continue

    if writer:
        writer.close()
    return saved_count


def writingmolecule(prepared_mol):
    for mol in prepared_mol:
        zinc_id = mol.GetProp("zinc_id") if mol.HasProp("zinc_id") else "unknown_id"  

        try:
            out_path = fr"C:\augustine\data\ligands\converted_ligands\{zinc_id}.sdf"
            Chem.MolToMolFile(mol, out_path)
            print(f"{zinc_id} has been successfully saved as .sdf!")
        except Exception as e:
            print(f"Failed to write file for {zinc_id}: {e}")

if __name__ == "__main__":
    main()


