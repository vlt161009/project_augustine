from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# main function
def main():
    #.smi file path
    filepath = "C:\\augustine\\ligands\\raw_smiles_ligands\\test_ligands\\smiles_test_script.smi.txt"
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

def lipinskiverify(selected_mols):

    for mol in selected_mols:
        current_rdkit_obj = mol
        global zinc_id
        zinc_id = current_rdkit_obj.GetProp("_Name")
        mol.SetProp("zinc_id", zinc_id)

        Violations = 0
        
        mw = Descriptors.MolWt(current_rdkit_obj)
        hbd_amnt = Descriptors.NumHDonors(current_rdkit_obj)
        hba_amnt = Descriptors.NumHAcceptors(current_rdkit_obj)
        logp = Descriptors.MolLogP(current_rdkit_obj)

        if mw >= 500: Violations +=1
        if hbd_amnt > 5: Violations +=1
        if hba_amnt > 10: Violations +=1
        if logp > 5: Violations +=1

        if Violations <= 1:
            print(f"{zinc_id} PASSED (Violations: {Violations})")
            yield current_rdkit_obj
        else:
            print(f"{zinc_id} FAILED (Violations: {Violations})")

def conversionpipeline(screened_mol):
    for mol in screened_mol:

        mol_obj_H = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol_obj_H, AllChem.ETKDGv3())
        if status == 0:
            yield mol_obj_H
        else:
            zinc_id = mol_obj_H.GetProp("zinc_id") if mol_obj_H.HasProp("zinc_id") else "Unknown"
            print(f"Skipping 3D embedding failure for {zinc_id}")


def writingmolecule(prepared_mol):
    for mol in prepared_mol:
        zinc_id = mol.GetProp("zinc_id") if mol.HasProp("zinc_id") else "unknown_id"

        try:
            out_path = fr"C:\augustine\ligands\converted_ligands\{zinc_id}.sdf"
            Chem.MolToMolFile(mol, out_path)
            print(f"{zinc_id} has been successfully saved as .sdf!")
        except Exception as e:
            print(f"Failed to write file for {zinc_id}: {e}")

if __name__ == "__main__":
    main()


