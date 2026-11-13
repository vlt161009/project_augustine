import os
from pathlib import Path
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, RDLogger

RDLogger.DisableLog('rdApp.*')


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

def stream_smiles_file(smi_path, delimiter=" ", smiles_col=0, name_col=1, has_header=True):

    with open(smi_path, "r", encoding="utf-8", errors="ignore") as f:
        if has_header:
            next(f, None)
        for idx, line in enumerate(f):
            parts = line.strip().split(delimiter)
            if len(parts) > max(smiles_col, name_col):
                yield (parts[smiles_col], parts[name_col])
            elif len(parts) == 1 and parts[0]:
                yield (parts[0], f"{smi_path.stem}_mol_{idx}")

def batched(iterable, chunk_size=1000):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk

def run_directory_pipeline(
        root_input_dir,
        output_dir,
        file_pattern="*.smi*",
        max_workers=None,
        chunk_size=1000,
        single_sdf_per_mol=True
):
    root_path = Path(root_input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    smi_files = sorted(root_path.rglob(file_pattern))
    print(f"Found {len(smi_files)} SMILES files across subdirectories.")

    workers = max_workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"Processing using {workers} CPU cores...")

    total_saved = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for file_idx, smi_file in enumerate(smi_files, start=1):
            futures = []
            mol_stream = stream_smiles_file(smi_file)

            for batch_idx, batch in enumerate(batched(mol_stream, chunk_size)):

                rel_folder = smi_file.parent.relative_to(root_path)
                target_folder = out_path / rel_folder
                target_folder.mkdir(parents=True, exist_ok=True)

                batch_sdf = None
                if not single_sdf_per_mol:
                    batch_sdf = target_folder / f"{smi_file.stem}_part{batch_idx}.sdf"

                futures.append(
                    executor.submit(process_batch, batch, str(target_folder), batch_sdf)
                )

            batch_sdf = None
            if not single_sdf_per_mol:
                batch_sdf = target_folder / f"{smi_file.stem}_part{batch_idx}.sdf"

            futures.append(
                executor.submit(process_batch, batch, str(target_folder), batch_sdf)
            )

        file_saved = sum(f.result() for f in futures)
        total_saved += file_saved
        print(f"[{file_idx}/{len(smi_files)}] {smi_file.name}: {file_saved} molecules saved.")

    print(f"\nPipeline complete! Total molecules converted: {total_saved}")
  
