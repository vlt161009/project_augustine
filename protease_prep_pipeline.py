from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO, PDBParser, Select


def cif_to_pdb(cif_file, pdb_path, structure_id="structure"):
    #cif_file = cif_file location
    #pdb_path = output pdb file location
    # structure_id = structure id for the output pdb file
    parser = MMCIFParser()

    structure = parser.get_structure(structure_id, cif_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)

def protease_cleaner():
    # f
    