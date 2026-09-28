from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO, Select
import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

filepath = "C:\\augustine\\main_protease\\raw_downloads\\"
newfilepath = "C:\\augustine\\main_protease\\cleaned_files\\"

def pipeline():
    os.makedirs(newfilepath, exist_ok=True)
    for file in os.listdir(filepath):
        if file.endswith(".cif"):
            name = file.split(".")[0]
            protease_cleaner(name)

class ProteaseSelect(Select):
    def accept_residue(self, residue):    
        res_id = residue.get_id() 
        
        if res_id[0].strip() == "":
            return 1  
        else:
            return 0  


def protease_cleaner(name):
    parser = MMCIFParser()
    structure = parser.get_structure(name, os.path.join(filepath, f"{name}.cif"))

    for model in structure:
        ascii_counter = 65  
        for chain in model:
            if len(chain.id) > 1:
                new_id = chr(ascii_counter)
                chain.id = new_id
                ascii_counter += 1
                if ascii_counter == 91: 
                    ascii_counter = 97 

    output_path = os.path.join(newfilepath, f"{name}_cleaned.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ProteaseSelect())
    
 
    with open(output_path, "r") as f:
        lines = f.readlines()
    with open(output_path, "w") as f:
        for line in lines:

            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                f.write(line)

    print(f"Cleaned and prepared {name}.cif -> saved as {name}_cleaned.pdb.")

if __name__ == "__main__":
    pipeline()

