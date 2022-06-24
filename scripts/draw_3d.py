from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import AllChem, PyMol
import os
from tqdm import tqdm
from PIL import Image


parser = ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

try:
    mol_viewer = PyMol.MolViewer()
except ConnectionRefusedError:
    raise ConnectionRefusedError(
        "[FATAL] Could not connect to PyMol server. Make sure to start it by running `pymol -R`"
    )

with open(args.input) as read_handle:
    all_smiles = [row.replace("\n", "") for row in read_handle]

if not os.path.exists(args.output):
    os.makedirs(args.output)

image_id = 0
with tqdm(total=len(all_smiles)) as progress_bar:
    for smiles in all_smiles:
        progress_bar.update(1)

        mol = Chem.MolFromSmiles(smiles)
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=10000)

        mol_viewer.ShowMol(mol)
        image: Image = mol_viewer.GetPNG(h=1000)

        image_id += 1
        image_path = "{}/{:06}.png".format(args.output, image_id)
        image.convert("RGB").save(image_path)
