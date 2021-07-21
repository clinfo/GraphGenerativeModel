import os
import time

from rdkit import Chem
from rdkit.Chem import Draw
import logging


class Sketcher(object):
    """
    Draws molecules
    """

    def __init__(self, experiment_name, path_to_save=None):
        self.enabled = False
        self.experiment_name = experiment_name
        if path_to_save is None:
            path_dir = os.path.abspath('.')
            path_to_save = os.path.join(path_dir, 'molecule_img', experiment_name)
        self.set_location(path_to_save)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def set_location(self, location):
        """
        Setting the location automatically enables the Sketcher
        :param location: Should be a folder, not a file name
        :return: None
        """
        self.location = location
        if not os.path.exists(self.location):
            os.makedirs(location)

        self.enable()

    def get_file_name(self, num_molecule, smiles, depth):
        """
        The file names are automatically generated based on the current timestamp
        :return: str
        """
        path_dir = os.path.join(
            self.location,
            f'molecule_{num_molecule}' if num_molecule is not None else '',
            '_'.join([str(t) for t in time.localtime()[0:5]])
        )

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        return os.path.join(path_dir, f"depth_{depth}_smiles_{smiles}.png")


    def generate_png(self, mol, depth, score):


        d2d = Draw.MolDraw2DCairo(400, 400)
        legend = f"nb bonds={depth} score={score}"
        mol.Compute2DCoords()
        d2d.DrawMolecule(mol, legend=legend)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    def draw(self, smiles, num_molecule=None, score=None):
        """
        :param smiles: str
        :return: None
        """
        if self.enabled:
            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                logging.debug("Cannot draw molecule: {}".format(smiles))
                return
            depth = len(molecule.GetBonds())

            filename = self.get_file_name(num_molecule, smiles, depth)
            img = self.generate_png(molecule, depth, score)

            # save png to file
            with open(filename, 'wb') as png_file:
                png_file.write(img)

    def draw_from_eval(self, output):
        for i, stat in enumerate(output['stat']):
            for smiles, score in zip(stat['smiles'], stat['score']):
                self.draw(smiles, i, score)
