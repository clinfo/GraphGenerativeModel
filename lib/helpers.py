import os
import time

from rdkit import Chem
from rdkit.Chem import Draw


class Sketcher(object):
    """
    Draws molecules
    """

    def __init__(self):
        self.location = None
        self.enabled = False

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
        if not self.location.endswith("/"):
            self.location += "/"

        if not os.path.exists(self.location):
            os.makedirs(location)

        self.enable()

    def get_file_name(self):
        """
        The file names are automatically generated based on the current timestamp
        :return: str
        """
        return self.location + '{0:.0f}'.format(time.time() * 10e6) + ".png"

    def draw(self, smiles):
        """
        :param smiles: str
        :return: None
        """
        if self.enabled:
            molecule = Chem.MolFromSmiles(smiles)
            filename = self.get_file_name()

            Draw.MolToFile(molecule, filename)
