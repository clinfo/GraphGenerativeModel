import joblib

from lib.data_structures import Compound, CompoundBuilder


class MoleculeLoader(object):
    """
    Load, prepare and serve molecules from a *.jbl file
    """

    def __init__(self, file_path, threshold=0.9):
        """
        Constructor
        :param file_path: where the dataset lies (.jbl file)
        :param threshold: for valid molecules
        """
        self.file_path = file_path
        self.threshold = threshold

        self.dataset = self.load()

    def load(self):
        return joblib.load(self.file_path)

    def fetch(self, molecules_to_process=10):
        """
        Creates a molecule iterator for the dataset
        :param molecules_to_process: how many molecules to retrieve
        :return: rdKit.Mol
        """
        for molecule_index in range(molecules_to_process):

            builder = CompoundBuilder(
                bonds=self.dataset["dense_adj"][molecule_index],
                atoms=self.dataset["feature"][molecule_index],
                threshold=self.threshold,
            )

            builder.initialize_atoms()
            builder.initialize_bonds()

            molecule, bonds = builder.parse()
            yield Compound(molecule, bonds, builder.bonds_prediction)
