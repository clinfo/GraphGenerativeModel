from abc import ABCMeta, abstractmethod

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class AbstractFilter(metaclass=ABCMeta):

    @abstractmethod
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        raise NotImplementedError


class PositiveRewardFilter(AbstractFilter):

    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        return reward > 0


class MolecularWeightFilter(AbstractFilter):

    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        mol.UpdatePropertyCache()
        return 300 < ExactMolWt(mol) < 500


class FilterFactory:

    POSITIVE_REWARD = "positive_reward"
    MOLECULAR_WEIGHT = "molecular_weight"

    @staticmethod
    def create(filter_name: str) -> AbstractFilter:
        options = {
            FilterFactory.POSITIVE_REWARD: PositiveRewardFilter,
            FilterFactory.MOLECULAR_WEIGHT: MolecularWeightFilter
        }

        return options[filter_name]()
