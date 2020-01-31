from abc import ABCMeta, abstractmethod

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class AbstractFilter(metaclass=ABCMeta):

    @abstractmethod
    def apply(self, smiles: str, reward: float) -> bool:
        raise NotImplementedError


class PositiveRewardFilter(AbstractFilter):

    def apply(self, smiles: str, reward: float) -> bool:
        return reward > 0


class MolecularWeightFilter(AbstractFilter):

    def apply(self, smiles: str, reward: float) -> bool:
        molecular_weight = ExactMolWt(Chem.MolFromSmiles(smiles))
        return 300 < molecular_weight < 500


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
