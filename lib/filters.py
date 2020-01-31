from abc import ABCMeta, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class AbstractFilter(metaclass=ABCMeta):

    @abstractmethod
    def apply(self, smiles: str, reward: float) -> bool:
        raise NotImplementedError


class NonNanRewardFilter(AbstractFilter):

    def apply(self, smiles: str, reward: float) -> bool:
        return not np.isnan(reward)


class PositiveRewardFilter(AbstractFilter):

    def apply(self, smiles: str, reward: float) -> bool:
        return reward > 0


class MolecularWeightFilter(AbstractFilter):

    def apply(self, smiles: str, reward: float) -> bool:
        molecular_weight = ExactMolWt(Chem.MolFromSmiles(smiles))
        return 300 < molecular_weight < 500


class FilterFactory:

    NON_NAN_REWARD = "non_nan_reward"
    POSITIVE_REWARD = "positive_reward"
    MOLECULAR_WEIGHT = "molecular_weight"

    @staticmethod
    def create(filter_name: str) -> AbstractFilter:
        options = {
            FilterFactory.NON_NAN_REWARD: NonNanRewardFilter,
            FilterFactory.POSITIVE_REWARD: PositiveRewardFilter,
            FilterFactory.MOLECULAR_WEIGHT: MolecularWeightFilter
        }

        return options[filter_name]()
