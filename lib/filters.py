from abc import ABCMeta, abstractmethod
import os
import pandas as pd


from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, "../toxic_substructure/mcf.csv"))
_pains = pd.read_csv(
    os.path.join(_base_dir, "../toxic_substructure/wehi_pains.csv"),
    names=["smarts", "names"],
)
_filters = [
    Chem.MolFromSmarts(x) for x in _mcf.append(_pains, sort=True)["smarts"].values
]


class AbstractFilter(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        raise NotImplementedError


class NonZeroRewardFilter(AbstractFilter):
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        return reward != 0


class PositiveRewardFilter(AbstractFilter):
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        return reward > 0


class MolecularWeightFilter(AbstractFilter):
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        mol.UpdatePropertyCache()
        return 200 < ExactMolWt(mol) < 500


class ToxicSubsetFilter(AbstractFilter):
    def apply(self, mol: Chem.Mol, reward: float) -> bool:
        Chem.SanitizeMol(mol)
        return not any(mol.HasSubstructMatch(smarts) for smarts in _filters)
        # return True


class FilterFactory:

    NON_ZERO_REWARD = "non_zero_reward"
    POSITIVE_REWARD = "positive_reward"
    MOLECULAR_WEIGHT = "molecular_weight"
    TOXIC_SUBSTURCTURE = "toxic_substructure"

    @staticmethod
    def create(filter_name: str) -> AbstractFilter:
        options = {
            FilterFactory.NON_ZERO_REWARD: NonZeroRewardFilter,
            FilterFactory.POSITIVE_REWARD: PositiveRewardFilter,
            FilterFactory.MOLECULAR_WEIGHT: MolecularWeightFilter,
            FilterFactory.TOXIC_SUBSTURCTURE: ToxicSubsetFilter,
        }

        return options[filter_name]()
