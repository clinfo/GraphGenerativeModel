import os
import sys
from abc import ABCMeta, abstractmethod

import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


class AbstractCalculator(metaclass=ABCMeta):

    @abstractmethod
    def calculate(self, mol: Chem.Mol) -> float:
        raise NotImplementedError


class AbstractEnergyCalculator(AbstractCalculator, metaclass=ABCMeta):

    VALID_FORCE_FIELDS = []

    def __init__(self, force_field: str):
        self.force_field = force_field

        if self.force_field not in self.VALID_FORCE_FIELDS:
            raise ValueError("'{}' cannot handle a '{}' force field!".format(
                self.__class__.__name__, self.force_field)
            )


class RdKitEnergyCalculator(AbstractEnergyCalculator):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF = "mmff"

    VALID_FORCE_FIELDS = [FORCE_FIELD_UFF, FORCE_FIELD_MMFF]

    def calculate(self, mol: Chem.Mol) -> float:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        Chem.GetSymmSSSR(mol)

        force_field = self.get_force_field(mol)
        force_field.Initialize()
        force_field.Minimize()

        return force_field.CalcEnergy()

    def get_force_field(self, molecule: Chem.Mol) -> AllChem.ForceField:
        Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule, randomSeed=0)

        if self.force_field == self.FORCE_FIELD_MMFF:
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            return AllChem.MMFFGetMoleculeForceField(molecule, properties)

        if self.force_field == self.FORCE_FIELD_UFF:
            return AllChem.UFFGetMoleculeForceField(molecule)


class BabelEnergyCalculator(AbstractEnergyCalculator):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF94 = "mmff94"
    FORCE_FIELD_MMFF94S = "mmff94s"
    FORCE_FIELD_GAFF = "gaff"
    FORCE_FIELD_GHEMICAL = "ghemical"

    VALID_FORCE_FIELDS = [
        FORCE_FIELD_UFF, FORCE_FIELD_MMFF94, FORCE_FIELD_MMFF94S, FORCE_FIELD_GAFF, FORCE_FIELD_GHEMICAL
    ]

    def calculate(self, mol: Chem.Mol) -> float:
        smiles = Chem.MolToSmiles(mol)

        molecule = pybel.readstring("smi", smiles)
        force_field = pybel._forcefields[self.force_field]

        force_field.Setup(molecule.OBMol)
        return force_field.Energy()


class LogpCalculator(AbstractCalculator):

    def calculate(self, mol: Chem.Mol) -> float:
        return max(MolLogP(mol) - 5, 0)


class MwCalculator(AbstractCalculator):

    def calculate(self, mol: Chem.Mol) -> float:
        return rdMolDescriptors.CalcExactMolWt(mol)


class QedCalculator(AbstractCalculator):

    def calculate(self, mol: Chem.Mol) -> float:
        return 1 - qed(mol)


class SaCalculator(AbstractCalculator):

    def calculate(self, mol: Chem.Mol) -> float:
        Chem.GetSymmSSSR(mol)
        return sascorer.calculateScore(mol)


class CalculatorFactory:

    ENERGY_RDKIT_UFF = "energy_rdkit_uff"
    ENERGY_RDKIT_MMFF = "energy_rdkit_mmff"

    ENERGY_BABEL_UFF = "energy_babel_uff"
    ENERGY_BABEL_MMFF = "energy_babel_mmff"
    ENERGY_BABEL_MMFFS = "energy_babel_mmffs"
    ENERGY_BABEL_GAFF = "energy_babel_gaff"
    ENERGY_BABEL_GHEMICAL = "energy_babel_ghemical"

    LOG_P = "log_p"
    QED = "qed"
    SA = "sa"
    MW = "mw"

    @staticmethod
    def create(reward_type: str) -> AbstractCalculator:
        options = {
            CalculatorFactory.ENERGY_RDKIT_UFF: RdKitEnergyCalculator(RdKitEnergyCalculator.FORCE_FIELD_UFF),
            CalculatorFactory.ENERGY_RDKIT_MMFF: RdKitEnergyCalculator(RdKitEnergyCalculator.FORCE_FIELD_MMFF),

            CalculatorFactory.ENERGY_BABEL_UFF: BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_UFF),
            CalculatorFactory.ENERGY_BABEL_MMFF: BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_MMFF94),
            CalculatorFactory.ENERGY_BABEL_MMFFS: BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_MMFF94S),
            CalculatorFactory.ENERGY_BABEL_GAFF: BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_GAFF),
            CalculatorFactory.ENERGY_BABEL_GHEMICAL: BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_GHEMICAL),

            CalculatorFactory.LOG_P: LogpCalculator(),
            CalculatorFactory.MW: MwCalculator(),
            CalculatorFactory.QED: QedCalculator(),
            CalculatorFactory.SA: SaCalculator()
        }

        return options[reward_type]
