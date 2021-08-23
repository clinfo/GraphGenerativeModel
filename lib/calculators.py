import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Union

import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

import importlib
import numpy as np

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


class AbstractCalculator(metaclass=ABCMeta):

    enable = True

    @abstractmethod
    def calculate(self, mol: Chem.Mol) -> float:
        raise NotImplementedError


class AbstractEnergyCalculator(AbstractCalculator, metaclass=ABCMeta):

    VALID_FORCE_FIELDS = []
    RECALCULATION_LOOPS = 5

    def __init__(self, force_field: str):
        self.force_field = force_field

        if self.force_field not in self.VALID_FORCE_FIELDS:
            raise ValueError(
                "'{}' cannot handle a '{}' force field!".format(
                    self.__class__.__name__, self.force_field
                )
            )


class RdKitEnergyCalculator(AbstractEnergyCalculator):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF = "mmff"

    VALID_FORCE_FIELDS = [FORCE_FIELD_UFF, FORCE_FIELD_MMFF]

    def calculate(self, mol: Chem.Mol) -> float:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol = Chem.AddHs(mol)
        Chem.GetSymmSSSR(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=self.RECALCULATION_LOOPS)
        values = []
        for cid in cids:
            force_field = self.get_force_field(mol, confId=cid)
            force_field.Initialize()
            force_field.Minimize()

            values.append(force_field.CalcEnergy())
        return min(values)

    def get_force_field(self, molecule: Chem.Mol, confId) -> AllChem.ForceField:
        Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)

        if self.force_field == self.FORCE_FIELD_MMFF:
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            return AllChem.MMFFGetMoleculeForceField(molecule, properties, confId)

        if self.force_field == self.FORCE_FIELD_UFF:
            return AllChem.UFFGetMoleculeForceField(molecule, confId)


def read_histogram_file(filename):
    hist_data = []
    sum_data = 0
    for line in open(filename):
        bin_data = line.split("\t")
        if len(bin_data) > 0:
            if bin_data[0] == "":
                bin_begin = -np.inf
            else:
                bin_begin = float(bin_data[0])
            if bin_data[1] == "":
                bin_end = np.inf
            else:
                bin_end = float(bin_data[1])
            v = float(bin_data[2])
            sum_data += v
            hist_data.append((bin_begin, bin_end, v))
    return hist_data, sum_data


class BabelEnergyCalculator(AbstractEnergyCalculator):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF94 = "mmff94"
    FORCE_FIELD_MMFF94S = "mmff94s"
    FORCE_FIELD_GAFF = "gaff"
    FORCE_FIELD_GHEMICAL = "ghemical"

    VALID_FORCE_FIELDS = [
        FORCE_FIELD_UFF,
        FORCE_FIELD_MMFF94,
        FORCE_FIELD_MMFF94S,
        FORCE_FIELD_GAFF,
        FORCE_FIELD_GHEMICAL,
    ]

    def calculate(self, mol: Chem.Mol) -> float:
        smiles = Chem.MolToSmiles(mol)

        values = []
        for _ in range(self.RECALCULATION_LOOPS):
            molecule = pybel.readstring("smi", smiles)
            force_field = pybel._forcefields[self.force_field]
            force_field.Setup(molecule.OBMol)

            values.append(force_field.Energy())

        return min(values)


class AtomWiseEnergyCalculator(AbstractCalculator):
    def __init__(
        self, energy_calculator: Union[RdKitEnergyCalculator, BabelEnergyCalculator]
    ):
        self.energy_calculator = energy_calculator

    def calculate(self, mol: Chem.Mol) -> float:
        energy = self.energy_calculator.calculate(mol)
        return energy / mol.GetNumAtoms()


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


class RingCountCalculator(AbstractCalculator):
    def calculate(self, mol: Chem.Mol) -> float:
        Chem.GetSymmSSSR(mol)
        return -mol.GetRingInfo().NumRings()


class HistogramCalculator(AbstractCalculator):
    BASE_PATH = "hist/"

    def __init__(self, calc, name):
        self.calc = calc
        filepath = self.BASE_PATH + name + ".tsv"
        if os.path.exists(filepath):
            self.hist_data, self.sum = read_histogram_file(filepath)
        else:
            self.enable = False
            self.hist_data, self.sum = None, None

    def calculate(self, mol: Chem.Mol) -> float:
        s = self.calc.calculate(mol)
        for bin_begin, bin_end, val in self.hist_data:
            if bin_begin < s and s <= bin_end:
                likelihood = val / ((bin_end - bin_begin) * self.sum)
                ll = -np.log(likelihood + 1.0e-10)
                return ll
        return 1.0e10


class CombinationCalculator(AbstractCalculator):
    def __init__(self, calcs: list, weights: list = None):
        self.calcs = calcs
        self.calc_weights = weights

    def calculate(self, mol: Chem.Mol) -> float:
        scores = []
        for i, calc in enumerate(self.calcs):
            score = calc.calculate(mol)
            if self.calc_weights is not None:
                scores.append(self.calc_weights[i] * score)
            else:
                scores.append(score)
        return sum(scores)


class CalculatorFactory:

    COMPOUND_ENERGY_RDKIT_UFF = "compound_energy_rdkit_uff"
    COMPOUND_ENERGY_RDKIT_MMFF = "compound_energy_rdkit_mmff"
    COMPOUND_ENERGY_BABEL_UFF = "compound_energy_babel_uff"
    COMPOUND_ENERGY_BABEL_MMFF = "compound_energy_babel_mmff"
    COMPOUND_ENERGY_BABEL_MMFFS = "compound_energy_babel_mmffs"
    COMPOUND_ENERGY_BABEL_GAFF = "compound_energy_babel_gaff"
    COMPOUND_ENERGY_BABEL_GHEMICAL = "compound_energy_babel_ghemical"

    ATOMWISE_ENERGY_RDKIT_UFF = "atomwise_energy_rdkit_uff"
    ATOMWISE_ENERGY_RDKIT_MMFF = "atomwise_energy_rdkit_mmff"
    ATOMWISE_ENERGY_BABEL_UFF = "atomwise_energy_babel_uff"
    ATOMWISE_ENERGY_BABEL_MMFF = "atomwise_energy_babel_mmff"
    ATOMWISE_ENERGY_BABEL_MMFFS = "atomwise_energy_babel_mmffs"
    ATOMWISE_ENERGY_BABEL_GAFF = "atomwise_energy_babel_gaff"
    ATOMWISE_ENERGY_BABEL_GHEMICAL = "atomwise_energy_babel_ghemical"

    LOG_P = "log_p"
    QED = "qed"
    SA = "sa"
    MW = "mw"
    RING_COUNT = "ring_count"

    @staticmethod
    def get_options():
        options = {
            CalculatorFactory.COMPOUND_ENERGY_RDKIT_UFF: RdKitEnergyCalculator(
                RdKitEnergyCalculator.FORCE_FIELD_UFF
            ),
            CalculatorFactory.COMPOUND_ENERGY_RDKIT_MMFF: RdKitEnergyCalculator(
                RdKitEnergyCalculator.FORCE_FIELD_MMFF
            ),
            CalculatorFactory.COMPOUND_ENERGY_BABEL_UFF: BabelEnergyCalculator(
                BabelEnergyCalculator.FORCE_FIELD_UFF
            ),
            CalculatorFactory.COMPOUND_ENERGY_BABEL_MMFF: BabelEnergyCalculator(
                BabelEnergyCalculator.FORCE_FIELD_MMFF94
            ),
            CalculatorFactory.COMPOUND_ENERGY_BABEL_MMFFS: BabelEnergyCalculator(
                BabelEnergyCalculator.FORCE_FIELD_MMFF94S
            ),
            CalculatorFactory.COMPOUND_ENERGY_BABEL_GAFF: BabelEnergyCalculator(
                BabelEnergyCalculator.FORCE_FIELD_GAFF
            ),
            CalculatorFactory.COMPOUND_ENERGY_BABEL_GHEMICAL: BabelEnergyCalculator(
                BabelEnergyCalculator.FORCE_FIELD_GHEMICAL
            ),
            CalculatorFactory.ATOMWISE_ENERGY_RDKIT_UFF: AtomWiseEnergyCalculator(
                RdKitEnergyCalculator(RdKitEnergyCalculator.FORCE_FIELD_UFF)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_RDKIT_MMFF: AtomWiseEnergyCalculator(
                RdKitEnergyCalculator(RdKitEnergyCalculator.FORCE_FIELD_MMFF)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_BABEL_UFF: AtomWiseEnergyCalculator(
                BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_UFF)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_BABEL_MMFF: AtomWiseEnergyCalculator(
                BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_MMFF94)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_BABEL_MMFFS: AtomWiseEnergyCalculator(
                BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_MMFF94S)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_BABEL_GAFF: AtomWiseEnergyCalculator(
                BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_GAFF)
            ),
            CalculatorFactory.ATOMWISE_ENERGY_BABEL_GHEMICAL: AtomWiseEnergyCalculator(
                BabelEnergyCalculator(BabelEnergyCalculator.FORCE_FIELD_GHEMICAL)
            ),
            CalculatorFactory.LOG_P: LogpCalculator(),
            CalculatorFactory.MW: MwCalculator(),
            CalculatorFactory.QED: QedCalculator(),
            CalculatorFactory.SA: SaCalculator(),
            CalculatorFactory.RING_COUNT: RingCountCalculator(),
        }
        opt = {}
        for key in options.keys():
            opt["hist_" + key] = HistogramCalculator(options[key], "hist_" + key)
        options.update(opt)
        # print(options)
        return options

    @staticmethod
    def get_external_calc(name, config):
        external_options = ["kgcn"]
        if name in external_options:
            # calc_obj=calc.kgcn.Calculator(config)
            mod_name = "calc." + name
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "Calculator")
            calc_obj = cls(config)
            return calc_obj
        else:
            return None

    @staticmethod
    def create(reward_type, reward_weights=None, config=None) -> AbstractCalculator:
        options = CalculatorFactory.get_options()

        def get_calc(name, config):
            if name in options:
                return options[name]
            return CalculatorFactory.get_external_calc(name, config)

        if type(reward_type) is str:
            return get_calc(reward_type, config)
        elif type(reward_type) is list:
            calcs = [get_calc(rt, config) for rt in reward_type]
            return CombinationCalculator(calcs, reward_weights)
        return options[reward_type]
