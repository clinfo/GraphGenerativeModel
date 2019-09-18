import pybel
from rdkit import Chem
from rdkit.Chem import AllChem


class EnergyCalculatorPrototype(object):

    VALID_FORCE_FIELDS = []

    def __init__(self, force_field):
        self.force_field = force_field

        if self.force_field not in self.VALID_FORCE_FIELDS:
            raise ValueError("'{}' cannot handle a '{}' force field!".format(self.__class__.__name__, self.force_field))

    def calculate(self, smiles):
        raise NotImplementedError


class RdKitEnergyCalculator(EnergyCalculatorPrototype):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF = "mmff"

    VALID_FORCE_FIELDS = [FORCE_FIELD_UFF, FORCE_FIELD_MMFF]

    def calculate(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        molecule.UpdatePropertyCache()

        force_field = self.get_force_field(molecule)
        force_field.Initialize()
        force_field.Minimize()

        return force_field.CalcEnergy()

    def get_force_field(self, molecule):
        """
        Force Field Factory
        :param molecule: rdKit.Mol
        :return: AllChem.ForceField
        :raises: ValueError (unknown force field)
        """

        Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule, randomSeed=0)

        if self.force_field == self.FORCE_FIELD_MMFF:
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            return AllChem.MMFFGetMoleculeForceField(molecule, properties)

        if self.force_field == self.FORCE_FIELD_UFF:
            return AllChem.UFFGetMoleculeForceField(molecule)


class BabelEnergyCalculator(EnergyCalculatorPrototype):

    FORCE_FIELD_UFF = "uff"
    FORCE_FIELD_MMFF96 = "mmff96"
    FORCE_FIELD_MMFF94S = "mmff94s"
    FORCE_FIELD_GAFF = "gaff"
    FORCE_FIELD_GHEMICAL = "ghemical"

    VALID_FORCE_FIELDS = [
        FORCE_FIELD_UFF, FORCE_FIELD_MMFF96, FORCE_FIELD_MMFF94S, FORCE_FIELD_GAFF, FORCE_FIELD_GHEMICAL
    ]

    def calculate(self, smiles):
        molecule = pybel.readstring("smi", smiles)
        force_field = pybel._forcefields[self.force_field]

        force_field.Setup(molecule.OBMol)
        return force_field.Energy()


class EnergyCalculatorFactory(object):

    AVAILABLE_CALCULATORS = {
        "rdkit": RdKitEnergyCalculator,
        "babel": BabelEnergyCalculator
    }

    @staticmethod
    def get(calculator):
        calculator, force_field = calculator.split("_")

        if calculator not in EnergyCalculatorFactory.AVAILABLE_CALCULATORS:
            raise ValueError(
                "Unknown energy calculator requested: '{}'. Available options are: '{}'".format(
                    calculator, EnergyCalculatorFactory.AVAILABLE_CALCULATORS.keys()
                )
            )

        calculator = EnergyCalculatorFactory.AVAILABLE_CALCULATORS[calculator]
        return calculator(force_field)
