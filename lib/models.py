import logging
import random
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from lib.calculators import AbstractCalculator
from lib.data_providers import MoleculeLoader
from lib.data_structures import Tree, Compound
from lib.filters import AbstractFilter


class MonteCarloTreeSearch:

    """
    Bonds that are tested. During expansion, the energy of each
    bond type is calculated, and the lowest one is selected.
    """
    AVAILABLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    """Available Output Types"""
    OUTPUT_FITTEST = "fittest"
    OUTPUT_DEEPEST = "deepest"
    OUTPUT_PER_LEVEL = "per_level"

    def __init__(
            self, data_provider: MoleculeLoader, minimum_depth, output_type,
            calculator: AbstractCalculator, filters: List[AbstractFilter], breath_to_depth_ratio=0
    ):
        """
        :param data_provider: MoleculeLoader
        :param minimum_depth: from the input parameters (see README.md for details)
        :param output_type: from the input parameters (see README.md for details)
        :param calculator: from the input parameters (see README.md for details)
        :param filters: from the input parameters (see README.md for details)
        :param breath_to_depth_ratio: from the input parameters (see README.md for details)
        """
        self.data_provider = data_provider
        self.minimum_depth = minimum_depth
        self.output_type = output_type
        self.breath_to_depth_ratio = breath_to_depth_ratio
        self.calculator = calculator
        self.filters = filters

    def start(self, molecules_to_process=10, iterations_per_molecule=100):
        """
        Creates an iterator that loads one molecule and passes it for processing.

        :param molecules_to_process: How many molecules from the dataset to process?
        :param iterations_per_molecule: How many times to iterate over a single molecule?
        :return: an iterator
        """
        for compound in self.data_provider.fetch(molecules_to_process=molecules_to_process):
            yield self.apply(compound, iterations=iterations_per_molecule)

    def apply(self, compound: Compound, iterations: int):
        """
        Optimize a molecule (that is, run the Monte Carlo Tree Search)
        :return: list(dict)
        """
        states_tree = Tree(compound)
        logging.info("Processing now: {}".format(compound.get_smiles()))
        logging.info("Bonds Count: {}".format(compound.bonds_count()))
        logging.info("Atoms Count: {}".format(compound.get_molecule().GetNumAtoms()))

        for _ in range(iterations):
            logging.debug("Iteration {}/{}".format(_ + 1, iterations))
            selection = self.select(states_tree)
            if selection is None:
                break

            new_node = self.expand(selection)

            if new_node is None:
                logging.debug("Lead node reached!")
                continue

            score = self.simulate(new_node)
            logging.debug("Score: {}".format(score))

            self.update(new_node)

        states_tree.print_tree()
        return self.prepare_output(states_tree)

    def select(self, tree: Tree):
        """
        The selection phase. See README.md for details. (in details on breath_to_depth_ratio)
        :param tree: Tree
        :return: Tree.Node
        """
        nodes_per_level = tree.group()
        levels = list(nodes_per_level.keys())
        ratio = abs(self.breath_to_depth_ratio)

        if ratio == 0:
            ratio = 1000000

        probabilities = np.random.dirichlet(np.ones(len(levels)) * ratio, 1)
        probabilities = np.sort(probabilities[0])

        if self.breath_to_depth_ratio < 0:
            probabilities = np.flip(probabilities)

        selected_level = np.random.choice(levels, 1, p=probabilities)[0]
        candidates = nodes_per_level[selected_level]

        scores = np.array([node.performance / node.visits for node in candidates])
        score_sum = np.sum(scores)

        scores = 1 - scores / score_sum if score_sum > 0 and len(scores) > 1 else [1 / len(scores)] * len(scores)
        scores /= np.sum(scores)  # normalize outputs (so they add up to 1)

        return np.random.choice(candidates, 1, p=scores)[0]

    def expand(self, node: Tree.Node):
        """
        In the expansion phase we loop over and calculate the energy of each possible bond type, then select the
        lowest one. The new molecule is then added as a child node to the input node. The bond cache in the compounds
        is also updated accordingly to reflect the changes.

        :param node: Tree.Node (from selection)
        :return: Tree.Node (new child)
        """
        compound = node.get_compound().clone()
        molecule = compound.get_molecule()

        available_bonds = node.get_compound().get_bonds()
        current_bonds = molecule.GetBonds()
        candidate_bonds = available_bonds

        if len(candidate_bonds) == 0:
            logging.debug("All bonds have been used.")
            return None

        if len(current_bonds) > 0:
            neighboring_bonds = []
            candidate_atoms = set()
            for bond in current_bonds:
                candidate_atoms.add(bond.GetBeginAtomIdx())

            for source_atom, destination_atom in available_bonds:
                if source_atom in candidate_atoms or destination_atom in candidate_bonds:
                    neighboring_bonds.append((source_atom, destination_atom))

            if len(neighboring_bonds) > 0:
                candidate_bonds = neighboring_bonds

        source_atom, destination_atom = random.sample(candidate_bonds, 1)[0]
        target_bond_index = molecule.AddBond(int(source_atom), int(destination_atom), BondType.UNSPECIFIED)
        target_bond = molecule.GetBondWithIdx(target_bond_index - 1)

        per_bond_energy_values = {}
        for bond_type in self.AVAILABLE_BONDS:
            target_bond.SetBondType(bond_type)
            per_bond_energy_values[bond_type] = self.calculate_reward(compound)

        target_bond.SetBondType(min(per_bond_energy_values, key=per_bond_energy_values.get))
        node.get_compound().remove_bond((source_atom, destination_atom))

        compound.remove_bond((source_atom, destination_atom))
        compound.flush_bonds()

        child = node.add_child(compound)
        return child

    def simulate(self, node: Tree.Node):
        """
        Randomly setting all remaining bonds will almost always lead to an invalid molecule. Thus, we calculate
        the reward based on the current molecule structure (which is valid) instead of performing a roll-out.
        :param node: Tree.Node
        :return: double
        """
        logging.debug("Simulating...")
        node.score = self.calculate_reward(node.compound)
        return node.score

    def update(self, node: Tree.Node):
        """
        Back-propagation. We update the score and number of times each node was visited. Since we are not
        using Upper Confidence Bounds during the selection process, these are mainly relevant for debugging purposes.
        :param node: Tree.Node
        :return: None
        """
        node.performance = node.score
        node.visits += 1

        if node.performance > Tree.INFINITY:
            return

        while node.depth > 0:
            node.parent.performance += node.performance
            node.parent.visits += 1
            node = node.parent

    def calculate_reward(self, compound: Compound):
        """
        Calculate the energy of the compound based on the requested force field.
        If the molecule is not valid, the reward will be infinity.

        :param compound: Compound
        :return: float
        """
        try:
            molecule = compound.clean(preserve=True)
            smiles = Chem.MolToSmiles(molecule)

            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError("Invalid molecule: {}".format(smiles))

            reward = self.calculator.calculate(molecule)
            if not all(filter_.apply(smiles, reward) for filter_ in self.filters):
                raise ValueError("This molecule failed to pass some filters: {}".format(smiles))

            logging.debug("{} : {} : {:.6f}".format(compound.get_smiles(), Chem.MolToSmiles(molecule), reward))
            return reward

        except (ValueError, RuntimeError, AttributeError) as e:
            logging.debug("[INVALID REWARD]: {} - {}".format(compound.get_smiles(), str(e)))
            return np.Infinity

    def prepare_output(self, tree: Tree):
        """
        Prepares the output based on the selected output type (input parameter).
        For details, see README.md (around the description for output_type)

        :param tree: Tree
        :return: list(dict)
        """

        if self.output_type == self.OUTPUT_FITTEST:
            output = tree.get_fittest(minimum_depth=self.minimum_depth)

            if output is None:
                logging.info("No molecules reaching the minimum depth")
                return None

            return self.format_output(output)

        if self.output_type == self.OUTPUT_DEEPEST:
            max_depth = tree.get_depth()
            output = None

            while output is None:
                output = tree.get_fittest(minimum_depth=max_depth)
                max_depth -= 1

            return self.format_output(output)

        if self.output_type == self.OUTPUT_PER_LEVEL:
            output = tree.get_fittest_per_level(minimum_depth=self.minimum_depth)
            return self.format_output(list(output.values()))

        raise ValueError("Unknown output type: {}".format(self.output_type))

    def format_output(self, nodes):
        """
        Gather and convert the result (along with other relevant data) to json format
        :param nodes: list(Tree.Node)
        :return: list(dict)
        """

        if not isinstance(nodes, list):
            nodes = [nodes]

        solutions = []
        for node in nodes:
            solutions.append({
                "smiles": Chem.MolToSmiles(node.get_compound().clean()),
                "depth": node.depth,
                "score": node.score
            })

        return solutions
