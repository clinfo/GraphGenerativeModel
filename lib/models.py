import logging
from typing import List

import numpy as np
from numpy.lib.utils import source
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from lib.calculators import AbstractCalculator
from lib.data_providers import MoleculeLoader
from lib.data_structures import Tree, Compound
from lib.filters import AbstractFilter

from rdkit.Chem.Descriptors import ExactMolWt, MolWt


class MonteCarloTreeSearch:

    """
    Bonds that are tested. During expansion, the reward for each
    bond type is calculated, and the lowest one is selected.
    """

    AVAILABLE_BONDS = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    """Available Output Types"""
    OUTPUT_FITTEST = "fittest"
    OUTPUT_DEEPEST = "deepest"
    OUTPUT_PER_LEVEL = "per_level"

    def __init__(
        self,
        data_provider: MoleculeLoader,
        minimum_depth,
        output_type,
        calculator: AbstractCalculator,
        filters: List[AbstractFilter],
        breath_to_depth_ratio=0,
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
        self.all_nodes = {}
        self.c = 0.8  # Explorative hyperparameter
        self.select_dict = {
            "breath_to_depth": "select_breath_to_depth",
            "MCTS_classic": "select_MCTS_classic",
        }

    def start(self, molecules_to_process=10, iterations_per_molecule=100):
        """
        Creates an iterator that loads one molecule and passes it for processing.

        :param molecules_to_process: How many molecules from the dataset to process?
        :param iterations_per_molecule: How many times to iterate over a single molecule?
        :return: an iterator
        """
        for i, compound in enumerate(
            self.data_provider.fetch(molecules_to_process=molecules_to_process)
        ):
            yield self.apply(compound, iterations=iterations_per_molecule, index=i)

    def apply(self, compound: Compound, iterations: int, index: int):
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
                continue

            score = self.simulate(new_node)

            logging.debug("Score: {}".format(score))

            self.update(new_node)
            print("step = ", _)
        states_tree.print_tree()
        states_tree.tree_to_dot(index=index)
        return self.prepare_output(states_tree)

    def select(self, tree: Tree, strat="MCTS_classic"):
        """
        The selection phase. Choose bteween the classic MCTS details or originally implemented by the
        breath_to_depth_ratio
        :param tree: Tree
        :return: Tree.Node
        """
        if strat in self.select_dict:
            return getattr(self, self.select_dict[strat])(tree)

    def select_breath_to_depth(self, tree: Tree):
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

        scores = np.array([abs(node.performance) / node.visits for node in candidates])
        score_sum = np.sum(scores)

        scores = (
            1 - scores / score_sum
            if score_sum > 0 and len(scores) > 1
            else [1 / len(scores)] * len(scores)
        )
        scores /= np.sum(scores)  # normalize outputs (so they add up to 1)

        return np.random.choice(candidates, 1, p=scores)[0]

    def select_MCTS_classic(self, tree: Tree):
        """
        Walk through the tree and select one of the node to expand.
        If the node selected can't be expended, another node is selected randomly
        from all the node that can be expanded in the tree.
        :return Tree.Node: the node selected to expand
        """
        node = tree.root

        while len(node.children) != 0:
            node = self.select_next_node(node)
        if node.is_terminal():
            # Increase score to avoid looping on this node
            node.score *= 1.3
            self.update(node)
            return self.select_unvisited_node(tree)
        return node

    def select_next_node(self, node: Tree.Node):
        """
        This function is the poilicy apply to select a children node
        :return Tree.Node: Node selected by walking through the tree
        """

        def ucb(node):
            """
            Strategy to select the node used by unitMCTS (https://arxiv.org/pdf/2010.16399.pdf)
            edit: change for argmin instead of argmax and ponderated by node probability
            """
            return node.performance / (
                node.visits + node.prior_occurence
            ) + self.c * np.sqrt(node.visits / np.log(node.parent.visits))

        performance = [ucb(child) for child in node.children]
        id_chosen_node = np.argmin(performance)
        return node.children[id_chosen_node]

    def select_unvisited_node(self, tree: Tree):
        """
        Select a pseudo random node that can be extended form all the node in the
        tree.
        The deeper the node the higher is chance to be selected.
        :return Tree.Node: Node selected to be expanded.
        """
        # Retrieve unvisited node
        all_node = tree.flatten()
        unvisited_node = [n for n in all_node if not n.is_expended()]
        # Select one
        # TO CHANGE
        performances = [n.depth / (n.performance + 0.1) for n in unvisited_node]
        performances = performances / np.sum(performances)
        id_node = np.random.choice(range(len(unvisited_node)), p=performances)
        return unvisited_node[id_node]

    def add_bond(self, node: Tree.Node, bond_mode="best"):
        """
        Add a new available bond to a given compound.
        :return Compound: Newly created Compound.
        """
        # source_atom, destination_atom = node.select_unvisited_bond()
        source_atom, destination_atom = node.select_unvisited_bond_aromatic()

        new_compound = node.get_compound().clone()
        new_compound.remove_bond((source_atom, destination_atom))
        new_compound.flush_bonds()

        molecule = new_compound.get_molecule()

        target_bond_index = molecule.AddBond(
            int(source_atom), int(destination_atom), BondType.UNSPECIFIED
        )
        target_bond = molecule.GetBondWithIdx(target_bond_index - 1)

        # if member of aromatic queue was selected
        if len(node.get_compound().get_aromatic_queue()) > 0:
            # set conjugate
            if node.get_compound().get_last_bondtype() == Chem.rdchem.BondType.SINGLE:
                bond_type = Chem.rdchem.BondType.DOUBLE
            else:
                bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_mode == "best":
            per_bond_rewards = {}
            for bond_type in self.AVAILABLE_BONDS:
                target_bond.SetBondType(bond_type)
                per_bond_rewards[bond_type] = self.calculate_reward(new_compound)

            bond_type = min(per_bond_rewards, key=per_bond_rewards.get)
        else:
            bond_type = self.AVAILABLE_BONDS[
                np.random.randint(len(self.AVAILABLE_BONDS), size=1)[0]
            ]

        target_bond.SetBondType(bond_type)

        new_compound.pass_aromatic_queue(node.get_compound().get_aromatic_queue())
        node.get_compound().reset_aromatic_queue()

        new_compound.add_bond_history(
            [source_atom, destination_atom], bond_type, node.get_node_history()
        )
        return new_compound

    def expand(self, node: Tree.Node, bond_mode="best"):
        """
        In the expansion phase we loop over and calculate the reward for each possible bond type, then select the
        lowest one. The new molecule is then added as a child node to the input node. The bond cache in the compounds
        is also updated accordingly to reflect the changes.

        :param node: Tree.Node (from selection)
        :return: Tree.Node (new child)
        """
        if node.is_expended() or node.is_terminal():
            logging.debug("already fully expended")
            return None

        child_dict = {}

        while node.is_expended() == False:
            new_compound = self.add_bond(node, bond_mode=bond_mode)
            id = new_compound.compute_hash()
            if id not in self.all_nodes:
                child = node.add_child(new_compound)
                self.all_nodes[id] = child
                child_dict[id] = child
            else:
                self.all_nodes[id].prior_occurence += 1

        if child_dict == {}:
            print("All paths already exist in the graph")
            node.penalize_node()
            return None

        # get first child
        children_view = child_dict.values()
        children_iterator = iter(children_view)
        child = next(children_iterator)
        return child

    def rollout_step(self, node, bond_mode="best"):
        """
        Simple rollout step generating a new compound from available bonds
        :param node: Tree.Node
        :return: Tree.Node
        """
        new_compound = self.add_bond(node, bond_mode=bond_mode)
        orphan = node.get_child(new_compound)
        del node
        return orphan

    def simulate(self, node: Tree.Node, rollout=True, mode="best"):
        """
        Randomly setting all remaining bonds will almost always lead to an invalid molecule. Thus, we calculate
        the reward based on the current molecule structure (which is valid) instead of performing a roll-out.
        :param node: Tree.Node
        :return: double
        """
        logging.debug("Simulating...")
        if rollout == True:
            start_node = node

            while node.get_compound().get_mass() < 200 and node.is_expended() == False:
                node = self.rollout_step(node, bond_mode=mode)

            start_node.score = self.calculate_reward(node.compound)
            node.score = start_node.score
        else:
            node.score = self.calculate_reward(node.compound)

            molecule = node.compound.clean(preserve=True)
            node.valid = (
                node.score < np.Infinity
                and node.depth >= self.minimum_depth
                and all(_filter.apply(molecule, node.score) for _filter in self.filters)
            )

        # encourage aromaticity
        if node.get_compound().is_aromatic():
            node.score /= 100

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
        backproped_score = node.score

        if node.performance > np.Infinity:
            return

        while node.depth > 0:
            node.parent.performance += backproped_score
            node.parent.visits += 1
            node = node.parent

    def calculate_reward(self, compound: Compound):
        """
        Calculate the reward of the compound based on the requested force field.
        If the molecule is not valid, the reward will be infinity.

        :param compound: Compound
        :return: float
        """
        try:
            molecule = compound.clean(preserve=True)
            smiles = Chem.MolToSmiles(molecule)

            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError("Invalid molecule: {}".format(smiles))

            molecule.UpdatePropertyCache()
            reward = self.calculator.calculate(molecule)
            if np.isnan(reward):
                raise ValueError("NaN reward encountered: {}".format(smiles))

            logging.debug(
                "{} : {} : {:.6f}".format(
                    compound.get_smiles(), Chem.MolToSmiles(molecule), reward
                )
            )
            return reward

        except (ValueError, RuntimeError, AttributeError) as e:
            logging.debug(
                "[INVALID REWARD]: {} - {}".format(compound.get_smiles(), str(e))
            )
            return np.Infinity

    def prepare_output(self, tree: Tree):
        """
        Prepares the output based on the selected output type (input parameter).
        For details, see README.md (around the description for output_type)

        :param tree: Tree
        :return: list(dict)
        """

        if self.output_type == self.OUTPUT_FITTEST:
            output = tree.get_fittest()

            if output is None:
                logging.info("No molecules reach the minimum depth")
                return None

            return self.format_output(output)

        output = tree.get_fittest_per_level()

        if len(output) == 0:
            logging.info("No molecules reach the minimum depth")
            return None

        if self.output_type == self.OUTPUT_DEEPEST:
            deepest_level = max(output.keys())
            return self.format_output(output[deepest_level])

        if self.output_type == self.OUTPUT_PER_LEVEL:
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
            solutions.append(
                {
                    "smiles": Chem.MolToSmiles(node.get_compound().clean()),
                    "depth": node.depth,
                    "score": node.score,
                }
            )

        return solutions
