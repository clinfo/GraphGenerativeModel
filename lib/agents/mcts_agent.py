import logging
from typing import List

import numpy as np

from lib.data_structures import Tree, Compound
from lib.filters import AbstractFilter


class MonteCarloTreeSearchAgent:

    """Available Output Types"""
    OUTPUT_FITTEST = "fittest"
    OUTPUT_DEEPEST = "deepest"
    OUTPUT_PER_LEVEL = "per_level"

    def __init__(
            self, minimum_depth: int, output_type: str, filters: List[AbstractFilter], breath_to_depth_ratio: float=0
    ):
        """
        :param minimum_depth: from the input parameters (see README.md for details)
        :param output_type: from the input parameters (see README.md for details)
        :param filters: from the input parameters (see README.md for details)
        :param breath_to_depth_ratio: from the input parameters (see README.md for details)
        """
        self.minimum_depth = minimum_depth
        self.output_type = output_type
        self.breath_to_depth_ratio = breath_to_depth_ratio
        self.filters = filters

    def reset(self, compound: Compound):
        """
        Resets the MCTS agent by reinitializing the tree from a root compound.
        :param compound: Compound
        :return: None
        """
        self.selected_node = None
        self.states_tree = Tree(compound)
        self.init_compound = compound.clone()

    def act(self, compound: Compound, reward: float):
        """
        Performs two operations.
        First, updates the state tree based on new child compound and reward from last iteration.
        Second, selects a node and bond to add to it for the next iteration.
        :param compound: compound obtained from last iteration after adding bond.
        :param reward: reward obtained at last iteration
        :return Compound: compound to process
        :return Tuple(int, int): bond to add
        """
        self.update_tree(compound, reward)
        self.selected_node = self.select_node(self.states_tree)
        self.selected_bond_indexes = self.select_bond(self.selected_node)

        return self.selected_node.get_compound().clone(), self.selected_bond_indexes

    def select_node(self, tree: Tree):
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

        scores = 1 - scores / score_sum if score_sum > 0 and len(scores) > 1 else [1 / len(scores)] * len(scores)
        scores /= np.sum(scores)  # normalize outputs (so they add up to 1)

        return np.random.choice(candidates, 1, p=scores)[0]

    def select_bond(self, node: Tree.Node):
        """
        In the expansion phase we loop over and calculate the reward for each possible bond type, then select the
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
            return None, None

        if len(current_bonds) > 0:
            neighboring_bonds = []
            candidate_atoms = set()
            for bond in current_bonds:
                candidate_atoms.add(bond.GetBeginAtomIdx())

            for source_atom, destination_atom in candidate_bonds:
                if source_atom in candidate_atoms or destination_atom in candidate_atoms:
                    neighboring_bonds.append((source_atom, destination_atom))

            if len(neighboring_bonds) > 0:
                candidate_bonds = neighboring_bonds
        source_atom, destination_atom = list(candidate_bonds)[np.random.choice(len(candidate_bonds), 1)[0]]
        return source_atom, destination_atom

    def update_tree(self, compound, reward):
        """
        Updates the state tree based on new child compound to add and associated reward.
        :param compound: compound obtained from last iteration after adding bond.
        :param reward: reward obtained at last iteration
        :return None:
        """
        if self.selected_node is not None and reward is not None:
            self.selected_node.get_compound().remove_bond(self.selected_bond_indexes)
            new_node = self.selected_node.add_child(compound)
            molecule = new_node.compound.clean(preserve=True)
            new_node.score = reward
            new_node.valid = new_node.score < np.Infinity and new_node.depth >= self.minimum_depth and all(
                _filter.apply(molecule, new_node.score) for _filter in self.filters
            )
            self.update(new_node)

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
            solutions.append({
                "smiles": node.get_compound().clean_smiles(),
                "depth": node.depth,
                "score": node.score
            })

        return solutions

    def get_output(self, compound):
        return self.prepare_output(self.states_tree)
