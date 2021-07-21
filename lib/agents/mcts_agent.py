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
        self.c = 0.3 # Explorative hyperparameter
        self.list_reward = []

    def reset(self, compound: Compound):
        """
        Resets the MCTS agent by reinitializing the tree from a root compound.
        :param compound: Compound
        :return: None
        """
        self.selected_node = None
        self.states_tree = Tree(compound)
        self.init_compound = compound.clone()

    def act(self, compound: Compound, reward:float):
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
        self.selected_node = self.tree_policy()
        if self.selected_node is None:
            return None, (None, None)
        self.selected_bond_indexes = self.select_unvisited_bond(self.selected_node)

        return self.selected_node.get_compound().clone(), self.selected_bond_indexes

    def tree_policy(self):
        """
        Walk through the tree and select one of the node to expand.
        If the node selected can't be expended, another node is selected randomly
        from all the node that can be expanded in the tree.
        :return Tree.Node: the node selected to expand
        """
        node = self.states_tree.root
        while node.is_expended() and not node.is_terminal():
            node = self.select_next_node(node)
        if node.is_terminal():
            # Decrease score to avoid looping on this node
            # node.score *= 0.7
            logging.debug("Reached terminal node")
            node.selection_score *= 1.1
            self.update(node)
            return self.select_unvisited_node()
        return node

    def select_next_node(self, node: Tree.Node):
        """
        This function is the poilicy apply to select a children node
        :return Tree.Node: Node selected by walking through the tree
        """
        def ucb(node):
            """
            Strategy to select the node used by unitMCTS (https://arxiv.org/pdf/2010.16399.pdf)
            """
            # return node.performance/node.visits + self.c * np.sqrt(np.log(node.parent.visits) / node.visits)
            return node.performance/node.visits - self.c * np.sqrt(np.log(node.parent.visits) / node.visits)

        performance = [ucb(child) for child in node.children]
        id_chosen_node = np.argmin(performance)
        # id_chosen_node = np.argmax(performance)
        return node.children[id_chosen_node]

    def select_unvisited_node(self):
        """
        Select a pseudo random node that can be extended form all the node in the
        tree.
        The deeper the node the higher is chance to be selected.
        :return Tree.Node: Node selected to be expanded.
        """
        # Retrieve unvisited node
        all_node = self.states_tree.flatten()
        unvisited_node = [n for n in all_node if not n.is_expended()]
        if len(unvisited_node) == 0:
            logging.info("All possible node have been explored")
            return None
        # Select one
        performances = [n.depth / n.performance for n in unvisited_node]
        # performances = [n.performance * n.depth / n.visits for n in unvisited_node]
        performances = performances / np.sum(performances)
        id_node = np.random.choice(range(len(unvisited_node)), p=performances)
        return unvisited_node[id_node]

    def select_unvisited_bond(self, node: Tree.Node):
        """
        Select randomly a bond to add to the molecule from the neighboring bond.
        Remove the selected bond from the list of unvisited bond.
        :return Tuple(int, int): bond selected
        """
        possible_bonds = node.unexplore_neighboring_bonds
        id_bond = np.random.choice(range(len(possible_bonds)))
        selected_bond = possible_bonds.pop(id_bond)
        node.unexplore_neighboring_bonds = possible_bonds
        return selected_bond

    def update_tree(self, compound, reward):
        """
        Updates the state tree based on new child compound to add and associated reward.
        :param compound: compound obtained from last iteration after adding bond.
        :param reward: reward obtained at last iteration
        :return None:
        """
        if self.selected_node is not None and reward is not None:
            duplicate = self.states_tree.find_duplicate(compound, self.selected_node.depth)
            if duplicate is None or duplicate.score < reward:
                if duplicate is not None:
                    duplicate.parent.children.remove(duplicate)

                new_node = self.selected_node.add_child(compound)
                # Update neighboring bonds to assure consistency in the next selection
                new_node.compound.compute_neighboring_bonds()
                molecule = new_node.compound.clean(preserve=True)
                # new_node.reward = reward
                new_node.score = reward
                new_node.selection_score = reward - 0.01 * self.selected_node.depth
                # new_node.valid = new_node.score == 0 and new_node.depth >= self.minimum_depth and all(
                new_node.valid = new_node.score < Tree.INFINITY and new_node.depth >= self.minimum_depth and all(
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
        performance = node.selection_score
        node.performance = performance
        node.visits += 1

        if node.performance > Tree.INFINITY:
        # if node.performance == 0:
            return

        while node.depth > 0:
            node.parent.performance += performance
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

    def get_output(self, compound: Compound, reward: float):
        """
        Returns output based on the current state of the Tree.
        For details, see README.md (around the description for output_type).

        :param compound: Compound, not actually used but necessary to have same format as other agents.
        :param reward: float, not actually used but necessary to have same format as other agents.
        :return: list(dict)
        """
        return self.prepare_output(self.states_tree)
