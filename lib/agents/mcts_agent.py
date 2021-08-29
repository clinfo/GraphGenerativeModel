from functools import partial
import logging
from typing import List

import numpy as np

from lib.data_structures import Tree, Compound
from lib.filters import AbstractFilter
from copy import deepcopy


class MonteCarloTreeSearchAgent:

    """Available Output Types"""

    OUTPUT_FITTEST = "fittest"
    OUTPUT_DEEPEST = "deepest"
    OUTPUT_PER_LEVEL = "per_level"

    def __init__(
        self,
        minimum_depth: int,
        output_type: str,
        filters: List[AbstractFilter],
        select_method: str,
        breath_to_depth_ratio: float = 0,
        tradeoff_param: float = 0,
        force_begin_ring: bool = False,
    ):
        """
        :param minimum_depth: from the input parameters (see README.md for details)
        :param output_type: from the input parameters (see README.md for details)
        :param filters: from the input parameters (see README.md for details)
        :param select_method: str: Selection method to use for the tree policy
        :param breath_to_depth_ratio: from the input parameters (see README.md for details)
        :param tradeoff_param: exploration exploitation tradeoff parameter
        """
        self.minimum_depth = minimum_depth
        self.output_type = output_type
        self.breath_to_depth_ratio = breath_to_depth_ratio
        self.filters = filters
        self.tradeoff_param = tradeoff_param
        self.force_begin_ring = force_begin_ring
        self.list_reward = []
        self.select_dict = {
            "breath_to_depth": "select_breath_to_depth",
            "MCTS_classic": "select_MCTS_classic",
            "MCTS_aromatic": "select_MCTS_aromatic",
            "random": "select_MCTS_classic",
        }
        self.select_bond_dict = {
            "breath_to_depth": "select_bond_breath_to_depth",
            "MCTS_classic": "select_bond_MCTS_classic",
            "random": "select_bond_MCTS_classic",
            "MCTS_aromatic": "select_bond_MCTS_aromatic",
        }
        assert (
            select_method in self.select_dict
        ), f"select_method must be in {list(self.select_dict.keys())}"
        self.tree_policy = getattr(self, self.select_dict[select_method])
        self.select_bond = getattr(self, self.select_bond_dict[select_method])
        if select_method == "random":
            self.select_bond = partial(self.select_bond, proba_type="random_on_pred")
        self.select_method = select_method

    def reset(self, compound: Compound):
        """
        Resets the MCTS agent by reinitializing the tree from a root compound.
        :param compound: Compound
        :return: None
        """
        self.selected_node = None
        self.states_tree = Tree(compound)
        self.init_compound = compound.clone()

    def act(self, compound: Compound, reward: float, score: float):
        """
        Performs two operations.
        First, updates the state tree based on new child compound and reward from last iteration.
        Second, selects a node and bond to add to it for the next iteration.
        :param compound: compound obtained from last iteration after adding bond.
        :param reward: reward obtained at last iteration
        :param score: reward on the compound of this node (not the one at the rollout)
        :return Compound: compound to process
        :return Tuple(int, int): bond to add
        """
        self.update_tree(compound, reward, score)
        self.selected_node = self.tree_policy()
        if self.selected_node is None:
            return None, (None, None)
        self.selected_bond_indexes = self.select_bond(self.selected_node)

        # infos needs to be passed before env
        new_compound = self.selected_node.get_compound().clone()
        new_compound.pass_parent_info(self.selected_node.compound)

        return new_compound, self.selected_bond_indexes

    def select_MCTS_classic(self):
        """
        Walk through the tree and select one of the node to expand.
        If the node selected can't be expended, another node is selected randomly
        from all the node that can be expanded in the tree.
        :return: Tree.Node: the node selected to expand
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

    def get_node_aromatic_process(self):
        """
        Return the node use by the aromatic mode if the creation of a cycle is ongoing.
        """
        return [
            n for n in self.states_tree.flatten() if len(n.compound.aromatic_queue) > 0
        ]

    def select_MCTS_aromatic(self):
        """
        Select the next node, if there is an aromatic process in going then
        the next node will be the one enabling this process. Otherwise the selection
        is similar to MCTS_classic
        :return: Tree.Node: the node selected to expand
        """
        in_progress_aromatic_node = self.get_node_aromatic_process()
        if len(in_progress_aromatic_node) > 0:
            node = in_progress_aromatic_node[0]
            if node.is_terminal():
                logging.debug("Reached terminal node")
                node.selection_score *= 1.1
                self.update(node)
                return self.select_unvisited_node()
            else:
                return node
        else:
            if self.force_begin_ring:
                # All Cycle have been created and the suppression is done only one time
                if (
                    len(self.states_tree.root.compound.available_cycles) == 0
                    and len(self.states_tree.root.unexplored_neighboring_bonds) != 0
                ):
                    # Delete all possible branches except the one after the rings
                    max_depth = max(self.states_tree.group())
                    for node in self.states_tree.flatten():
                        if node.depth < max_depth:
                            node.unexplored_neighboring_bonds = []
            return self.select_MCTS_classic()

    def select_breath_to_depth(self):
        """
        The selection phase. See README.md for details. (in details on breath_to_depth_ratio)
        :param tree: Tree
        :return: Tree.Node
        """
        nodes_per_level = self.states_tree.group()
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

        scores = np.array(
            [abs(node.performance) / (node.visits + 1) for node in candidates]
        )
        score_sum = np.sum(scores)

        scores = (
            1 - scores / score_sum
            if score_sum > 0 and len(scores) > 1
            else [1 / len(scores)] * len(scores)
        )
        scores /= np.sum(scores)  # normalize outputs (so they add up to 1)

        return np.random.choice(candidates, 1, p=scores)[0]

    def select_next_node(self, node: Tree.Node):
        """
        This function is the poilicy apply to select a children node
        :return: Tree.Node: Node selected by walking through the tree
        """

        def ucb(node):
            """
            Strategy to select the node used by unitMCTS (https://arxiv.org/pdf/2010.16399.pdf)
            :return: float
            """
            # return node.performance/node.visits + self.tradeoff_param * np.sqrt(np.log(node.parent.visits) / node.visits)
            return node.performance / node.visits - self.tradeoff_param * np.sqrt(
                np.log(node.parent.visits) / node.visits
            )

        performance = [ucb(child) for child in node.children]
        id_chosen_node = np.argmin(performance)
        # id_chosen_node = np.argmax(performance)
        return node.children[id_chosen_node]

    def select_unvisited_node(self):
        """
        Select a pseudo random node that can be extended form all the node in the
        tree.
        The deeper the node the higher is chance to be selected.
        :return: Tree.Node: Node selected to be expanded.
        """
        # Retrieve unvisited node
        all_node = self.states_tree.flatten()
        unexpended_node = [n for n in all_node if not n.is_expended()]
        if len(unexpended_node) == 0:
            logging.info("All possible node have been explored")
            return None
        # Select one
        performances = [n.depth / (n.performance / n.visits) for n in unexpended_node]
        # performances = [n.performance * n.depth / n.visits for n in unexpended_node]
        performances = performances / np.sum(performances)
        id_node = np.random.choice(range(len(unexpended_node)), p=performances)
        return unexpended_node[id_node]

    def select_bond_breath_to_depth(self, node: Tree.Node):
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
                if (
                    source_atom in candidate_atoms
                    or destination_atom in candidate_atoms
                ):
                    neighboring_bonds.append((source_atom, destination_atom))

            if len(neighboring_bonds) > 0:
                candidate_bonds = neighboring_bonds
        source_atom, destination_atom = list(candidate_bonds)[
            np.random.choice(len(candidate_bonds), 1)[0]
        ]
        return source_atom, destination_atom

    def select_bond_MCTS_classic(self, node: Tree.Node, proba_type="random"):
        """
        Select randomly a bond to add to the molecule from the neighboring bond.
        Remove the selected bond from the list of unvisited bond.
        :param node: Tree.Node:Node to expand
        :param proba_type: str: How to select the expansion
        :return: Tuple(int, int): bond selected
        """
        possible_bonds = node.unexplored_neighboring_bonds
        if proba_type == "random":
            p = None
        elif proba_type == "random_on_pred":
            p = node.compound.get_pred_proba_next_bond(
                node.unexplored_neighboring_bonds
            )
        id_bond = np.random.choice(range(len(possible_bonds)), p=p)
        selected_bond = possible_bonds.pop(id_bond)
        node.unexplored_neighboring_bonds = possible_bonds
        return selected_bond

    def select_bond_MCTS_aromatic(self, node: Tree.Node, proba_type="random"):
        """
        Select a bond that favors the formation of an aromatic ring.
        Switch that bond in the first position of the aromatic queue to be deleted by the child.
        :param node: Tree.Node:Node to expand
        :param proba_type: str: How to select the expansion
        :return: Tuple(int, int): bond selected
        """
        possible_bonds = node.unexplored_neighboring_bonds
        aromatic_queue = deepcopy(node.get_compound().fill_aromatic_queue())
        sorted_bonds = [sorted(bond) for bond in possible_bonds]
        id_bond = -1
        if len(aromatic_queue) > 0:
            for i, bond in enumerate(aromatic_queue):
                if bond in sorted_bonds:
                    id_bond = sorted_bonds.index(bond)
                    # put chosen bond in first position of the aromatic queue
                    chosen_bond = node.get_compound().aromatic_queue.pop(i)
                    # We need the lenght of the aromatic queue to be superior to zero in add_bond
                    node.get_compound().aromatic_queue.insert(0, chosen_bond)
                    break
        else:
            if proba_type == "random":
                p = None
            elif proba_type == "random_on_pred":
                p = node.compound.get_pred_proba_next_bond(
                    node.unexplored_neighboring_bonds
                )

            id_bond = np.random.choice(range(len(possible_bonds)), p=p)

        # case to handle if cycle bond was removed by remove_full_atom_other_bond
        if id_bond == -1:
            id_bond = np.random.choice(range(len(possible_bonds)), p=None)

        selected_bond = possible_bonds.pop(id_bond)
        node.unexplored_neighboring_bonds = possible_bonds

        return selected_bond

    def update_tree(self, compound, reward, score):
        """
        Updates the state tree based on new child compound to add and associated reward.
        :param compound: compound obtained from last iteration after adding bond.
        :param reward: reward obtained at last iteration
        :return None:
        """
        if self.selected_node is not None and reward is not None:
            # compound.bond_history.update(self.selected_node.compound.bond_history)

            compound.compute_hash()
            duplicate = self.states_tree.find_duplicate(compound)
            if duplicate is None:  # or duplicate.score > reward:
                new_node = self.selected_node.add_child(compound)
                # Update neighboring bonds to assure consistency in the next selection
                new_node.compound.compute_neighboring_bonds()
                molecule = new_node.compound.clean(preserve=True)
                # new_node.reward = reward
                new_node.score = score
                new_node.selection_score = reward  # - 0.01 * self.selected_node.depth
                # new_node.valid = new_node.score == 0 and new_node.depth >= self.minimum_depth and all(
                new_node.valid = (
                    new_node.score < Tree.INFINITY
                    and new_node.depth >= self.minimum_depth
                    and all(
                        _filter.apply(molecule, new_node.score)
                        for _filter in self.filters
                    )
                )
                self.update(new_node)

                if duplicate is not None:
                    if len(duplicate.children) > 0:
                        new_node.children = duplicate.children
                        new_node.unexplored_neighboring_bonds = (
                            new_node.unexplored_neighboring_bonds
                        )
                    duplicate.parent.children.remove(duplicate)

                self.states_tree.id_nodes[compound.hash_id] = new_node

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
            solutions.append(
                {
                    "smiles": node.get_compound().clean_smiles(),
                    "depth": node.depth,
                    "score": node.score,
                }
            )

        return solutions

    def get_output(self, compound: Compound, reward: float, save_to_dot=False, index=0):
        """
        Returns output based on the current state of the Tree.
        For details, see README.md (around the description for output_type).

        :param compound: Compound, not actually used but necessary to have same format as other agents.
        :param reward: float, not actually used but necessary to have same format as other agents.
        :return: list(dict)
        """
        if save_to_dot:
            self.states_tree.tree_to_dot(index=index)
        return self.prepare_output(self.states_tree)
