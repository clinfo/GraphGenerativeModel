from lib.data_structures import Tree
from typing import List
from lib.agents.mcts_agent import MonteCarloTreeSearchAgent
from lib.config import Config
import logging
from lib.calculators import AbstractCalculator
from rdkit import Chem
import numpy as np
import pickle
from lib.helpers import Sketcher

class Evaluation(object):
    """
    Keep track of the molecule generated during the training as well as some other metrics.
    """
    def __init__(self, experiment_name: str, reward_calculator: AbstractCalculator, config: Config) -> None:
        """
        :param config: config of the experiment
        """
        self.stats = []
        self.outputs = []
        self.all_mean_score_levels = {}
        self.experiment_name = experiment_name
        self.calculator = reward_calculator
        self.config = config
        self.trees = []
        self.test_metric = []

    def clean_tree(self, agent: MonteCarloTreeSearchAgent, current_node: Tree.Node = None):
        """
        breath_to_depth selection method enable the creation of compound with
        mutiple molecule. In order to evauate the model efficiently, there is a
        need to change the depth of some nodeand to recompute the scores.
        """
        if current_node is None:
            current_node = agent.states_tree.root
        for child in current_node.children:
            smile = child.compound.clean_smiles(preserve=True)
            if len(smile.split('.')) > 1:
                # Find biggest Molecule in smiles
                mols = [Chem.MolFromSmiles(s) for s in smile.split('.')]
                mol_bonds = [len(m.GetBonds()) if m is not None else 0 for m in mols]
                id_biggest_mol = np.argmax(mol_bonds)
                # Update stat of mol with biggest molecule stat
                child.depth = mol_bonds[id_biggest_mol]
                child.compound.molecule = mols[id_biggest_mol]
                child.score = self.calculator.calculate(mols[id_biggest_mol])
                child.valid = child.valid and child.depth >= self.config.minimum_output_depth and all(
                    _filter.apply(mols[id_biggest_mol], child.score) for _filter in agent.filters
                )
            self.clean_tree(agent, child)
        return agent

    def compute_metric(self, agent: MonteCarloTreeSearchAgent):
        # In case of old processing set compound with multiple molecule to the right depth
        if self.config.select_method == "breath_to_depth":
            agent = self.clean_tree(agent)

        self.trees.append(agent.states_tree)
        for mol_per_level in [1, 10]:
            mean_score_level = self.compute_mean_score(agent.states_tree, mol_per_level)
            mean_score_levels = self.all_mean_score_levels.get(mol_per_level, [])
            self.all_mean_score_levels.update({mol_per_level: mean_score_levels + [mean_score_level]})

    def compute_mean_score(self, tree, mol_per_level):
        # Retrieve level of interest
        level_of_interest = list(tree.group().items())[self.config.minimum_output_depth:]
        # Compute mean score of k best molecule
        mean_score_level = {}
        for level, list_node in level_of_interest:
            id_best_molecule = np.argsort([n.score for n in list_node])[:mol_per_level]
            mean_best_score = np.mean([n.score for n in np.array(list_node)[id_best_molecule]])
            mean_score_level.update({level: mean_best_score})
        return mean_score_level

    def compute_overall_metric(self):
        self.overall_result = {}
        for mol_per_level in [1, 10]:
            all_mean_score_levels = self.all_mean_score_levels[mol_per_level]
            overall_mean_score = self.compute_overall_mean_score(all_mean_score_levels)
            self.overall_result[mol_per_level] = overall_mean_score

    def compute_overall_mean_score(self, mean_score_levels):
        overall_mean_score = {}
        depth_score = [max(mean_score) if len(mean_score) > 0 else 0 for mean_score in mean_score_levels]
        max_depth = 0 if len(depth_score) == 0 else max(depth_score)
        for depth in range(self.config.minimum_output_depth, max_depth + 1):
            value_depth = []
            for mean_score in mean_score_levels:
                value = mean_score.get(depth, None)
                if value is None:
                    continue
                else:
                    value_depth.append(value)
            overall_mean_score[depth] = np.mean(value_depth) if len(value_depth) > 0 else None
        return overall_mean_score

    def save(self):
        """
        Save the object
        """
        filename = f'eval_{self.config.experiment_name}_{self.config.seed}.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Saved eval result to {filename}")

    def get_best_node_per_molecule(self):
        best_node = {}
        for num_mol, tree in enumerate(self.trees):
            best_node.update({num_mol: tree.get_fittest_per_level()})
        return best_node

    def generate_images(self):
        """
        Generate smiles output with score and number of bonds.
        :param output: Data store with the `save_stat` fonction of the Evaluation class
        """
        sketcher = Sketcher(self.experiment_name)
        best_mol = self.get_best_node_per_molecule()
        for num_mol, best_node in best_mol.items():
            for level, node in best_node.items():
                smiles = node.compound.clean_smiles(preserve=True)
                sketcher.draw(smiles, num_mol, node.score)

class EvaluationAggregate(object):

    def __init__(self, list_eval: List[Evaluation]):
        self.list_eval = list_eval
        self.config = list_eval[0].config

    def compact_result(self):
        self.overall_result = {}
        for mol_per_level in [1, 10]:
            self.overall_result[mol_per_level] = self.compact_overall_result(mol_per_level)


    def compact_overall_result(self, mol_per_level):
        output = {}
        depth_eval = [max(e.overall_result[mol_per_level]) if len(e.overall_result[mol_per_level]) > 0 else 0 for e in self.list_eval]
        max_depth = max(depth_eval)
        for depth in range(self.config.minimum_output_depth, max_depth + 1):
            value_depth = []
            for eval_ in self.list_eval:
                value = eval_.overall_result[mol_per_level].get(depth, None)
                if value is None:
                    continue
                else:
                    value_depth.append(value)
            output[depth] = (np.mean(value_depth), np.std(value_depth))
        return output


    def get_best_node_per_molecule(self):
        best_mol = {}
        for num_mol, level_trees in enumerate(zip(*[e.trees for e in self.list_eval])):
            best_nodes = {}
            all_nodes = []
            # Retrieving all node accross all run
            for tree in level_trees:
                all_nodes += tree.flatten()
            # Retrieving best node per level
            for node in all_nodes:
                if node.valid and (node.depth not in best_nodes or node.score < best_nodes[node.depth].score):
                    best_nodes[node.depth] = node
            best_mol.update({num_mol: best_nodes})
        return best_mol

    def draw_best_mol_per_lvl(self):
        sketcher = Sketcher(f"{self.config.experiment_name}_{self.config.seed}")
        best_mol = self.get_best_node_per_molecule()
        for num_mol, best_node in best_mol.items():
            for level, node in best_node.items():
                smiles = node.compound.clean_smiles(preserve=True)
                sketcher.draw(smiles, num_mol, node.score)

    def save(self):
        """
        Save the EvaluationAggregate object
        """
        filename = f'eval_agg_{self.config.experiment_name}_{self.config.seed}.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved eval result to {filename}")



def compact_eval(list_eval):
    mean_result = {}
    max_depth = max([max(e.mean_score_levels[0]) if len(e.mean_score_levels[0]) > 0 else 0 for e in list_eval])
    for i in range(11, max_depth+1):
        value = []
        for eval in list_eval:
            value_eval = eval.mean_score_levels[0].get(i, None)
            if value_eval is None:
                continue
            else:
                value.append(value_eval)
        mean_result[i] = np.mean(value)
    a = 0

def compare_dict(a, b):
    output = {}
    for i in np.unique(list(a.keys()) + list(b.keys())):
        output.update({i: a.get(i, 0) - b.get(i, 0)})
    return output

if __name__ == "__main__":
    # compare_evaluation("classic_42.pkl", "classic_43.pkl")
    # compact_eval(["classic_42_102.pkl", "classic_42_106.pkl", "classic_42_270.pkl", "classic_42_435.pkl", "classic_42_860.pkl",
    # "classic_42_20.pkl", "classic_42_71.pkl", "classic_42_121.pkl", "classic_42_614.pkl", "classic_42_700.pkl"])
    # compact_eval(["classic_43_255.pkl", "classic_43_277.pkl", "classic_43_320.pkl", "classic_43_817.pkl", "classic_43_836.pkl",
    # "classic_43_16.pkl", "classic_43_58.pkl", "classic_43_187.pkl", "classic_43_307.pkl", "classic_43_657.pkl"])
    # compact_eval(["random_42_102.pkl", "random_42_106.pkl", "random_42_270.pkl", "random_42_435.pkl", "random_42_860.pkl",
    # "random_42_20.pkl", "random_42_71.pkl", "random_42_121.pkl", "random_42_614.pkl", "random_42_700.pkl"])
    # compact_eval(["random_43_255.pkl", "random_43_277.pkl", "random_43_320.pkl", "random_43_817.pkl", "random_43_836.pkl",
    # "random_43_16.pkl", "random_43_58.pkl", "random_43_187.pkl", "random_43_307.pkl", "random_43_657.pkl"])
    # compact_eval(["breath_to_depth_42_102.pkl", "breath_to_depth_42_106.pkl", "breath_to_depth_42_270.pkl", "breath_to_depth_42_435.pkl", "breath_to_depth_42_860.pkl",
    # "breath_to_depth_42_20.pkl", "breath_to_depth_42_71.pkl", "breath_to_depth_42_121.pkl", "breath_to_depth_42_614.pkl", "breath_to_depth_42_700.pkl"])
    compact_eval(["breath_to_depth_43_255.pkl", "breath_to_depth_43_277.pkl", "breath_to_depth_43_320.pkl", "breath_to_depth_43_817.pkl", "breath_to_depth_43_836.pkl",
    "breath_to_depth_43_16.pkl", "breath_to_depth_43_58.pkl", "breath_to_depth_43_187.pkl", "breath_to_depth_43_307.pkl", "breath_to_depth_43_657.pkl"])