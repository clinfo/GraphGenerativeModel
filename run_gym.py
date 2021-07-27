import argparse
import json
import logging
import gym
import numpy as np
import lib.gym_mol
from rdkit import Chem

# For old version of tensorflow and rdkit
# if you don't use tensorflow and kgcn, please comment out this line
# import tensorflow as tf


from rdkit import RDLogger

from lib.calculators import CalculatorFactory
from lib.config import Config
from lib.data_providers import MoleculeLoader
from lib.filters import FilterFactory
from lib.agents import MonteCarloTreeSearchAgent, RandomAgent
from lib.helpers import Sketcher
from eval import Evaluation

RDLogger.logger().setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

config = Config.load(args.config)

if config.seed is not None:
    np.random.seed(config.seed)

logging.basicConfig(format="%(message)s", level=config.logging)
molecule_loader = MoleculeLoader(file_path=config.dataset, threshold=config.threshold)
reward_calculator = CalculatorFactory.create(config.reward_calculator, config.reward_weights, config)
sketcher = Sketcher(config.experiment_name)
filters = [FilterFactory.create(filter_) for filter_ in config.filters]

eval = Evaluation(config.experiment_name, reward_calculator)
env = gym.make("molecule-v0")
env.initialize(reward_calculator, config.max_mass, config.rollout_type)

if config.agent == "MonteCarloTreeSearch":
    agent = MonteCarloTreeSearchAgent(
        filters=filters,
        minimum_depth=config.minimum_output_depth,
        output_type=config.output_type,
        select_method=config.select_method,
        breath_to_depth_ratio=config.breath_to_depth_ratio,
        tradeoff_param=config.tradeoff_param
    )
elif config.agent == "Random":
    agent = RandomAgent()
else:
    raise ValueError(f"Agent: {config.agent} not implemented. Choose from 'MonteCarloTreeSearch', 'Random'")

for i, compound in enumerate(molecule_loader.fetch(molecules_to_process=config.generate)):
    env.set_compound(compound)
    env.reset()
    agent.reset(compound)
    reward = 0
    done = False
    for k in range(config.monte_carlo_iterations):
        logging.debug(f"Iteration {k}/{config.monte_carlo_iterations}, {Chem.MolToSmiles(compound.clean(preserve=True))}, Reward {reward}")
        compound, action = agent.act(compound, reward)
        compound, reward, done, info = env.step(compound, action)
        if done:
            logging.info("End of generation")
            break

    output = agent.get_output(compound, reward)
    print(json.dumps(output, indent=4))
    eval.add_output(output)
    # for molecule in output:
        # sketcher.draw(molecule["smiles"], i)
eval.save_stat(config)
