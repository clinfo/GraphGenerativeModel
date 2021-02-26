import argparse
import json
import logging
import gym
import numpy as np
import lib.gym_mol

# For old version of tensorflow and rdkit
# if you don't use tensorflow and kgcn, please comment out this line
import tensorflow as tf


from rdkit import RDLogger

from lib.calculators import CalculatorFactory
from lib.config import Config
from lib.data_providers import MoleculeLoader
from lib.filters import FilterFactory
from lib.agents import MonteCarloTreeSearchAgent, RandomAgent, PPO2Agent

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
filters = [FilterFactory.create(filter_) for filter_ in config.filters]

env = gym.make("molecule-v0")
env.initialize(reward_calculator, config.draw)

if config.agent == "MonteCarloTreeSearch":
    agent = MonteCarloTreeSearchAgent(
        filters=filters,
        minimum_depth=config.minimum_output_depth,
        output_type=config.output_type,
        breath_to_depth_ratio=config.breath_to_depth_ratio,
    )
elif config.agent == "Random":
    agent = RandomAgent()
elif config.agent == "PPO2":
    agent = PPO2Agent(env)
else:
    raise ValueError(f"Agent: {config.agent} not implemented. Choose from 'MonteCarloTreeSearch', 'Random'")

for compound in molecule_loader.fetch(molecules_to_process=config.generate):
    env.set_compound(compound)
    observation = env.reset()
    agent.reset(compound)
    info = {"compound": compound}
    reward = 0
    done = False
    for k in range(config.monte_carlo_iterations):
        action = agent.act(observation, reward, info, done)
        observation, reward, done, info = env.step(action)
        if done:
            break

    output = agent.get_output(info["compound"])
    print(json.dumps(output, indent=4))
    for molecule in output:
        env.render(molecule["smiles"])