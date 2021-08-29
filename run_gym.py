import argparse
from functools import partial
import json
from lib.data_structures import Cycles
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
from eval import Evaluation, EvaluationAggregate
import concurrent.futures

import random


def run(config, seed=None):
    random.seed(43)
    if seed is not None:
        logging.info(f"Starting run with seed {seed}")
        np.random.seed(seed)

    molecule_loader = MoleculeLoader(
        file_path=config.dataset, threshold=config.threshold
    )
    reward_calculator = CalculatorFactory.create(
        config.reward_calculator, config.reward_weights, config
    )
    filters = [FilterFactory.create(filter_) for filter_ in config.filters]

    eval = Evaluation(
        config.experiment_name + f"_{seed}"
        if seed is not None
        else config.experiment_name,
        reward_calculator,
        config,
    )
    env = gym.make("molecule-v0")
    env.initialize(reward_calculator, config.max_mass, config.rollout_type)

    if config.agent == "MonteCarloTreeSearch":
        agent = MonteCarloTreeSearchAgent(
            filters=filters,
            minimum_depth=config.minimum_output_depth,
            output_type=config.output_type,
            select_method=config.select_method,
            breath_to_depth_ratio=config.breath_to_depth_ratio,
            tradeoff_param=config.tradeoff_param,
            force_begin_ring=config.force_begin_ring,
        )
    elif config.agent == "Random":
        agent = RandomAgent()
    else:
        raise ValueError(
            f"Agent: {config.agent} not implemented. Choose from 'MonteCarloTreeSearch', 'Random'"
        )

    for i, compound in enumerate(
        molecule_loader.fetch(molecules_to_process=config.generate)
    ):
        compound.set_cycles(
            Cycles(compound, config).get_cycles_of_sizes(config.accepted_cycle_sizes)
        )
        env.set_compound(compound)
        env.reset()
        agent.reset(compound)
        reward = score = 0
        done = False
        logging.info(f"Molecule {i}/{config.generate}, seed {seed}")
        for k in range(config.monte_carlo_iterations):
            logging.debug(
                f"Iteration {k}/{config.monte_carlo_iterations}, {Chem.MolToSmiles(compound.clean(preserve=True))}, Reward {reward}, Score {score}"
            )
            compound, action = agent.act(compound, reward, score)
            compound, reward, done, info, score = env.step(compound, action)
            if done:
                logging.info("End of generation")
                break

        output = agent.get_output(compound, reward, config.save_to_dot, i)
        if config.agent == "MonteCarloTreeSearch":
            eval.compute_metric(agent)

    if config.agent == "MonteCarloTreeSearch":
        eval.compute_overall_metric()
    return eval


def main():
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = Config.load(args.config)

    logging.basicConfig(format="%(message)s", level=config.logging)

    if config.seed is not None:
        np.random.seed(config.seed)

    if config.concurrent_run is not None:
        seeds = np.random.randint(0, 1000, config.concurrent_run)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            run_config = partial(run, config)
            evals = []
            for eval in executor.map(run_config, seeds):
                evals += [eval]
        if config.agent == "MonteCarloTreeSearch":
            eval_agg = EvaluationAggregate(evals)
            eval_agg.draw_best_mol_per_lvl()
            eval_agg.compact_result()
            print(eval_agg.overall_result)

    else:
        eval = run(config)
        if config.agent == "MonteCarloTreeSearch":
            eval.save()


if __name__ == "__main__":
    main()
