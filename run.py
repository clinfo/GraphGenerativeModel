import argparse
import json
import logging

from rdkit import RDLogger

from lib.calculators import CalculatorFactory
from lib.config import Config
from lib.data_providers import MoleculeLoader
from lib.filters import FilterFactory
from lib.helpers import Sketcher
from lib.models import MonteCarloTreeSearch

RDLogger.logger().setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

config = Config.load(args.config)

logging.basicConfig(format="%(message)s", level=config.logging)
molecule_loader = MoleculeLoader(file_path=config.dataset, threshold=config.threshold)
reward_calculator = CalculatorFactory.create(config.reward_calculator, config.reward_weights, config)
filters = [FilterFactory.create(filter_) for filter_ in config.filters]

model = MonteCarloTreeSearch(
    data_provider = molecule_loader,
    calculator=reward_calculator,
    minimum_depth=config.minimum_output_depth,
    output_type=config.output_type,
    filters=filters,
    select_method=config.select_method,
    breath_to_depth_ratio=config.breath_to_depth_ratio,
    tradeoff_param=config.tradeoff_param,
    force_begin_ring=config.force_begin_ring,
    save_to_dot=config.save_to_dot,
    max_mass=config.max_mass,
)
"""
sketcher = Sketcher()
if config.draw is not None:
    sketcher.set_location(config.draw)
"""

for molecules in model.start(config.generate, config.monte_carlo_iterations):
    if molecules is None:
        continue

    print(json.dumps(molecules, indent=4))
    #for molecule in molecules:
    #    sketcher.draw(molecule["smiles"])
