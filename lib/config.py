import json
import logging
from typing import NamedTuple, Optional, List

from lib.calculators import CalculatorFactory
from lib.filters import FilterFactory
from lib.agents import MonteCarloTreeSearchAgent


class Config(NamedTuple):

    dataset: str
    agent: Optional[str] = "MonteCarloTreeSearch"
    generate: Optional[int] = 10

    threshold: Optional[float] = 0.15
    monte_carlo_iterations: Optional[int] = 1000
    minimum_output_depth: Optional[int] = 20
    output_type: Optional[str] = MonteCarloTreeSearchAgent.OUTPUT_FITTEST
    breath_to_depth_ratio: Optional[float] = 1

    reward_calculator: Optional[str] = CalculatorFactory.COMPOUND_ENERGY_BABEL_MMFF
    reward_weights: Optional[List[float]] = None

    kgcn_model_py: str=None
    kgcn_model: str=None


    filters: Optional[List[str]] = [FilterFactory.POSITIVE_REWARD, FilterFactory.MOLECULAR_WEIGHT]

    draw: Optional[str] = None
    logging: Optional[int] = logging.CRITICAL

    seed: Optional[int] = None

    @classmethod
    def load(cls, config_path: str) -> "Config":
        with open(config_path) as read_handle:
            return cls(**json.load(read_handle))
