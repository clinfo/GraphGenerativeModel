import json
import logging
from typing import NamedTuple, Optional, List

from lib.calculators import CalculatorFactory
from lib.filters import FilterFactory
from lib.models import MonteCarloTreeSearch


class Config(NamedTuple):

    dataset: str
    generate: Optional[int] = 10

    threshold: Optional[float] = 0.15
    monte_carlo_iterations: Optional[int] = 1000
    minimum_output_depth: Optional[int] = 20
    output_type: Optional[str] = MonteCarloTreeSearch.OUTPUT_FITTEST
    breath_to_depth_ratio: Optional[float] = 1

    reward_calculator: Optional[str] = CalculatorFactory.ENERGY_BABEL_MMFF
    filters: Optional[List[str]] = [
        FilterFactory.NON_NAN_REWARD,
        FilterFactory.POSITIVE_REWARD,
        FilterFactory.MOLECULAR_WEIGHT
    ]

    draw: Optional[str] = None
    logging: Optional[int] = logging.CRITICAL

    @classmethod
    def load(cls, config_path: str) -> "Config":
        with open(config_path) as read_handle:
            return cls(**json.load(read_handle))
