import json
import logging
from typing import NamedTuple, Optional, List

from lib.calculators import CalculatorFactory
from lib.filters import FilterFactory
from lib.agents import MonteCarloTreeSearchAgent


class Config(NamedTuple):

    dataset: str
    experiment_name: str
    agent: Optional[str] = "MonteCarloTreeSearch"
    generate: Optional[int] = 10
    concurrent_run: Optional[int] = None

    select_method: Optional[str] = "MCTS_classic"
    rollout_type: Optional[str] = "standard"
    threshold: Optional[float] = 0.14
    monte_carlo_iterations: Optional[int] = 1000
    minimum_output_depth: Optional[int] = 20
    output_type: Optional[str] = MonteCarloTreeSearchAgent.OUTPUT_FITTEST
    breath_to_depth_ratio: Optional[float] = 1
    tradeoff_param: Optional[float] = 0
    max_mass: int = 100
    accepted_cycle_sizes: List[int] = [5, 6]
    force_begin_ring: bool = False
    save_to_dot: bool = False

    reward_calculator: Optional[str] = CalculatorFactory.COMPOUND_ENERGY_BABEL_MMFF
    reward_weights: Optional[List[float]] = None
    tanimoto_smiles: Optional[str] = None

    kgcn_model_py: str = None
    kgcn_model: str = None

    filters: Optional[List[str]] = [
        FilterFactory.POSITIVE_REWARD,
        FilterFactory.MOLECULAR_WEIGHT,
    ]

    draw: Optional[str] = None
    logging: Optional[int] = logging.CRITICAL

    seed: Optional[int] = None

    @classmethod
    def load(cls, config_path: str) -> "Config":
        with open(config_path) as read_handle:
            return cls(**json.load(read_handle))
