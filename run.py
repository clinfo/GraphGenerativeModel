import argparse
import json
import logging

from rdkit import RDLogger

from lib.data_providers import MoleculeLoader
from lib.helpers import Sketcher
from lib.models import MonteCarloTreeSearch

RDLogger.logger().setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="Path to the input data")
parser.add_argument('--generate', type=int, default=10, help="How many molecules to generate?")
parser.add_argument('--threshold', type=float, default=0.1, help="Minimum threshold for potential bonds")
parser.add_argument('--monte_carlo_iterations', type=int, default=1000, help="How many times to iterate over the tree")
parser.add_argument('--force_field_iterations', type=int, default=100000, help="Max iterations when minimizing energy")
parser.add_argument('--minimum_output_depth', type=int, default=20, help="The output needs at least this many bonds")
parser.add_argument('--draw', type=str, required=False, help="If specified, will draw the molecules to this folder")
parser.add_argument(
    '--logging', type=int, default=logging.WARNING, help="Logging level. Smaller number means more logs"
)
parser.add_argument(
    "--output_type", type=str, default=MonteCarloTreeSearch.OUTPUT_FITTEST,
    help="Options: fittest | deepest | per_level"
)
parser.add_argument(
    "--breath_to_depth_ration", type=float, default=10000, help="Optimize for exploitation or exploration"
)
parser.add_argument(
    '--force_field', type=str, default=MonteCarloTreeSearch.FORCE_FIELD_UFF,
    help="How to calculate the energy. Options: uff | mmff"
)
args = parser.parse_args()

logging.basicConfig(format="%(message)s", level=args.logging)
molecule_loader = MoleculeLoader(file_path=args.dataset, threshold=args.threshold)
model = MonteCarloTreeSearch(
    data_provider=molecule_loader,
    force_field=args.force_field,
    minimum_depth=args.minimum_output_depth,
    output_type=args.output_type,
    breath_to_depth_ration=args.breath_to_depth_ration,
    force_field_iterations=args.force_field_iterations
)

sketcher = Sketcher()
if args.draw is not None:
    sketcher.set_location(args.draw)

for molecules in model.start(args.generate, args.monte_carlo_iterations):
    if molecules is None:
        continue

    print(json.dumps(molecules, indent=4))
    for molecule in molecules:
        sketcher.draw(molecule["smiles"])
