# Molecule Optimization with Monte Carlo Tree Search

We attempt to optimize the molecules generated from the GCN-K adjacency matrices using a Monte Carlo Tree Search approach.
To generate the input data, please follow the instructions from the [GCN-K repository](https://github.com/clinfo/GraphCNN)
or reach out to the person in charge from Kyoto University.

## Installation & Requirements

You can use conda for the setup. First, create the virtual environment:
```bash
conda env create -f environment.yml
```

Then use it whenever you need it:
```bash
conda activate graph_mcts
```

## File Structure

Relevant files:
- `lib/data_providers.py` - Loads, prepares and servers the molecules from the input .jbl files
- `lib/data_structures.py` - Data models and resources for compounds, nodes and trees
- `lib/energy_calculators.py` - Energy calculation/minimization tools
- `lib/helpers.py` - Helper classes (ie: drawing molecules)
- `lib/models.py` - Monte Carlo Tree Search implementation
- `environment.yml` - Requirements and dependencies
- `run.py` - Command line interface. You use this file to run the optimizer.

Trivial files:
- `data/*` - Sample datasets
- `img/*` - Sample molecules generated (sketched)
- `logs/*` - Sample output and performance measurements

## Usage

`run.py` is the entry point for the optimizer. You can display a help menu by running:
```bash
python run.py -h

    usage: run.py [-h] --dataset DATASET [--generate GENERATE]
              [--threshold THRESHOLD]
              [--monte_carlo_iterations MONTE_CARLO_ITERATIONS]
              [--minimum_output_depth MINIMUM_OUTPUT_DEPTH] [--draw DRAW]
              [--logging LOGGING] [--output_type OUTPUT_TYPE]
              [--breath_to_depth_ratio BREATH_TO_DEPTH_RATIO]
              [--energy_calculator ENERGY_CALCULATOR]

    arguments:
        -h, --help                  Show this help message and exit
        --dataset                   Path to the input data
        --generate                  How many molecules to generate?
        --threshold                 Minimum threshold for potential bonds
        --monte_carlo_iterations    How many times to iterate over the tree
        --minimum_output_depth      The output needs at least this many bonds
        --draw                      If specified, will draw the molecules to this folder
        --logging                   Logging level. Smaller number means more logs
        --output_type               Options: fittest | deepest | per_level
        --breath_to_depth_ratio     Optimize for exploitation or exploration
        --energy_calculator         How to calculate the energy. Options: rdkit_uff | rdkit_mmff | babel_uff | babel_mmff94 | babel_mmff94s | babel_gaff | babel_ghemical

```

### threshold

The adjacency matrix provided by GCN-K has a specific structure. For each molecule, each atom has a probability
to be of a certain type (ie: atom #1 could have a 70% chance to be C, 20% to be O and 10% to be one of the other 
42 types the model allows). Similarly, each atom pair has a certain probability to have a bond between them.

With this threshold, you can control the minimum value required to consider a bond. As it is a percentage, 
it accepts values between 0 and 1

A large threshold will have very small bond to atom ratio, while a very small one will have a very high one. This
will result in sparse, several small molecules in the first case, or very unstable molecules in the second one. 
Values between 0.10 and 0.15 seem to work best.

![](img/15664526171860650.png)
*threshold=0.25, too big*

![](img/15664525442788234.png)
*threshold=0.05, too small*

### monte_carlo_iterations
Monte Carlo Tree Search is an optimization algorithm that runs for an infinite number of iterations. 
Use this parameter to specify when to stop.

In our approach, we start from the set of atoms, with no bonds between them and add a new one in each iteration.
If the parameter is too small, it might not be enough to add enough bonds to form a large molecule, or an optimal one.

Generally, the deepest levels of the tree are not yet good enough because only a few iterations had the opportunity 
to expand on them.

Note: this parameter influences execution time the most

### force_field
The energy of the molecule is used as reward for the algorithm. The smaller the energy, the better.
To calculate the energy we need to create a force field, and there are 2 types of implementation for 
supported force fields: rdkit force fields and open babel force fields.


7 options are available for energy calculations:

- `rdkit_uff` - The [Universal Force Field](https://doi.org/10.1021/ja00051a040) is an all atom potential which considers 
only the element, the hybridization and the connectivity (implemented in rdKit)
- `rdkit_mmff` - The [Merck Molecular Force Field](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P) 
is similar to the [MM3 Force Field](https://doi.org/10.1021/ja00205a001) (implemented in rdKit)
- `babel_uff` - The same [Universal Force Field](https://doi.org/10.1021/ja00051a040) (implemented in OpenBabel)
- `babel_mmff94` - The same [Merck Molecular Force Field](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P)
(implemented in OpenBabel)
- `babel_mmff94s` - [MMFF94S](https://doi.org/10.1002/(SICI)1096-987X(199905)20:7%3C720::AID-JCC7%3E3.0.CO;2-X) is 
a "static" variant of the MMFF force field. (implemented in OpenBabel)
- `babel_gaff` - The [Generalized Amber Force Field](https://doi.org/10.1002/jcc.20035) (implemented in OpenBabel)
- `babel_ghemical` - The [Ghemical Force Field](https://open-babel.readthedocs.io/en/latest/Forcefields/ghemical.html) (implemented in OpenBabel)



 
### output_type
We implemented 3 different ways to select the output/best solution:

- **fittest** - Will output the molecule with the smallest energy. But note, smaller molecules tend to be 
more stable and have smaller energy, thus this approach tends to output only a C-C molecule or something similar. 
Always use this option along with the "minimum_output_depth" parameter. 

- **deepest** - This approach will output the fittest molecule, but only if it is a node from the deepest level 
of the tree. The molecules from the deepest levels are usually not stable enough, since the states haven't 
been visited many times.

- **per_level** - Several molecules will be printed, one for the best molecule from each level of the tree. 
Since the tree can become very deep, it is recommended to use this option along with "minimum_output_depth" as well.    

### minimum_output_depth
Sets the minimum tree level to look at when picking the winner. 
Nodes with depth smaller than "minimum_output_depth" are ignored.

### breath_to_depth_ratio
Molecule energy is not a good way to select the node to expand since it tends to favor smaller molecules.
The best working solution we found is a two-factor pseudo-random one.

##### Step 1
We perform a weighted random choice for the level to perform the expansion on based on the "breath_to_depth_ratio".
The higher the value, the more "random" the selection will be. Lower values will result in a values that are 
more skewed towards deeper levels. To achieve an opposite effect, use a negative value:

As an example, assuming our tree currently has a depth/level of 5. The following "breath_to_depth_ratio" values might
produce similar probabilities:

| breath_to_depth_ratio | level1 % | level2 % | level3 % | level4 % | level5 % |
| --------------------- | -------- | -------- | -------- | -------- | -------- |
| 0.5                   | 0.000047 | 0.000062 | 0.003793 | 0.360444 | 0.635651 |
| 1                     | 0.048671 | 0.087203 | 0.160737 | 0.291216 | 0.412170 |
| 10                    | 0.106983 | 0.171757 | 0.180209 | 0.217732 | 0.323316 |
| 100                   | 0.188933 | 0.192691 | 0.201109 | 0.208459 | 0.208805 |
| -0.1                  | 0.993638 | 0.006341 | 0.000014 | 0.000006 | 0        |
| -1                    | 0.410542 | 0.334045 | 0.138279 | 0.111738 | 0.005394 |
| -10                   | 0.219435 | 0.219109 | 0.204048 | 0.184172 | 0.173234 |
| -100000               | 0.200854 | 0.200463 | 0.200139 | 0.200045 | 0.198496 |

We recommend using low, positive values.

##### Step 2
Once a level is selected, we perform a weighted random choice based on the fitness of each node in the level.

### logging
The value of this parameter is an integer (use increments of 10). When the value is:
- 10 - everything is usually logged, including energy calculations and exceptions
- 20 - only the molecules output and its energy is logged
- 30+ - only the result is logged

### example

```
python -u run.py --dataset=data/gen.multi.test.jbl --generate=10 --monte_carlo_iterations=1000 --threshold=0.10 --logging=50 --minimum_output_depth=20 --output_type=per_level --breath_to_depth_ratio=0.5 --energy_calculator=rdkit_uff --draw=test_rduff/
```
