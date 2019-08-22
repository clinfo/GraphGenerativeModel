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
              [--force_field_iterations FORCE_FIELD_ITERATIONS]
              [--minimum_output_depth MINIMUM_OUTPUT_DEPTH] [--draw DRAW]
              [--logging LOGGING] [--output_type OUTPUT_TYPE]
              [--breath_to_depth_ration BREATH_TO_DEPTH_RATION]
              [--force_field FORCE_FIELD]

    arguments:
        -h, --help                  Show this help message and exit
        --dataset                   Path to the input data
        --generate                  How many molecules to generate?
        --threshold                 Minimum threshold for potential bonds
        --monte_carlo_iterations    How many times to iterate over the tree
        --force_field_iterations    Max iterations when minimizing energy
        --minimum_output_depth      The output needs at least this many bonds
        --draw                      If specified, will draw the molecules to this folder
        --logging                   Logging level. Smaller number means more logs
        --output_type               Options: fittest | deepest | per_level
        --breath_to_depth_ration    Optimize for exploitation or exploration
        --force_field               How to calculate the energy. Options: uff | mmff

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
To calculate the energy we need to create a force field. There are 2 force field implementations
you can choose from:

- `uff` - The [Universal Force Field](https://doi.org/10.1021/ja00051a040) is an all atom potential which considers 
only the element, the hybridization and the connectivity 
- `mmff` - The [Merck Molecular Force Field](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P) 
is similar to the [MM3 Force Field](https://doi.org/10.1021/ja00205a001)
 
### force_field_iterations
Energy minimization is an optimization problem in itself. The molecule structures are solid, so it is
safe to use very large values here. The minimization should converge fast.

### output_type
We implemented 3 different ways to select the output/best solution:

- **fittest** - Will output the molecule with the smallest energy. But note, smaller molecules then to be 
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

### breath_to_depth_ration
Molecule energy is not a good way to select the node to expand since it tends to favor smaller molecules.
The best working solution we found is a two-factor pseudo-random one.

##### Step 1
We perform a weighted random choice for the level to perform the expansion on based on the "breath_to_depth_ration".
When the value is 0, each level has an equal chance to be selected, a positive value will favour deeper levels and 
a negative one will prefer higher ones. From a certain point onward, to achieve a significant difference the value 
needs to increase exponentially. 

We recommend using a high value, in the order of 10,000+.  

##### Step 2
Once a level is selected, we perform a weighted random choice based on the fitness of each node in the level.

### logging
The value of this parameter is an integer (use increments of 10). When the value is:
- 10 - everything is usually logged, including energy calculations and exceptions
- 20 - only the molecules output and its energy is logged
- 30+ - only the result is logged
