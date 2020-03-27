import logging
from collections import defaultdict

import numpy as np
from rdkit import Chem


class Compound(object):
    """
    Contains the atoms as an rdkit.Mol and the set of possible bonds as a list of tuples
    """
    def __init__(self, molecule, bonds):
        """
        :param molecule: rdKit.Mol
        :param bonds: list((int, int))
        """
        self.molecule = molecule
        self.bonds = bonds
        self.initial_bonds = bonds.copy()

    def get_smiles(self):
        return Chem.MolToSmiles(self.molecule)

    def get_molecule(self):
        return self.molecule

    def get_initial_bonds(self):
        """
        The set of complete/initial bonds. Before any alterations to the molecules
        That is, before any bonds are actually selected)
        :return: list((int, int))
        """
        return self.initial_bonds

    def get_bonds(self):
        """
        The set of current/remaining bonds
        :return: list((int, int))
        """
        return self.bonds

    def remove_bond(self, bond):
        """
        :param bond: (int, int)
        :return: None
        """
        self.bonds.remove(bond)

    def bonds_count(self):
        return len(self.bonds)

    def flush_bonds(self):
        """
        Set the current/remaining bonds as the complete/initial bonds.
        Helpful when the Compound is cloned
        :return: None
        """
        self.initial_bonds = self.bonds

    def clean(self, preserve=False):
        """
        Delete isolated atoms.

        These atoms have no bonds connecting to them. Ever. They will always be "alone".
        They are bad because they are useless, makes the main molecule hard to read and increases time required
        for energy minimization significantly.

        :param preserve: (bool) - When true, the local molecule will be cloned, not overwritten
        :return: rdKit.Mol
        """
        molecule = Chem.RWMol(self.molecule) if preserve else self.molecule
        atoms_to_delete = []

        for atom_index in range(molecule.GetNumAtoms()):
            if len(molecule.GetAtomWithIdx(atom_index).GetBonds()) == 0:
                atoms_to_delete.append(atom_index)

        for atom_index in sorted(atoms_to_delete, reverse=True):
            molecule.RemoveAtom(atom_index)

        return molecule

    def clone(self):
        """
        Create an identical Compound to the current on
        :return: Compound
        """
        molecule = Chem.RWMol(self.molecule)
        return Compound(molecule, self.initial_bonds.copy())


class CompoundBuilder(object):

    """
    Compound builder used by the MoleculeLoader
    """

    """Atomic Symbols Map"""
    ATOM_SYMBOL_MAPPING = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
        'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
        'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', '*'
    ]

    """Hybridization Type Map"""
    HYBRIDIZATION_MAPPING = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

    """Bonds Map"""
    BOND_TYPE_MAPPING = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.TRIPLE
    }

    def __init__(self, bonds, atoms, threshold=0.2):
        """
        :param bonds: Bond data
        :param atoms: Atom data
        :param threshold: input parameter. see README.md for details.
        """
        self.atoms = atoms
        self.bonds = bonds
        self.threshold = threshold

        self.molecule = Chem.RWMol()
        self.bonds_cache = set()

    def parse(self):
        """
        Return the atoms (as rdkit Mol) and bonds (as list of tuples)
        :return: rdKit.Mol, list((int, int))
        """
        return self.molecule, self.bonds_cache

    def get_atom_features(self, features):
        """
        :param features: atom data
        :return: dict
        """
        return {
            "symbol": self.get_chemical_symbol(features, sample=True),
            "degree": np.argmax(features[44:55]),
            "implicit_valence": np.argmax(features[55:62]),
            "formal_charge": features[62],
            "radical_electrons": features[63],
            "hybridization": self.HYBRIDIZATION_MAPPING[int(np.argmax(features[64:69]))],
            "is_aromatic": features[69],
            "hydrogen_atoms_count": np.argmax(features[70:75])
        }

    def get_chemical_symbol(self, features, sample=True):
        """
        :param features: atom data
        :param sample: perform a weighed random selection instead of an argmax
        :return: str
        """
        if sample:
            symbol_probabilities = features[0:44] / np.sum(features[0:44])
            symbol_index = np.random.choice(np.arange(len(symbol_probabilities)), p=symbol_probabilities)
        else:
            symbol_index = np.argmax(features[0:44])

        return self.ATOM_SYMBOL_MAPPING[int(symbol_index)]

    def filter_bonds(self, bonds, threshold):
        """
        Remove bonds that don't match the threshold

        :param bonds: list of bonds
        :param threshold: input parameter. see README.md for details.
        :return: filtered list of bonds
        """
        return np.where(bonds > threshold)

    def initialize_atoms(self):
        """
        Add atoms to the local molecule (rdKit.Mol)
        :return: None
        """
        for atom in self.atoms:
            if np.max(atom[0:44]) > 0.01:
                atomic_symbol = self.get_atom_features(atom).get("symbol")
                atom = Chem.Atom(atomic_symbol)

                self.molecule.AddAtom(atom)

    def initialize_bonds(self):
        """
        Sanitize, filter and validate bonds.
        Creates a list of tuples that we can easily work with
        :return: None
        """
        bonds = self.filter_bonds(self.bonds, self.threshold)
        for source_atom, destination_atom in zip(bonds[1], bonds[2]):
            if source_atom < destination_atom and (source_atom, destination_atom) not in self.bonds_cache:
                self.bonds_cache.add((source_atom, destination_atom))


class Tree(object):
    """
    Basic tree structure
    """

    """Top limit for the score"""
    INFINITY = 10e12

    class Node(object):
        """
        Tree node object
        """

        def __init__(self, compound: Compound, parent):
            """
            :param compound: Compound object
            :param parent: parent Tree.Node
            """
            self.compound = compound
            self.parent = parent
            self.children = []

            self.visits = 1
            self.score = 0
            self.performance = 0
            self.depth = 0
            
            self.valid = False

        def add_child(self, compound):
            child = Tree.Node(compound, self)
            child.depth = self.depth + 1

            self.children.append(child)
            return child

        def is_leaf_node(self):
            return len(self.children) == 0

        def get_compound(self):
            return self.compound

        def get_smiles(self):
            return self.compound.get_smiles()

    def __init__(self, root: Compound):
        self.root = Tree.Node(root, None)

    def get_depth(self, current_node=None, depth=0):
        """
        Retrieve the depth of the tree
        :param current_node: ignore. used in recursion.
        :param depth: ignore. used in recursion.
        :return: int
        """
        if current_node is None:
            current_node = self.root

        if current_node.depth > depth:
            depth = current_node.depth

        for child in current_node.children:
            depth = self.get_depth(child, depth)

        return depth

    def flatten(self, current_node=None, nodes_list=None):
        """
        Retrieve a flat list of all tree nodes
        :param current_node: ignore. used in recursion.
        :param nodes_list: ignore. used in recursion.
        :return: list(Tree.Node)
        """
        if current_node is None:
            current_node = self.root

        if nodes_list is None:
            nodes_list = []

        nodes_list.append(current_node)
        for child in current_node.children:
            if child.score < self.INFINITY:
                nodes_list = self.flatten(child, nodes_list)

        return nodes_list

    def group(self, current_node=None, nodes_list=None):
        """
        Retrieve a flat list of tree nodes, grouped by level
        :param current_node: ignore. used in recursion.
        :param nodes_list: ignore. used in recursion.
        :return: dict(int: list(Tree.Node))
        """
        if current_node is None:
            current_node = self.root

        if nodes_list is None:
            nodes_list = defaultdict(list)

        nodes_list[current_node.depth].append(current_node)
        for child in current_node.children:
            if child.score < self.INFINITY:
                nodes_list = self.group(child, nodes_list)

        return nodes_list

    def get_fittest(self, current_node=None, current_best=None):
        """
        Retrieve the node with the lowest score

        :param current_node: ignore. used in recursion.
        :param current_best: ignore. used in recursion.
        :return: Tree.Node
        """
        if current_node is None:
            current_node = self.root

        for node in current_node.children:

            is_better = current_best is None or node.score < current_best.score
            if node.valid and is_better:
                current_best = node

            if not node.is_leaf_node():
                current_best = self.get_fittest(node, current_best)

        return current_best

    def get_fittest_per_level(self):
        """
        Retrieve the node with the lowest score for each level
        :return: dict(int: Tree.Node)
        """
        all_nodes = self.flatten()
        best_nodes = {}

        for node in all_nodes:
            if node.valid and (node.depth not in best_nodes or node.score < best_nodes[node.depth].score):
                best_nodes[node.depth] = node

        return best_nodes

    def print_tree(self, node=None, best=None):
        """
        For debugging purposes. Also calculates the tree winner

        :param node: ignore. used in recursion.
        :param best: ignore. used in recursion.
        :return: None
        """
        if node is None:
            node = self.root

        if best is None:
            best = self.get_fittest()

        smiles = node.get_smiles()
        text = "Level {}: {} - {}".format(node.depth, smiles, node.score)
        if len(node.children) == 0:
            text += " (leaf)"

        if best is not None and smiles == best.get_smiles():
            text += " (winner)"

        logging.info(text)

        for child in node.children:
            self.print_tree(child)
