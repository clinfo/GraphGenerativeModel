import logging
from collections import defaultdict
import random
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from copy import deepcopy


class Compound(object):
    """
    Contains the atoms as an rdkit.Mol and the set of possible bonds as a list of tuples
    """

    def __init__(self, molecule, bonds, bonds_prediction):
        """
        :param molecule: rdKit.Mol
        :param bonds: list((int, int))
        """
        self.molecule = molecule
        self.bonds = bonds
        self.bonds_prediction = bonds_prediction
        self.initial_bonds = bonds.copy()
        self.neighboring_bonds = self.compute_neighboring_bonds()
        self.bond_history = dict()
        self.cycle_bonds = []
        self.available_cycles = []
        self.aromatic_queue = []
        self.hash_id = -1
        self.last_bondtype = 0
        self.aromatic_bonds_counter = 0

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

    def get_atoms(self, clean=False):
        if clean:
            return self.clean(preserve=True).GetAtoms()
        else:
            atom_ids = set()
            for bond in self.molecule.GetBonds():
                atom_ids.add(bond.GetBeginAtomIdx())
                atom_ids.add(bond.GetEndAtomIdx())
            return set([self.molecule.GetAtomWithIdx(id) for id in atom_ids])

    def get_atoms_id(self):
        atoms = self.get_atoms()
        return set([a.GetIdx() for a in atoms])

    def remove_bond(self, bond):
        """
        :param bond: (int, int)
        :return: None
        """
        self.bonds.remove(bond)
        # Need to recompute neighboring_bonds for consistency
        self.neighboring_bonds = self.compute_neighboring_bonds()

    def bonds_count(self):
        return len(self.bonds)

    def flush_bonds(self):
        """
        Set the current/remaining bonds as the complete/initial bonds.
        Helpful when the Compound is cloned
        :return: None
        """
        self.initial_bonds = self.bonds.copy()

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

    def clean_smiles(self, preserve=True):
        return Chem.MolToSmiles(self.clean(preserve))

    def clone(self):
        """
        Create an identical Compound to the current on
        :return: Compound
        """
        molecule = Chem.RWMol(self.molecule)
        return Compound(molecule, self.initial_bonds.copy(), self.bonds_prediction)

    def compute_neighboring_bonds(self):
        """
        Compute neighboring bonds of an node in the tree and store it in the
        neighboring_bonds variable.
        In the case of a compound with no bonds, return all possible bonds
        :return: List
        """
        self.remove_full_atom_other_bond()
        molecule = self.get_molecule()

        candidate_bonds = self.get_bonds()
        current_bonds = molecule.GetBonds()

        if len(candidate_bonds) == 0:
            # logging.debug("All bonds have been used.")
            return []

        self.neighboring_bonds = []
        if len(current_bonds) > 0:
            candidate_atoms = set()
            for bond in current_bonds:
                candidate_atoms.add(bond.GetBeginAtomIdx())
                candidate_atoms.add(bond.GetEndAtomIdx())
            for source_atom, destination_atom in candidate_bonds:
                if (
                    source_atom in candidate_atoms
                    or destination_atom in candidate_atoms
                ):
                    self.neighboring_bonds.append((source_atom, destination_atom))
        # root case
        else:
            self.neighboring_bonds = list(candidate_bonds)

        return self.neighboring_bonds

    def get_mass(self):
        """
        :return: float: mass of the molecule
        """
        mol = self.clean(preserve=True)
        atoms = mol.GetAtoms()
        return np.sum([a.GetMass() for a in atoms])

    def remove_full_atom_other_bond(self):
        """
        Remove all bonds that are link to an atom which have all its possible bond
        used.
        """
        full_atom_id = self.get_full_atoms_id()
        bond_to_remove = []
        for bond in self.bonds:
            for a in full_atom_id:
                if a in bond:
                    bond_to_remove.append(bond)
        [self.bonds.remove(b) for b in bond_to_remove]

    def get_full_atoms_id(self):
        """
        Return all atom id of the molecule which can't have any additional bond
        :return: List(int): Id of the atoms
        """
        return [
            i for i, a in enumerate(self.molecule.GetAtoms()) if self.is_atom_full(a)
        ]

    def is_atom_full(self, atom):
        """
        Compute if an atom can receive another bond.
        :return: int:
        """
        return self.get_free_valence(atom) == 0

    def get_free_valence(self, atom):
        """
        Get the free valence number that can accept new bonds.
        :param atom: Chem.Atom
        :return: int:
        """
        atom.UpdatePropertyCache()
        valence = atom.GetTotalValence()
        used = np.sum([b.GetValenceContrib(atom) for b in atom.GetBonds()])
        return valence - used

    def filter_bond_type(
        self, source_atom_id, destination_atom_id, available_bond_type
    ):
        """
        Filter bond type possible between the two selected atoms.
        :param source_atom_id: int
        :param destination_atom_id: int
        :param available_bond_type: List[BondType]
        :return: List[BondType]
        """
        available_bond_type = available_bond_type.copy()
        source_atom = self.molecule.GetAtomWithIdx(source_atom_id)
        destination_atom = self.molecule.GetAtomWithIdx(destination_atom_id)
        valence_possible = min(
            [
                self.get_free_valence(source_atom),
                self.get_free_valence(destination_atom),
            ]
        )
        for i, bond in list(enumerate(available_bond_type))[::-1]:
            if valence_possible < int(bond):
                available_bond_type.pop(i)
        return available_bond_type

    def add_bond_history(self, selected_bond, bond_type):
        """
        Add the tuple of atom id of the new bonds to the bond history.
        Used to create a hash for each compound.
        Example of an entry:
        key (str): (7, 12)
        value (BondType): BondType.Single
        :param selected_bond: Tuple(int, int)
        :param bond_type: BondType, selected bond type
        """
        self.last_bondtype = bond_type
        #print(self.last_bondtype)
        self.bond_history[str(selected_bond)] = bond_type

    def pass_parent_info(self, parent_compound):
        self.cycle_bonds = parent_compound.cycle_bonds.copy()
        self.bond_history.update(parent_compound.bond_history)
        self.aromatic_queue = parent_compound.aromatic_queue.copy()
        self.last_bondtype = parent_compound.last_bondtype

        # delete parent aromatic queue
        parent_compound.aromatic_queue = []

    def compute_hash(self):
        """
        Compute a hash based on the bond history to avoid duplicate nodes
        :return: int
        """
        self.hash_id = hash(str(sorted(self.bond_history.items())))
        return self.hash_id

    def get_pred_proba_next_bond(self, possible_bonds):
        """
        Return a probability for the possible bonds based on prediction outputed by
        kgcn.
        :param possible_bonds: possible new bonds to expand the compound
        :return: list of probabilities for each possible bonds
        """
        predition_score = [self.bonds_prediction[str(b)] for b in possible_bonds]
        return predition_score / sum(predition_score)

    def get_atom_id_used(self):
        output = []
        for bond_string in self.bond_history.keys():
            output += re.findall("\d+", bond_string)
        return np.sort(np.unique(output))

    def set_cycles(self, cycles):
        self.cycle_bonds = cycles
        print(self.cycle_bonds)

    def compute_available_cycles(self):
        """
        Compute the list of available cycles from the current molecule
        :return: list
        """
        self.remove_full_atom_other_bond()
        molecule = self.get_molecule()
        # print("self.cycle_bonds", self.cycle_bonds)
        candidate_cycles = deepcopy(self.cycle_bonds)
        sorted_bonds = [sorted(bond) for bond in self.get_bonds()]
        current_bonds = molecule.GetBonds()
        self.available_cycles = []

        if len(current_bonds) > 0:
            for cycle in candidate_cycles:
                cleanup = False
                # cleanup and break cycle is using a non available bond
                for bond in cycle:
                    if bond not in sorted_bonds:
                        self.cycle_bonds.remove(cycle)
                        cleanup = True
                        break

                if cleanup:
                    break

                candidate_atoms = set()

                for bond in current_bonds:
                    candidate_atoms.add(bond.GetBeginAtomIdx())
                    candidate_atoms.add(bond.GetEndAtomIdx())

                for source_atom, destination_atom in cycle:
                    if (
                        source_atom in candidate_atoms
                        or destination_atom in candidate_atoms
                    ):
                        self.available_cycles.append(cycle)
                        break
        # root case
        else:
            self.available_cycles = candidate_cycles

        # 2 verions of the cycles are allowed so we duplicate every available cycles_within_cycles
        self.available_cycles = [
            cycle for cycle in self.available_cycles for _ in range(2)
        ]
        print("self.available_cycles = ", self.available_cycles)
        return self.available_cycles

    def fill_aromatic_queue(self):
        """
        Pass the current aromatic queue if not empty
        If empty, compute available cycles then pick a cycle among the available cycles
        :return: list
        """
        if self.aromatic_queue == []:
            if len(self.available_cycles) > 0:
                self.aromatic_queue = self.available_cycles.pop(0)
                return self.aromatic_queue
        else:
            return self.aromatic_queue
        return []

    def get_aromatic_queue(self):
        return self.aromatic_queue

    def reset_aromatic_queue(self):
        self.aromatic_queue = []

    def get_last_bondtype(self):
        return self.last_bondtype

    def is_aromatic(self):
        """
        Check if whole compound is aromatic
        :return: bool
        """

        ri = self.clean(preserve=True).GetRingInfo()
        print("------------> ", self.clean_smiles())
        print("------------> ", ri.AtomRings())
        #print(self.molecule.GetRingInfo().ri.AtomRings())
        if len(ri.AtomRings()) == 0:
            return False

        for id in ri.BondRings():
            if not self.molecule.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True


class CompoundBuilder(object):

    """
    Compound builder used by the MoleculeLoader
    """

    """Atomic Symbols Map"""
    ATOM_SYMBOL_MAPPING = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "H",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "*",
    ]

    """Hybridization Type Map"""
    HYBRIDIZATION_MAPPING = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    """Bonds Map"""
    BOND_TYPE_MAPPING = {
        0: Chem.rdchem.BondType.SINGLE,
        1: Chem.rdchem.BondType.DOUBLE,
        2: Chem.rdchem.BondType.TRIPLE,
        3: Chem.rdchem.BondType.AROMATIC,
        4: Chem.rdchem.BondType.OTHER,
        5: "Conjugate",
    }

    def __init__(self, bonds, atoms, threshold=0.2):
        """
        :param bonds: Bond data
        :param atoms: Atom data
        :param threshold: input parameter. see README.md for details.
        """
        self.bonds = bonds
        self.threshold = threshold

        self.atoms = atoms
        self.is_atom_valid = np.max(np.array(self.atoms)[:, 0:44], axis=1) > 0.01

        self.molecule = Chem.RWMol()
        self.bonds_cache = set()
        self.bonds_prediction = dict()

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
        result = {
            "symbol": self.get_chemical_symbol(features, sample=True),
            "degree": np.argmax(features[44:55]),
            "implicit_valence": np.argmax(features[55:62]),
            "formal_charge": features[62],
            "radical_electrons": features[63],
            "hybridization": self.HYBRIDIZATION_MAPPING[
                int(np.argmax(features[64:69]))
            ],
            "is_aromatic": features[69],
            "hydrogen_atoms_count": np.argmax(features[70:75]),
        }
        if len(features) >= 76:
            ring_size = list(range(3, 8))
            result["ring"] = (features[75],)
            result["ring_size"] = ring_size[np.argmax(features[76:81])]
        return result

    def get_chemical_symbol(self, features, sample=True):
        """
        :param features: atom data
        :param sample: perform a weighed random selection instead of an argmax
        :return: str
        """
        if sample:
            symbol_probabilities = features[0:44] / np.sum(features[0:44])
            # symbol_index = np.random.choice(np.arange(len(symbol_probabilities)), p=symbol_probabilities)
            symbol_index = random.choices(
                np.arange(len(symbol_probabilities)), weights=symbol_probabilities
            )[0]
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
        for index, atom in enumerate(self.atoms):
            if self.is_atom_valid[index]:
                atomic_symbol = self.get_atom_features(atom).get("symbol")
                atom = Chem.Atom(atomic_symbol)

                self.molecule.AddAtom(atom)

    def initialize_bonds(self):
        """
        Sanitize, filter and validate bonds.
        Creates a list of tuples that we can easily work with
        :return: None
        """
        invalid_atoms = np.logical_not(self.is_atom_valid)
        bonds = self.filter_bonds(self.bonds, self.threshold)

        for source_atom, destination_atom in zip(bonds[1], bonds[2]):
            bond = (
                source_atom - sum(invalid_atoms[:source_atom]),
                destination_atom - sum(invalid_atoms[:destination_atom]),
            )
            if (
                source_atom < destination_atom
                and self.is_atom_valid[source_atom]
                and self.is_atom_valid[destination_atom]
                and bond not in self.bonds_cache
            ):

                self.bonds_cache.add(bond)
                # Used for random mode
                self.bonds_prediction.update(
                    {str(bond): max(self.bonds[:, source_atom, destination_atom])}
                )


class Cycles:
    """
    This class contains tools to extracts all the possible cycles within a given molecule
    Used only once in Tree class
    """

    def __init__(self, compound):
        self.compound = compound
        self.bonds = compound.get_bonds()

        self.all_atoms = self.get_all_atoms()
        self.n_atoms = self.get_atoms_number()

        self.adj_matrix = self.get_adjacency_matrix()

        self.marks = [0] * (self.n_atoms + 1)
        self.parents = [0] * (self.n_atoms + 1)
        self.states = [0] * (self.n_atoms + 1)

        self.cycle_number = 0
        self.cycles = [[] for i in range(100)]
        self.subcycles = []
        self.cycle_sorted_pairs = []

        self.compute_all_cycles()

    def get_adjacency_matrix(self):
        """
        get the adjacency matrix from list of bonds
        :return:
        """
        adj_matrix = [[] for i in range(self.n_atoms + 1)]
        for bond in self.bonds:
            i, j = bond
            adj_matrix[i].append(j)
            adj_matrix[j].append(i)
        return adj_matrix

    def get_all_atoms(self):
        """
        get list of all atoms coordonated from bond list
        :return: list
        """
        atoms = [coord for coords in self.bonds for coord in coords]
        return list(set(atoms))

    def get_atoms_number(self):
        """
        get the number of atoms
        :return: int
        """
        return max(self.all_atoms)

    def DFScycles(self, current, parent):
        """
        Use DFS and painting algorithm to detect large cycles
        :return: None
        """
        if self.states[current] == 2:
            return

        if self.states[current] == 1:
            self.cycle_number += 1
            backtrack = parent

            self.cycles[self.cycle_number].append(parent)

            self.states[backtrack] = 1

            while backtrack != current:
                backtrack = self.parents[backtrack]
                self.cycles[self.cycle_number].append(backtrack)
            return

        self.parents[current] = parent
        self.states[current] = 1

        for neighbor in self.adj_matrix[current]:
            if neighbor != parent:
                self.DFScycles(neighbor, current)

        self.states[current] = 2

    def compute_large_cycles(self):
        """
        Use DFScycle to detect large cycles from bonds using the adjacency matrix
        :return: None
        """
        self.parents = [0] * (self.n_atoms + 1)
        self.states = [0] * (self.n_atoms + 1)
        self.cycle_number = 0

        self.DFScycles(2, 0)

    def cycles_within_cycle(
        self, cycle, current, start, parent, max_depth, path, depth
    ):
        """
        Detect any subcycle in larger cycle with recursion
        :return: None
        """
        if depth > max_depth:
            return

        elif current == start and depth > 1:
            self.subcycles.append(path)
            return

        path.append(current)

        for neighbor in self.adj_matrix[current]:
            if neighbor in cycle and neighbor != parent and neighbor not in path[1:]:
                self.cycles_within_cycle(
                    cycle, neighbor, start, current, max_depth, path.copy(), depth + 1
                )

    def compute_all_subcycles(self):
        """
        Detect any subcycle in all larger cycles
        :return: None
        """
        for cycle in self.cycles:
            if len(cycle) > 3:
                for id in cycle:
                    self.cycles_within_cycle(cycle, id, id, 0, len(cycle), [], 0)

    def get_cycle_bonds(self, cycle):
        cycle_bonds = []
        for i, node in enumerate(cycle):
            bond = [cycle[i], cycle[(i + 1) % len(cycle)]]
            bond = sorted(bond)
            cycle_bonds.append(bond)
        return cycle_bonds

    def clean_cycles(self):
        self.cycles = list(filter(None, self.cycles))

    def remove_duplicates(self):
        """
        Remove all potential cycle duplicates
        :return: None
        """
        cleaned_cycles = []
        for cycle in self.cycles:
            cycle_bonds = self.get_cycle_bonds(cycle)
            sorted_pairs = sorted(cycle_bonds)
            if sorted_pairs not in self.cycle_sorted_pairs:
                cleaned_cycles.append(cycle)
                self.cycle_sorted_pairs.append(sorted_pairs)

        self.cycles = cleaned_cycles

    def get_cycles_of_sizes(self, accepted_sizes=[3, 5, 6]):
        return [x for x in self.cycle_sorted_pairs if len(x) in accepted_sizes]

    def get_cycles(self):
        return self.cycles

    def get_cycle_pairs(self):
        return self.cycle_sorted_pairs

    def get_subcycles(self):
        return self.subcycles

    def compute_all_cycles(self):
        self.compute_large_cycles()
        self.clean_cycles()
        self.compute_all_subcycles()
        self.cycles = self.cycles + self.subcycles
        self.remove_duplicates()


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
            self.unexplored_neighboring_bonds = (
                self.get_compound().compute_neighboring_bonds()
            )
            self.children = []
            self.parent = parent
            self.visits = 0
            self.score = 0
            self.selection_score = 0
            self.performance = 0
            self.depth = 0
            self.valid = False
            self.get_compound().compute_available_cycles()
            self.cleaned_smiles = self.compute_clean_smiles()


        def add_child(self, compound):
            child = Tree.Node(compound, self)
            child.depth = self.depth + 1

            self.children.append(child)
            return child

        def is_expended(self):
            """
            Are all children nodes created ?
            """
            return len(self.unexplored_neighboring_bonds) == 0

        def is_terminal(self):
            if self.is_expended():
                return len(self.children) == 0
            else:
                return len(self.get_compound().neighboring_bonds) == 0

        def is_leaf_node(self):
            return len(self.children) == 0

        def get_compound(self):
            return self.compound

        def get_smiles(self):
            return self.compound.get_smiles()

        def compute_clean_smiles(self):
            return self.compound.clean_smiles()

        def get_clean_smiles(self):
            return self.cleaned_smiles

    def __init__(self, root: Compound):
        root.set_cycles(Cycles(root).get_cycles_of_sizes(accepted_sizes=[6]))
        self.root = Tree.Node(root, None)
        self.id_nodes = dict()

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
            if node.valid and (
                node.depth not in best_nodes
                or node.score < best_nodes[node.depth].score
            ):
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

    def find_duplicate(self, compound):
        """
        Retrieve if it exist the node that correspond to the same compound.
        :param compound:Compound
        :return: Tree.Node
        """
        return self.id_nodes.get(compound.hash_id, None)

    def tree_to_dot(self, index=0, clean=True):
        """
        Convert our tree to dot format
        :param node: ignore. used in recursion.
        :param best: ignore. used in recursion.
        :return: None
        """
        all_nodes = self.flatten()
        unique_edge_list = []

        dot_graph = "digraph G { \n overlap = scale; \n"

        for node in all_nodes:
            for child in node.children:

                performance_indice = int(
                    child.performance / (child.visits + child.prior_occurence)
                )
                label = " [label= " + str(performance_indice) + "];"

                if node == self.root:
                    edge = '"root" -> "' + child.get_clean_smiles() + '"'
                else:
                    edge = (
                        '"'
                        + node.get_clean_smiles()
                        + '" -> "'
                        + child.get_clean_smiles()
                        + '"'
                    )

                if clean:
                    if edge not in unique_edge_list:
                        unique_edge_list.append(edge)
                        dot_graph += edge + "\n"
                else:
                    dot_graph += edge + label + "\n"

        dot_graph += "}"

        file = open("test/dot_graph_" + str(index) + ".gv", "wt")
        file.write(dot_graph)
        file.close()
