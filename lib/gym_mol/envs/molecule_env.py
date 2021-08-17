import gym
import numpy as np
import logging

from gym import spaces
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from typing import Union, Tuple

from lib.data_structures import Compound, CompoundBuilder
from lib.calculators import AbstractCalculator

import copy
import itertools

class MoleculeEnv(gym.Env):
    AVAILABLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    def initialize(self, calculator: AbstractCalculator, max_mass: int, rollout_type: str="standard"):
        """
        Own initialization function since gym.make doesn't support passing arguments.
        :param calculator: used for reward computation
        :param max_mass: maximum mass which stop the rollout phase
        :param render_location: path to save rendered molecules to
        :return: None
        """
        self.calculator = calculator
        self.max_mass = max_mass
        rollout_dict = {
            "standard": "rollout_standard",
            "no_rollout": "no_rollout"
        }
        self.rollout = getattr(self, rollout_dict[rollout_type])


    def set_compound(self, compound: Compound):
        """
        Sets the initial compound. The initial compounds implicitely defines the action space.
        :param compound: Compound
        :return: None
        """
        self.init_compound = compound.clone()

    def step(self, compound: Compound, action: Union[int, Tuple[int, int]]):
        """
        Performs a step by adding a bond to a given compound and computing the associated reward.
        :param compound: Compound to add bond to
        :param action: int or tuple(int, int) bond number or bond indexes
        :return compound: updated compound
        :return reward: reward for selected action
        :return done: whether episode is completed
        :return info: contains additional information
        """
        source_atom, destination_atom = self.action_mapper[action] if isinstance(action, int) else action
        if source_atom is None or destination_atom is None:
            # logging.debug("No bonds selected to add to compound")
            reward = None
            score = None
        elif (source_atom, destination_atom) not in compound.get_bonds():
            logging.debug("Selected bond is already in compound")
            reward = np.Infinity
            score = np.Infinity
            # reward = 0
        else:
            compound = self.add_bond(compound, source_atom, destination_atom)
            rollout_compound = self.rollout(compound)
            reward = self.calculate_reward(rollout_compound) #- 0.1 * (len(rollout_compound.molecule.GetBonds()) - len(compound.molecule.GetBonds())), 0.0001)
            score = self.calculate_reward(compound)
        done = self._is_done(compound, reward)

        info = {}
        return compound, reward, done, info, score

    def add_bond(
        self,
        compound: Compound,
        source_atom: int,
        destination_atom: int,
        select_type: str = "best",
        is_rollout: bool = False):
        """
        Adds a bond to a given compound.
        :param compound: Compound to add bond to
        :param source_atom: index of source atom
        :param destination_atom: index of destination atom
        :param select_type: "best" or "random" selection method to use for the bond type
        :param is_rollout: indicate inside the rollout function to optimize computation
        :return compound: updated Compound
        """
        # TO BE MODIFIED FOR AROMATIC MODE

        molecule = compound.get_molecule()
        target_bond_index = molecule.AddBond(int(source_atom), int(destination_atom), BondType.UNSPECIFIED)
        target_bond = molecule.GetBondWithIdx(target_bond_index - 1)

        available_bonds = compound.filter_bond_type(int(source_atom), int(destination_atom), self.AVAILABLE_BONDS)

         # if member of aromatic queue was selected
        if len(compound.get_aromatic_queue()) > 0:
            # set conjugate
            if compound.get_last_bondtype() == Chem.rdchem.BondType.SINGLE:
                bond_type = Chem.rdchem.BondType.DOUBLE
            else:
                bond_type = Chem.rdchem.BondType.SINGLE
        if select_type=="best":
            per_bond_rewards = {}
            for bond_type in available_bonds:
                target_bond.SetBondType(bond_type)
                per_bond_rewards[bond_type] = self.calculate_reward(compound)

            target_bond.SetBondType(min(per_bond_rewards, key=per_bond_rewards.get))

        elif select_type=="random":
            bond_type = available_bonds[np.random.choice(available_bonds)]
            target_bond.SetBondType(bond_type)
        else:
            raise ValueError(f"select_type must be best or random given: {select_type}")

        if not is_rollout:
            compound.add_bond_history([source_atom, destination_atom], target_bond.GetBondType())
        compound.remove_bond((source_atom, destination_atom))
        compound.flush_bonds()
        return compound

    def rollout_standard(self, compound: Compound):
        """
        Compute one rollout step, where a bond is added until the maximum mass is
        reached or there is no more bond to be added.
        :param compound: Compound
        :return Compound
        """
        compound = compound.clone()
        counter = 0
        while compound.get_mass() < self.max_mass or counter > 5:
            if len(compound.neighboring_bonds) > 0:
                id_bond = np.random.choice(range(len(compound.neighboring_bonds)))
                source_atom, destionation_atom = compound.neighboring_bonds[id_bond]
                compound = self.add_bond(compound, source_atom, destionation_atom, "best", is_rollout=True)
                counter += 1
            else:
                break
        return compound

    def no_rollout(self, compound: Compound):
        return compound

    def rollout_mdp(self, compound: Compound):
        """
        Compute one rollout step, where a bond is added until the maximum mass is
        reached or there is no more bond to be added.
        :param compound: Compound
        :return Compound
        """
        atoms = compound.get_atoms()
        atoms = set([CompoundBuilder.ATOM_SYMBOL_MAPPING[a.GetAtomMapNum()] for a in atoms])
        smiles = Chem.MolToSmiles(compound.clean(preserve=True))
        actions = self.get_valid_actions(smiles, atoms, True, True, set([4, 6]), True)
        return actions

    def calculate_reward(self, compound: Compound):
        """
        Calculate the reward of the compound based on the requested force field.
        If the molecule is not valid, the reward will be infinity.

        :param compound: Compound
        :return: float
        """
        try:
          molecule = compound.clean(preserve=True)
          smiles = Chem.MolToSmiles(molecule)

          if Chem.MolFromSmiles(smiles) is None:
              raise ValueError("Invalid molecule: {}".format(smiles))

          molecule.UpdatePropertyCache()
          reward = self.calculator.calculate(molecule)
          if np.isnan(reward):
              raise ValueError("NaN reward encountered: {}".format(smiles))

        #   logging.debug("{} : {} : {:.6f}".format(compound.get_smiles(), Chem.MolToSmiles(molecule), reward))
          return reward

        except (ValueError, RuntimeError, AttributeError) as e:
          logging.debug("[INVALID REWARD]: {} - {}".format(compound.get_smiles(), str(e)))
          return np.Infinity
        #   return 0


    def _is_done(self, compound: Compound, reward: float):
        """
        Returns whether episode is over.
        :param compound: current processed compound
        :param reward: current obtained reward
        :return bool:
        """
        if compound is None and reward is None:
            return True
        return False

    def _reset_action_space(self):
        """
        Sets action space based on initial compound.
        Action space is a discrete space of length the number of potential bonds of the initial compound.
        :param None:
        :return None:
        """
        n_bonds = len(self.init_compound.get_initial_bonds())
        self.action_space = spaces.Discrete(n_bonds)
        self.action_mapper = {k: (source_atom, destination_atom) for k, (source_atom, destination_atom) in enumerate(self.init_compound.initial_bonds)}
        self.action_mapper[-1] = (None, None)

    def reset(self):
        """
        Resets the environment
        :param None:
        :return None:
        """
        self._reset_action_space()

    def close(self):
        """
        Closes the environment.
        """
        pass

    def get_atom_valences(self, atom_types):
        """Creates a list of valences corresponding to atom_types.
        Note that this is not a count of valence electrons, but a count of the
        maximum number of bonds each element will make. For example, passing
        atom_types ['C', 'H', 'O'] will return [4, 1, 2].
        Args:
            atom_types: List of string atom types, e.g. ['C', 'H', 'O'].
        Returns:
            List of integer atom valences.
        """
        periodic_table = Chem.GetPeriodicTable()
        return [
            max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
        ]

    def get_valid_actions(self,
        state,
        atom_types,
        allow_removal,
        allow_no_modification,
        allowed_ring_sizes,
        allow_bonds_between_rings,
    ):
        """Computes the set of valid actions for a given state.
        Args:
            state: String SMILES; the current state. If None or the empty string, we
            assume an "empty" state with no atoms or bonds.
            atom_types: Set of string atom types, e.g. {'C', 'O'}.
            allow_removal: Boolean whether to allow actions that remove atoms and bonds.
            allow_no_modification: Boolean whether to include a "no-op" action.
            allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
            actions that would create rings with disallowed sizes.
            allow_bonds_between_rings: Boolean whether to allow actions that add bonds
            between atoms that are both in rings.
        Returns:
            Set of string SMILES containing the valid actions (technically, the set of
            all states that are acceptable from the given state).
        Raises:
            ValueError: If state does not represent a valid molecule.
        """
        if not state:
            # Available actions are adding a node of each type.
            return copy.deepcopy(atom_types)
        mol = Chem.MolFromSmiles(state)
        if mol is None:
            raise ValueError("Received invalid state: %s" % state)
        atom_valences = {
            atom_type: self.get_atom_valences([atom_type])[0] for atom_type in atom_types
        }
        atoms_with_free_valence = {}
        for i in range(1, max(atom_valences.values())):
            # Only atoms that allow us to replace at least one H with a new bond are
            # enumerated here.
            atoms_with_free_valence[i] = [
                atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i
            ]
        valid_actions = set()
        valid_actions.update(
            self._atom_addition(
                mol,
                atom_types=atom_types,
                atom_valences=atom_valences,
                atoms_with_free_valence=atoms_with_free_valence,
            )
        )
        valid_actions.update(
            self._bond_addition(
                mol,
                atoms_with_free_valence=atoms_with_free_valence,
                allowed_ring_sizes=allowed_ring_sizes,
                allow_bonds_between_rings=allow_bonds_between_rings,
            )
        )
        if allow_removal:
            valid_actions.update(self._bond_removal(mol))
        if allow_no_modification:
            valid_actions.add(Chem.MolToSmiles(mol))
        return valid_actions


    def _atom_addition(self, state, atom_types, atom_valences, atoms_with_free_valence):
        """Computes valid actions that involve adding atoms to the graph.
    Actions:
        * Add atom (with a bond connecting it to the existing graph)
    Each added atom is connected to the graph by a bond. There is a separate
    action for connecting to (a) each existing atom with (b) each valence-allowed
    bond type. Note that the connecting bond is only of type single, double, or
    triple (no aromatic bonds are added).
    For example, if an existing carbon atom has two empty valence positions and
    the available atom types are {'C', 'O'}, this section will produce new states
    where the existing carbon is connected to (1) another carbon by a double bond,
    (2) another carbon by a single bond, (3) an oxygen by a double bond, and
    (4) an oxygen by a single bond.
    Args:
        state: RDKit Mol.
        atom_types: Set of string atom types.
        atom_valences: Dict mapping string atom types to integer valences.
        atoms_with_free_valence: Dict mapping integer minimum available valence
        values to lists of integer atom indices. For instance, all atom indices in
        atoms_with_free_valence[2] have at least two available valence positions.
    Returns:
        Set of string SMILES; the available actions.
    """
        bond_order = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        atom_addition = set()
        for i in bond_order:
            for atom in atoms_with_free_valence[i]:
                for element in atom_types:
                    if atom_valences[element] >= i:
                        new_state = Chem.RWMol(state)
                        idx = new_state.AddAtom(Chem.Atom(element))
                        new_state.AddBond(atom, idx, bond_order[i])
                        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                        # When sanitization fails
                        if sanitization_result:
                            continue
                        atom_addition.add(Chem.MolToSmiles(new_state))
        return atom_addition


    def _bond_addition(
        self, state, atoms_with_free_valence, allowed_ring_sizes, allow_bonds_between_rings
    ):
        """Computes valid actions that involve adding bonds to the graph.
    Actions (where allowed):
        * None->{single,double,triple}
        * single->{double,triple}
        * double->{triple}
    Note that aromatic bonds are not modified.
    Args:
        state: RDKit Mol.
        atoms_with_free_valence: Dict mapping integer minimum available valence
        values to lists of integer atom indices. For instance, all atom indices in
        atoms_with_free_valence[2] have at least two available valence positions.
        allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
        actions that would create rings with disallowed sizes.
        allow_bonds_between_rings: Boolean whether to allow actions that add bonds
        between atoms that are both in rings.
    Returns:
        Set of string SMILES; the available actions.
    """
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        bond_addition = set()
        for valence, atoms in atoms_with_free_valence.items():
            for atom1, atom2 in itertools.combinations(atoms, 2):
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
                new_state = Chem.RWMol(state)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                if bond is not None:
                    if bond.GetBondType() not in bond_orders:
                        continue  # Skip aromatic bonds.
                    idx = bond.GetIdx()
                    # Compute the new bond order as an offset from the current bond order.
                    bond_order = bond_orders.index(bond.GetBondType())
                    bond_order += valence
                    if bond_order < len(bond_orders):
                        idx = bond.GetIdx()
                        bond.SetBondType(bond_orders[bond_order])
                        new_state.ReplaceBond(idx, bond)
                    else:
                        continue
                # If do not allow new bonds between atoms already in rings.
                elif not allow_bonds_between_rings and (
                    state.GetAtomWithIdx(atom1).IsInRing()
                    and state.GetAtomWithIdx(atom2).IsInRing()
                ):
                    continue
                # If the distance between the current two atoms is not in the
                # allowed ring sizes
                elif (
                    allowed_ring_sizes is not None
                    and len(Chem.rdmolops.GetShortestPath(state, atom1, atom2))
                    not in allowed_ring_sizes
                ):
                    continue
                else:
                    new_state.AddBond(atom1, atom2, bond_orders[valence])
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                bond_addition.add(Chem.MolToSmiles(new_state))
        return bond_addition


    def _bond_removal(self, state):
        """Computes valid actions that involve removing bonds from the graph.
    Actions (where allowed):
        * triple->{double,single,None}
        * double->{single,None}
        * single->{None}
    Bonds are only removed (single->None) if the resulting graph has zero or one
    disconnected atom(s); the creation of multi-atom disconnected fragments is not
    allowed. Note that aromatic bonds are not modified.
    Args:
        state: RDKit Mol.
    Returns:
        Set of string SMILES; the available actions.
    """
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        bond_removal = set()
        for valence in [1, 2, 3]:
            for bond in state.GetBonds():
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state).GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                new_state = Chem.RWMol(state)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state, clearAromaticFlags=True)
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order -= valence
                if bond_order > 0:  # Downgrade this bond.
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    bond_removal.add(Chem.MolToSmiles(new_state))
                elif bond_order == 0:  # Remove this bond entirely.
                    atom1 = bond.GetBeginAtom().GetIdx()
                    atom2 = bond.GetEndAtom().GetIdx()
                    new_state.RemoveBond(atom1, atom2)
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    smiles = Chem.MolToSmiles(new_state)
                    parts = sorted(smiles.split("."), key=len)
                    # We define the valid bond removing action set as the actions
                    # that remove an existing bond, generating only one independent
                    # molecule, or a molecule and an atom.
                    if len(parts) == 1 or len(parts[0]) == 1:
                        bond_removal.add(parts[-1])
        return bond_removal
