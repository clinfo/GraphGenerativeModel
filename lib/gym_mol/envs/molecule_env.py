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
    AVAILABLE_BONDS = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]

    def initialize(
        self,
        calculator: AbstractCalculator,
        max_mass: int,
        rollout_type: str = "standard",
    ):
        """
        Own initialization function since gym.make doesn't support passing arguments.
        :param calculator: used for reward computation
        :param max_mass: maximum mass which stop the rollout phase
        :param render_location: path to save rendered molecules to
        :return: None
        """
        self.calculator = calculator
        self.max_mass = max_mass
        rollout_dict = {"standard": "rollout_standard", "no_rollout": "no_rollout"}
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
        source_atom, destination_atom = (
            self.action_mapper[action] if isinstance(action, int) else action
        )
        if source_atom is None or destination_atom is None:
            # logging.debug("No bonds selected to add to compound")
            reward = None
            score = None
        elif (source_atom, destination_atom) not in compound.get_bonds():
            logging.debug("Selected bond is already in compound")
            reward = np.Infinity
            score = np.Infinity
        else:
            compound = self.add_bond(compound, source_atom, destination_atom)
            rollout_compound = self.rollout(compound)
            reward = self.calculate_reward(rollout_compound, rollout=True)
            score = self.calculate_reward(compound, rollout=False)
        done = self._is_done(compound, reward)

        info = {}
        return compound, reward, done, info, score

    def add_bond(
        self,
        compound: Compound,
        source_atom: int,
        destination_atom: int,
        select_type: str = "best",
        is_rollout: bool = False,
    ):
        """
        Adds a bond to a given compound.
        :param compound: Compound to add bond to
        :param source_atom: index of source atom
        :param destination_atom: index of destination atom
        :param select_type: "best" or "random" selection method to use for the bond type
        :param is_rollout: indicate inside the rollout function to optimize computation
        :return compound: updated Compound
        """
        molecule = compound.get_molecule()
        target_bond_index = molecule.AddBond(
            int(source_atom), int(destination_atom), BondType.UNSPECIFIED
        )
        target_bond = molecule.GetBondWithIdx(target_bond_index - 1)

        available_bonds = compound.filter_bond_type(
            int(source_atom), int(destination_atom), self.AVAILABLE_BONDS
        )

        # if member of aromatic queue was selected
        if len(compound.get_aromatic_queue()) > 0 and not is_rollout:
            # set conjugate and handle starting condition for queue duplicates, one start by single the other by double
            if (
                compound.get_last_bondtype() == Chem.rdchem.BondType.SINGLE
            ):  # or compound.aromatic_bonds_counter%2 == 1:
                bond_type = Chem.rdchem.BondType.DOUBLE
            else:
                bond_type = Chem.rdchem.BondType.SINGLE

            if bond_type not in available_bonds:
                bond_type = Chem.rdchem.BondType.SINGLE

            # remove chosen element from aromatic queue
            compound.aromatic_queue.pop(0)
            # Useless because of selection effect
            # compound.aromatic_bonds_counter+=1

        elif select_type == "best":
            per_bond_rewards = {}
            for bond_type in available_bonds:
                target_bond.SetBondType(bond_type)
                per_bond_rewards[bond_type] = self.calculate_reward(compound)
            bond_type = min(per_bond_rewards, key=per_bond_rewards.get)

        elif select_type == "random":
            bond_type = available_bonds[np.random.choice(available_bonds)]

        else:
            raise ValueError(f"select_type must be best or random given: {select_type}")

        target_bond.SetBondType(bond_type)

        if not is_rollout:
            compound.add_bond_history(
                [source_atom, destination_atom], target_bond.GetBondType()
            )

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
        while compound.get_mass() < self.max_mass:
            if len(compound.neighboring_bonds) > 0:
                id_bond = np.random.choice(range(len(compound.neighboring_bonds)))
                source_atom, destionation_atom = compound.neighboring_bonds[id_bond]
                compound = self.add_bond(
                    compound, source_atom, destionation_atom, "best", is_rollout=True
                )
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
        atoms = set(
            [CompoundBuilder.ATOM_SYMBOL_MAPPING[a.GetAtomMapNum()] for a in atoms]
        )
        smiles = Chem.MolToSmiles(compound.clean(preserve=True))
        actions = self.get_valid_actions(smiles, atoms, True, True, set([4, 6]), True)
        return actions

    def calculate_reward(self, compound: Compound, rollout=True):
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

            # encourage aromaticity
            if rollout == False:
                if compound.is_aromatic():
                    reward /= 2

            if np.isnan(reward):
                raise ValueError("NaN reward encountered: {}".format(smiles))

            return reward

        except (ValueError, RuntimeError, AttributeError) as e:
            logging.debug(
                "[INVALID REWARD]: {} - {}".format(compound.get_smiles(), str(e))
            )
            return np.Infinity


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
        self.action_mapper = {
            k: (source_atom, destination_atom)
            for k, (source_atom, destination_atom) in enumerate(
                self.init_compound.initial_bonds
            )
        }
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
