import gym
import numpy as np
import logging

from gym import spaces
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from typing import Union, Tuple

from lib.data_structures import Compound, Tree
from lib.calculators import AbstractCalculator
from lib.helpers import Sketcher

class MoleculeEnv(gym.Env):
    AVAILABLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
    INFINITY = Tree.INFINITY

    def initialize(self, calculator: AbstractCalculator, render_location: str=None):
        """
        Own initialization function since gym.make doesn't support passing arguments.
        :param calculator: used for reward computation
        :param render_location: path to save rendered molecules to
        :return: None
        """
        self.calculator = calculator
        self.sketcher = Sketcher()
        if render_location is not None:
            self.sketcher.set_location(render_location)


    def set_compound(self, compound: Compound):
        """
        Sets the initial compound. The initial compounds implicitely defines the action space.
        :param compound: Compound
        :return: None
        """
        self.init_compound = compound.clone()

    def step(self, action):
        """
        Performs a step by adding a bond to a given compound and computing the associated reward.
        :param action: np.array from action space
        :return observation: observed state after performing action
        :return reward: reward for selected action
        :return done: whether episode is completed
        :return info: contains additional information
        """
        bond = action[0]
        state = action[1:]
        compound = self._state_to_compound(state)
        source_atom, destination_atom = self.action_mapper[bond]
        if source_atom is None or destination_atom is None:
            logging.debug("No bonds selected to add to compound")
            reward = None
        elif (source_atom, destination_atom) not in compound.get_bonds():
            logging.debug("Selected bond is already in compound")
            reward = -self.INFINITY
        else:
            compound = self.add_bond(compound, source_atom, destination_atom)
            reward = -self.calculate_reward(compound)

        done = self._is_done(compound, reward)
        observation = self._compound_to_state(compound)
        if self._hash_state(observation) not in self.cache:
            self.cache[self._hash_state(observation)] = compound

        info = {"compound": compound}
        return observation, reward, done, info
    

    def add_bond(self, compound: Compound, source_atom: int, destination_atom: int):
        """
        Adds a bond to a given compound.
        :param compound: Compound to add bond to
        :param source_atom: index of source atom
        :param destination atom: index of destination atom
        :return compound: updated Compound
        """
        molecule = compound.get_molecule()
        target_bond_index = molecule.AddBond(int(source_atom), int(destination_atom), BondType.UNSPECIFIED)
        target_bond = molecule.GetBondWithIdx(target_bond_index - 1)

        per_bond_rewards = {}
        for bond_type in self.AVAILABLE_BONDS:
            target_bond.SetBondType(bond_type)
            per_bond_rewards[bond_type] = self.calculate_reward(compound)

        target_bond.SetBondType(min(per_bond_rewards, key=per_bond_rewards.get))

        compound.remove_bond((source_atom, destination_atom))
        compound.flush_bonds()
        return compound

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

          logging.debug("{} : {} : {:.6f}".format(compound.get_smiles(), Chem.MolToSmiles(molecule), reward))
          return reward

        except (ValueError, RuntimeError, AttributeError) as e:
          logging.debug("[INVALID REWARD]: {} - {}".format(compound.get_smiles(), str(e)))
          return self.INFINITY


    def _is_done(self, compound: Compound, reward: float):
        """
        Returns whether episode is over.
        :param compound: current processed compound
        :param reward: current obtained reward
        :return bool:
        """
        return False

    def _reset_action_space(self):
        """
        Sets action space based on initial compound.
        Action space is a discrete space of length the number of potential bonds of the initial compound.
        :param None:
        :return None:
        """
        n_bonds = self.init_compound.bonds_count()
        self.action_space = spaces.MultiDiscrete([n_bonds] + [2 for _ in range(n_bonds)])
        self.action_mapper = self.init_compound.get_index_to_bond_mapper()
        self.action_mapper[-1] = (None, None)
        self.action_reverse_mapper = {v: k for k,v in self.action_mapper.items()}
        self.n_actions = n_bonds

    def _reset_observation_space(self):
        """
        Resets observation space based on possible bonds of initial compound.
        """
        self.observation_space = spaces.MultiBinary(self.init_compound.bonds_count())

    def _compound_to_state(self, compound: Compound):
        """
        Converts compound to array encoding activated bonds.
        :param compound: compound to encode
        :return state: binary np.array where 1 indicates presence of bond
        """
        state = np.ones(self.n_actions, dtype=np.int8)
        for bond in compound.get_bonds():
            state[self.action_reverse_mapper[bond]] = 0
        return state

    def _hash_state(self, state: np.array):
        """
        Converts state np.array into single int by considering the array as a binary encoding of a number.
        :param state: binary np.array
        :return int:
        """
        return sum([x*2**k for k,x in enumerate(state)])

    def _state_to_compound(self, state: np.array):
        """
        Converts state np.array to a compound by adding bonds one by one from initial compound.
        Not guaranteed to give accurate results since the order of bonds is not taken into account
        :param state: binary np.array
        :return compound:
        """
        n = self._hash_state(state)
        if n in self.cache:
            return self.cache[n].clone()
        else:
            compound = self.init_compound.clone()
            for i, bond in enumerate(state):
                if bond:
                    source_atom, destination_atom = self.action_mapper[i]
                    compound = self.add_bond(compound, source_atom, destination_atom)
            self.cache[n] = compound
            return compound.clone()

    def reset(self):
        """
        Resets the environment
        :param None:
        :return None:
        """
        self._reset_action_space()
        self._reset_observation_space()
        init_observation = np.zeros(self.n_actions, dtype=np.int8)
        self.cache = {self._hash_state(init_observation): self.init_compound.clone()}
        return init_observation

    def render(self, smiles: str):
        """
        Renders a molecule by drawing it.
        :param smiles: smiles string representation of the molecule
        :return None:
        """
        self.sketcher.draw(smiles)

    def close(self):
        """
        Closes the environment.
        """
        pass