import gym
import numpy as np
import logging

from gym import spaces
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from typing import Union, Tuple

from lib.data_structures import Compound
from lib.calculators import AbstractCalculator
from lib.helpers import Sketcher

class MoleculeEnv(gym.Env):
    AVAILABLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

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
            logging.debug("No bonds selected to add to compound")
            reward = None
        elif (source_atom, destination_atom) not in compound.get_bonds():
            logging.debug("Selected bond is already in compound")
            reward = np.Infinity
        else:
            compound = self.add_bond(compound, source_atom, destination_atom)
            reward = self.calculate_reward(compound)

        done = self._is_done(compound, reward)

        info = {}
        return compound, reward, done, info
    

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
          return np.Infinity


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
