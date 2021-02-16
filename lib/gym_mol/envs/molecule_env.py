import gym
import numpy as np
import logging

from gym import spaces
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from lib.data_structures import Compound

class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    AVAILABLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    def initialize(self, calculator):
        self.calculator = calculator

    def set_compound(self, compound):
        self.init_compound = compound.clone()

    def step(self, compound, action):
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

        return compound, reward, done
    

    def add_bond(self, compound, source_atom, destination_atom):
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


    def _is_done(self, compound, reward):
        return False

    def _reset_action_space(self):
        n_bonds = len(self.init_compound.initial_bonds)
        self.action_space = spaces.Discrete(n_bonds)
        self.action_mapper = {k: (source_atom, destination_atom) for k, (source_atom, destination_atom) in enumerate(self.init_compound.initial_bonds)}
        self.action_mapper[-1] = (None, None)
        self.n_actions = n_bonds

    def reset(self):
        self._reset_action_space()

    def render(self, mode='human'):
        pass

    def close(self):
        pass