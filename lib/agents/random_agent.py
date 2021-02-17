import numpy as np
from gym import spaces
from lib.data_structures import Compound

class RandomAgent:
    """
    Basic random agent.
    """

    def reset(self, compound: Compound):
        """
        Sets the action space from a given compound.
        :param compound: Compound to get number of potential bonds from
        :return None:
        """
        n_bonds = compound.bonds_count()
        self.action_space = spaces.MultiDiscrete([n_bonds] + [2 for _ in range(n_bonds)])

    def act(self, observation: np.array, reward: float, info: dict, done: bool):
        """
        Randomly samples an action from action space
        :param observation: np.array encoding the activated bonds for the compound
        :param reward: reward obtained at last iteration
        :param info: dictionnary containing additional information, essentialy the updated compound
        :param done: boolean indicating whether episode is over or not
        :return action: np.array encoding the selected node and bond to add
        """
        return self.action_space.sample()

    def get_output(self, compound: Compound):
        """
        Returns output in json format
        :param: compound: compound to output
        :return list(dict): output in json format
        """
        return [{"smiles": compound.clean_smiles()}]