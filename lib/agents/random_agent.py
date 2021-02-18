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
		self.action_space = spaces.Discrete(len(compound.get_initial_bonds()))

	def act(self, compound: Compound, reward: float):
		"""
		Randomly samples an action from action space
		:param compound: Compound used at previous iteration
		:param reward: reward obtained at previous iteration
		:return compound: Compound to act on
		:return int: action index
		"""
		return compound, self.action_space.sample()

	def get_output(self, compound: Compound):
		"""
		Returns output in json format
		:param: compound: compound to output
		:return list(dict): output in json format
		"""
		return [{"smiles": compound.clean_smiles()}]
