import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from lib.data_structures import Compound


class PPO2Agent:

    def __init__(self, env: gym.Env, learn_steps: int=50000, **kwargs):
        """
        Proximal Policy Optimization algorithm.
        Paper: https://arxiv.org/abs/1707.06347
        :param: env: Gym environment to operate on.
        :param learn_steps: Number of steps for learning phase on the environement.
        :param kwargs: Optional arguments that will be passed to stable_baselines PPO2 class.
        """
        self.env = env
        self.kwargs = kwargs
        self.learn_steps = learn_steps

    def reset(self, *args, **kwargs):
        """
        Instantiate PPO2 from stable_baselines PPO2 class.
        :param: args: Has to be able to accept arguments to behave as other agents.
        """
        self.ppo2 = PPO2(MlpPolicy, self.env, verbose=1, **self.kwargs)
        self.ppo2.learn(total_timesteps=self.learn_steps)

    def act(self, observation, reward, info, done):
        """
        Performs an inference step to select action to perform
        :param observation: np.array encoding the activated bonds for the compound
        :param reward: reward obtained at last iteration
        :param info: dictionnary containing additional information, essentialy the updated compound
        :param done: boolean indicating whether episode is over or not
        :return action: np.array encoding the selected node and bond to add
        """
        return self.ppo2.predict(observation)[0]

    def get_output(self, compound: Compound, reward: float):
        """
        Returns output in json format
        :param: compound: compound to output
        :return list(dict): output in json format
        """
        return [{"smiles": compound.clean_smiles(), "reward": reward}]

