from gym.envs.registration import register

register(
    id="molecule-v0",
    entry_point="lib.gym_mol.envs:MoleculeEnv",
)
