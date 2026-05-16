import numpy as np
d1 = np.load("data_vClean.npz")
d2 = np.load("data_vRecover.npz")
states = np.concatenate([d1["states"], d2["states"]])
actions = np.concatenate([d1["actions"], d2["actions"]])
np.savez("data_vFinal.npz", states=states, actions=actions)