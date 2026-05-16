import numpy as np
d1 = np.load("data_v6combined.npz")
d2 = np.load("data_v7.npz")
states = np.concatenate([d1["states"], d2["states"]])
actions = np.concatenate([d1["actions"], d2["actions"]])
np.savez("data_v7combined.npz", states=states, actions=actions)