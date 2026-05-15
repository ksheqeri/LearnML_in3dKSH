import numpy as np
d1 = np.load("data_v2.npz")
d2 = np.load("data_v3.npz")
states = np.concatenate([d1["states"], d2["states"]])
actions = np.concatenate([d1["actions"], d2["actions"]])
np.savez("data_v3combined.npz", states=states, actions=actions)
