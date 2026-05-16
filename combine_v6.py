import numpy as np
d1 = np.load("data_v5combined.npz")
d2 = np.load("data_v6.npz")
states = np.concatenate([d1["states"], d2["states"]])
actions = np.concatenate([d1["actions"], d2["actions"]])
np.savez("data_v6combined.npz", states=states, actions=actions)