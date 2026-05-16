import numpy as np
d1 = np.load("data_vFresh.npz")
d2 = np.load("data_vFresh2.npz")
d3 = np.load("data_vFresh3.npz")
states = np.concatenate([d1["states"], d2["states"], d3["states"]])
actions = np.concatenate([d1["actions"], d2["actions"], d3["actions"]])
np.savez("data_vFreshCombined.npz", states=states, actions=actions)