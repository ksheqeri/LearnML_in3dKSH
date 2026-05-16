import numpy as np
d1 = np.load("data_vFreshCombined.npz")
d2 = np.load("data_vFresh4.npz")
states = np.concatenate([d1["states"], d2["states"]])
actions = np.concatenate([d1["actions"], d2["actions"]])
np.savez("data_vFresh4Combined.npz", states=states, actions=actions)