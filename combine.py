"""Combine multiple .npz data files into one.
Edit the `FILES` list each iteration, then run:
    python combine.py
"""
import numpy as np

# =========================================================================
# EDIT THIS EACH ITERATION — list all files to combine
# =========================================================================
FILES = [
    "data_iter01a.npz",
    "data_iter02a.npz",
    "data_iter02b.npz",
    "data_iter03.npz",
]
OUTPUT = "data_combined.npz"
# =========================================================================

all_states = []
all_actions = []

for f in FILES:
    d = np.load(f)
    all_states.append(d["states"])
    all_actions.append(d["actions"])
    print(f"  {f}: {d['states'].shape[0]} samples")

states = np.concatenate(all_states)
actions = np.concatenate(all_actions)
np.savez(OUTPUT, states=states, actions=actions)
print(f"\nSaved {OUTPUT} — total samples: {len(states)}")