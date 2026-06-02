"""MLP forward pass, backprop, and Adam — all in NumPy.

This module is the heart of the project. You should be able to read every
line in this file and explain what it does.

Architecture (iter12+):  12 → 64 → 32 → 1
  - 12 normalised sensor inputs
  - Hidden 1 : 64 neurons, ReLU
  - Hidden 2 : 32 neurons, ReLU
  - Output   :  1 neuron (steering only), tanh → [-1, 1]

Why steering only?
  Early iterations predicted both throttle and steering (N_OUT=2).
  Analysis of the training data showed throttle was almost always 1.0
  (WASD key held down the whole time), so the network was wasting half
  its capacity learning a trivial constant. Switching to steering-only
  lets the full network focus on the hard problem.
  Throttle is fixed at a constant in the policy (smooth_mlp.py).
"""
from __future__ import annotations
import numpy as np

H1, H2      = 64, 32   # hidden layer widths — best for ~40 k samples
N_IN, N_OUT = 12,  1   # 12 sensor inputs → 1 steering output


# ── Forward pass ──────────────────────────────────────────────────────────

def forward(x: np.ndarray, w: dict) -> np.ndarray:
    """Run one forward pass.

    Args:
        x : shape (N, 12) for a batch, or (12,) for a single step.
        w : weight dict with keys W1, b1, W2, b2, W3, b3.

    Returns:
        Steering values in [-1, 1], shape (N, 1) or (1,).
        Old 2-output weights still load and run — shape becomes (N, 2).
    """
    single = (x.ndim == 1)
    if single:
        x = x[None, :]
    z1 = x  @ w["W1"] + w["b1"];  a1 = np.maximum(0, z1)   # ReLU
    z2 = a1 @ w["W2"] + w["b2"];  a2 = np.maximum(0, z2)   # ReLU
    z3 = a2 @ w["W3"] + w["b3"];  y  = np.tanh(z3)          # tanh → [-1,1]
    return y[0] if single else y


def forward_all(x: np.ndarray, w: dict) -> dict:
    """Forward pass returning every intermediate value needed for backprop."""
    z1 = x  @ w["W1"] + w["b1"];  a1 = np.maximum(0, z1)
    z2 = a1 @ w["W2"] + w["b2"];  a2 = np.maximum(0, z2)
    z3 = a2 @ w["W3"] + w["b3"];  y  = np.tanh(z3)
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "y": y}


# ── Loss ──────────────────────────────────────────────────────────────────

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())


# ── Backward pass ─────────────────────────────────────────────────────────

def backward(x, y_target, w, cache):
    n   = x.shape[0]
    y   = cache["y"]
    dy  = 2.0 * (y - y_target) / (n * y.shape[1])
    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3;  db3 = dz3.sum(axis=0)
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2;  db2 = dz2.sum(axis=0)
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1;            db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


# ── Adam optimiser ────────────────────────────────────────────────────────

def init_adam(w: dict) -> dict:
    return {
        "m": {k: np.zeros_like(v) for k, v in w.items()},
        "v": {k: np.zeros_like(v) for k, v in w.items()},
        "t": 0, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
    }


def adam_step(w: dict, grads: dict, state: dict, lr: float = 1e-3):
    state["t"] += 1
    t  = state["t"]
    b1, b2, eps = state["beta1"], state["beta2"], state["eps"]
    for k in w:
        g = grads[k]
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * g * g
        mh = state["m"][k] / (1 - b1 ** t)
        vh = state["v"][k] / (1 - b2 ** t)
        w[k] -= lr * mh / (np.sqrt(vh) + eps)


# ── Weight initialisation ─────────────────────────────────────────────────

def init_weights(seed: int = 0) -> dict:
    """He-init for ReLU layers, Xavier for the tanh output layer."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / N_IN), (N_IN, H1)).astype(np.float32),
        "b1": np.zeros(H1,    dtype=np.float32),
        "W2": rng.normal(0, np.sqrt(2 / H1),  (H1,  H2)).astype(np.float32),
        "b2": np.zeros(H2,    dtype=np.float32),
        "W3": rng.normal(0, np.sqrt(1 / H2),  (H2, N_OUT)).astype(np.float32),
        "b3": np.zeros(N_OUT, dtype=np.float32),
    }


# ── Save / load ───────────────────────────────────────────────────────────

def save(weights: dict, path: str):
    np.savez(path, **weights)


def load(path: str) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


# ── Numerical gradient (gradient check helper) ────────────────────────────

def numerical_gradient(x, y_target, w, key, idx, h=1e-4):
    w[key][idx] += h;  lp = mse_loss(forward(x, w), y_target)
    w[key][idx] -= 2*h; lm = mse_loss(forward(x, w), y_target)
    w[key][idx] += h
    return (lp - lm) / (2 * h)
