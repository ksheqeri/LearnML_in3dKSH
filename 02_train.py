"""Step 2 — Train the network.

Quick start (smoothing only — do this first):
    python 02_train.py --data data_combined.npz --tag iter12 --epochs 500 --smooth 3.0

Full command (all improvements):
    python 02_train.py --data data_combined.npz --tag iter13 --epochs 500 ^
        --smooth 3.0 --fix-labels --nav-inject 8000 --balance

Improvements over the original baseline
─────────────────────────────────────────
1. Steering-only output  (Y = actions[:, 1:2])
   The network predicts steering only. Throttle was nearly always 1.0 in
   the recorded data so learning it added noise without value.

2. --smooth SIGMA  (recommended: 3.0)
   Gaussian low-pass filter along the time axis of the steering targets.
   WASD produces discrete {-1, 0, +1} jumps between frames. The network
   sees contradictory targets at every transition and learns to average
   toward zero. Smoothing converts jumps into ramps so it learns
   proportional control. Largest single improvement: val loss ~40% lower.

3. --fix-labels
   Some recorded frames have a large heading error but steering = 0
   (key not pressed at the right moment). Those samples teach the network
   to ignore heading error. They are replaced with the proportional
   correction: steering = -heading_error / pi * 0.8

4. --nav-inject N  (recommended: 8000)
   Injects N synthetic samples with analytically correct steering.
   Each is a copy of a real open-road sensor state with its steering
   replaced by: steering = -clip(heading_error / pi) * gain
   Gives the network a clean, unambiguous heading-correction signal
   alongside the recorded driving data.

5. --balance
   Long straights dominate WASD recordings — steering near zero for
   extended periods. That biases the network to go straight even when
   it should turn. Keeping only 35% of near-zero samples balances the
   steering distribution.

6. Gradient clipping at ±1.0 (prevents exploding gradients)

7. Multi-seed training (--n-seeds, default 5)
   Trains with N different random initialisations and keeps the best
   validation loss. 5 seeds reliably finds a better minimum than any
   single run with minimal extra effort.
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# my_backward — DO NOT CHANGE
# Verified by gradient check. Works for any output dimension.
# =========================================================================
def my_backward(x, y_target, w, cache):
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


# =========================================================================
# Gradient check — float64 for numerical precision
# =========================================================================
def gradient_check():
    rng = np.random.default_rng(0)
    w   = {k: v.astype(np.float64) for k, v in nn_mod.init_weights(seed=0).items()}
    x   = rng.normal(size=(8, N_FEATURES)).astype(np.float64)
    y   = rng.uniform(-1, 1, size=(8, N_ACTIONS)).astype(np.float64)
    cache = nn_mod.forward_all(x, w)
    grads = my_backward(x, y, w, cache)

    print("\ngradient check:")
    for key in w:
        max_err = 0.0
        flat    = w[key].size
        for _ in range(5):
            idx = np.unravel_index(rng.integers(0, flat), w[key].shape)
            num = nn_mod.numerical_gradient(x, y, w, key, idx)
            ana = grads[key][idx]
            max_err = max(max_err, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
        flag = "OK" if max_err < 1e-4 else "BUG"
        print(f"  {key}: {max_err:.2e}  {flag}")
        assert max_err < 1e-4, f"gradient bug in {key}"


# =========================================================================
# Data quality helpers
# =========================================================================

def smooth_actions(Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur along the time axis of steering targets.

    Converts hard {-1, 0, +1} WASD jumps into smooth ramps.
    sigma=3 ≈ 0.15 s at 20 Hz.
    """
    smoothed = gaussian_filter1d(Y.astype(np.float64), sigma=sigma, axis=0)
    return np.clip(smoothed, -1.0, 1.0).astype(np.float32)


def fix_contradictory_labels(states_raw: np.ndarray,
                              Y: np.ndarray) -> np.ndarray:
    """Replace steering≈0 where heading_error is large."""
    Y            = Y.copy()
    heading_norm = np.clip(states_raw[:, 1] / np.pi, -1.0, 1.0)
    bad          = (np.abs(states_raw[:, 1]) > 0.3) & (np.abs(Y[:, 0]) < 0.1)
    Y[bad, 0]    = np.clip(-heading_norm[bad] * 0.8, -1.0, 1.0).astype(np.float32)
    print(f"  fix_labels: replaced {bad.sum():,} samples ({bad.mean()*100:.1f}%)")
    return Y


def inject_nav_data(states_raw: np.ndarray, Y: np.ndarray,
                    n: int, gain: float = 0.8) -> tuple:
    """Inject n synthetic proportional-navigation samples.

    Copies real sensor states from open sections of track and replaces
    their steering with: -clip(heading_error / pi) * gain
    """
    rng      = np.random.default_rng(1)
    open_idx = np.where(states_raw[:, 3] > 10.0)[0]
    idx      = rng.choice(open_idx, size=n, replace=len(open_idx) < n)
    syn_s    = states_raw[idx].copy()
    syn_Y    = np.clip(-np.clip(syn_s[:, 1] / np.pi, -1.0, 1.0) * gain,
                       -1.0, 1.0).astype(np.float32).reshape(-1, 1)
    print(f"  nav_inject: +{n:,} synthetic samples (gain={gain})")
    return (np.concatenate([states_raw, syn_s], axis=0),
            np.concatenate([Y,          syn_Y], axis=0))


def balance_steering(X: np.ndarray, Y: np.ndarray,
                     keep_frac: float = 0.35) -> tuple:
    """Keep only keep_frac of near-zero steering samples."""
    rng      = np.random.default_rng(7)
    straight = np.abs(Y[:, 0]) < 0.1
    keep_idx = rng.choice(np.where(straight)[0],
                          size=int(straight.sum() * keep_frac), replace=False)
    idx      = np.sort(np.concatenate([np.where(~straight)[0], keep_idx]))
    print(f"  balance: {len(Y):,} → {len(idx):,} "
          f"(kept {keep_frac*100:.0f}% of straights)")
    return X[idx], Y[idx]


# =========================================================================
# Dataset inspection
# =========================================================================
def inspect_dataset(states_raw, actions, tag):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>22s}: [{col.min():+7.2f}, {col.max():+7.2f}]  "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")
    viz.plot_action_histograms(actions, out=f"fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{tag}.png")


# =========================================================================
# Training loop
# =========================================================================
def _train_one_seed(X, Y, epochs, lr, batch_size, val_frac, seed):
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    Xva, Yva = X[perm[:n_val]],  Y[perm[:n_val]]
    Xtr, Ytr = X[perm[n_val:]], Y[perm[n_val:]]

    w     = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)
    tr_losses, va_losses = [], []
    best_val, best_w = float("inf"), {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            xb, yb  = Xtr[idx[i:i+batch_size]], Ytr[idx[i:i+batch_size]]
            cache   = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb);  n_b += 1
            grads   = my_backward(xb, yb, w, cache)
            for k in grads:
                grads[k] = np.clip(grads[k], -1.0, 1.0)  # gradient clipping
            nn_mod.adam_step(w, grads, state, lr=lr)

        v = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        tr_losses.append(ep_loss / max(1, n_b));  va_losses.append(v)
        if v < best_val:
            best_val = v;  best_w = {k: w[k].copy() for k in w}
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:4d}  train={tr_losses[-1]:.4f}  "
                  f"val={v:.4f}  best={best_val:.4f}")

    return best_w, tr_losses, va_losses, best_val


def train(X, Y, epochs=500, lr=1e-3, batch_size=64, val_frac=0.1, n_seeds=5):
    """Try n_seeds random initialisations, return weights with lowest val loss."""
    best_val, best_w, best_tr, best_va = float("inf"), None, [], []
    for seed in range(n_seeds):
        print(f"\n── seed {seed} / {n_seeds-1} ──")
        w, tr, va, bv = _train_one_seed(X, Y, epochs, lr, batch_size, val_frac, seed)
        print(f"  best val: {bv:.4f}")
        if bv < best_val:
            best_val, best_w, best_tr, best_va = bv, w, tr, va
    print(f"\n✓ best overall val: {best_val:.4f}")
    return best_w, best_tr, best_va


# =========================================================================
# Entry point
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       default="data_combined.npz")
    ap.add_argument("--tag",        default="iter12")
    ap.add_argument("--epochs",     type=int,   default=500)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch",      type=int,   default=64)
    ap.add_argument("--smooth",     type=float, default=0.0,
                    help="Gaussian sigma for action smoothing. 0=off. Use 3.0.")
    ap.add_argument("--fix-labels", action="store_true",
                    help="Replace steering=0 where heading_error is large.")
    ap.add_argument("--nav-inject", type=int,   default=0,
                    help="Synthetic nav samples to inject. Use 8000.")
    ap.add_argument("--nav-gain",   type=float, default=0.8)
    ap.add_argument("--balance",    action="store_true",
                    help="Undersample straight-driving to reduce forward bias.")
    ap.add_argument("--n-seeds",    type=int,   default=5,
                    help="Random seeds to try. Best val wins.")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions_raw = d["states"], d["actions"]
    print(f"Loaded: {states_raw.shape[0]:,} samples from {args.data}")
    inspect_dataset(states_raw, actions_raw, tag=args.tag)

    # Steering-only target — column 1 of actions
    Y = actions_raw[:, 1:2].astype(np.float32)
    print(f"\nY (steering): [{Y.min():+.3f}, {Y.max():+.3f}]  std={Y.std():.3f}")

    if args.fix_labels:
        Y = fix_contradictory_labels(states_raw, Y)

    if args.smooth > 0.0:
        before = Y.std()
        Y      = smooth_actions(Y, sigma=args.smooth)
        print(f"  smooth sigma={args.smooth}: std {before:.3f} → {Y.std():.3f}")

    X = normalize_states(states_raw)

    if args.nav_inject > 0:
        states_raw, Y = inject_nav_data(states_raw, Y,
                                        n=args.nav_inject, gain=args.nav_gain)
        X = normalize_states(states_raw)

    if args.balance:
        X, Y = balance_steering(X, Y)

    print(f"\nFinal: {X.shape[0]:,} samples  X{X.shape}  Y{Y.shape}")

    gradient_check()

    weights, tr_losses, va_losses = train(
        X, Y, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch, n_seeds=args.n_seeds)

    out = f"nav_{args.tag}.npz"
    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")
    nn_mod.save(weights, out)
    print(f"\nSaved {out}  (best val = {min(va_losses):.4f})")


if __name__ == "__main__":
    main()
