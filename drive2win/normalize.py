"""Input normalisation for the navigation network.

12 input features in order:
  speed, heading_error, checkpoint_distance,
  ray_0_front, ray_1_+45, ray_2_+90, ray_3_+135,
  ray_4_back,  ray_5_-135, ray_6_-90, ray_7_-45,
  ground_friction

Output:
  1 value — steering only.
  Throttle is a fixed constant in the policy, not predicted by the network.

Two helper functions are required by benchmark.py at inference time:
  sensors_to_input() — converts a live sensors dict to the normalised
                       12-vector the network expects.
  clip_action()      — converts network output to (throttle, steering),
                       handling both old 2-output and new 1-output weights.
"""
from __future__ import annotations
import numpy as np

# ── Normalisation constants ───────────────────────────────────────────────
SPD_MAX  = 20.0
DIST_MAX = 100.0
RAY_MAX  = 50.0
FRIC_MAX = 1.5

# ── Metadata ──────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "speed", "heading_error", "checkpoint_distance",
    "ray_0_front", "ray_1_+45",  "ray_2_+90",  "ray_3_+135",
    "ray_4_back",  "ray_5_-135", "ray_6_-90",  "ray_7_-45",
    "ground_friction",
]
ACTION_NAMES = ["steering"]
N_FEATURES   = 12
N_ACTIONS    = 1


# ── Batch normalisation (used during training) ────────────────────────────

def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Normalise a raw (N, 12) state array to float32 in roughly [-1, 1].

      col 0   speed            ÷ SPD_MAX   clipped to [-1,  1]
      col 1   heading_error    ÷ π         clipped to [-1,  1]
      col 2   checkpoint_dist  ÷ DIST_MAX  clipped to [ 0,  1]
      col 3-10  8 rays         ÷ RAY_MAX   clipped to [ 0,  1]
      col 11  ground_friction  ÷ FRIC_MAX  clipped to [ 0,  1]
    """
    s = np.asarray(states_raw, dtype=np.float32).copy()
    s[:, 0]    = np.clip(s[:, 0]    / SPD_MAX,   -1.0, 1.0)
    s[:, 1]    = np.clip(s[:, 1]    / np.pi,     -1.0, 1.0)
    s[:, 2]    = np.clip(s[:, 2]    / DIST_MAX,   0.0, 1.0)
    s[:, 3:11] = np.clip(s[:, 3:11] / RAY_MAX,    0.0, 1.0)
    s[:, 11]   = np.clip(s[:, 11]   / FRIC_MAX,   0.0, 1.0)
    return s


# ── Live-inference helpers (imported by benchmark.py) ─────────────────────

def sensors_to_input(sensors: dict) -> np.ndarray:
    """Convert a sensors dict to a normalised (12,) float32 vector.

    Accepts the standard state["sensors"] dict from the WebSocket.
    Pads rays to 8 values if the server sends fewer.
    """
    rays = list(sensors.get("rays", [50.0] * 8))
    while len(rays) < 8:
        rays.append(50.0)

    raw = np.array([
        sensors.get("speed",                0.0),
        sensors.get("heading_error",        0.0),
        sensors.get("checkpoint_distance", 50.0),
        float(rays[0]), float(rays[1]), float(rays[2]), float(rays[3]),
        float(rays[4]), float(rays[5]), float(rays[6]), float(rays[7]),
        sensors.get("ground_friction",      1.0),
    ], dtype=np.float32)

    out = raw.copy()
    out[0]    = np.clip(raw[0]    / SPD_MAX,   -1.0, 1.0)
    out[1]    = np.clip(raw[1]    / np.pi,     -1.0, 1.0)
    out[2]    = np.clip(raw[2]    / DIST_MAX,   0.0, 1.0)
    out[3:11] = np.clip(raw[3:11] / RAY_MAX,    0.0, 1.0)
    out[11]   = np.clip(raw[11]   / FRIC_MAX,   0.0, 1.0)
    return out


def clip_action(a: np.ndarray,
                default_throttle: float = 0.65) -> tuple[float, float]:
    """Convert network output to (throttle, steering), both clipped to [-1, 1].

    Handles both weight formats:
      1-output weights (iter12+): network outputs steering only.
                                  Throttle is the fixed default (0.65).
      2-output weights (iter07b and earlier): uses both outputs directly.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if len(a) == 1:
        return default_throttle, float(np.clip(a[0], -1.0, 1.0))
    return float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
