"""Policy module — steering-only network with recovery and checkpoint guidance.

This module provides make_policy(), which plugs into the benchmark via
--module drive2win.smooth_mlp.

Design
────────────────
Throttle
  Fixed at 0.65 for every frame. A trained network that has learned
  "go forward and steer" naturally produces a stable throttle, and a
  constant value matches that expectation cleanly.

Steering
  Output comes from the neural network. A partial heading-correction
  blend is added as the car approaches each checkpoint, but the network
  always contributes — it is never fully removed from the output.

  steer_final = steer_nn * (1 - blend) + steer_homing * blend

Recovery
  Three independent detectors trigger a brief reverse:
    1. Wedge detector    — physically stuck against a wall
    2. No-progress       — checkpoint distance unchanged for ~3 seconds
    3. Orbit escape      — circling a checkpoint without threading it

  After reversing, the network takes back over immediately.

Obstacle avoidance
  Only active when an isolated obstacle blocks the front ray while a 45°
  ray shows a clear gap. Blends up to 40% gap-steering into the output.
  Does not fire on normal track walls (no gap found → condition false).

Usage:
    python 03_benchmark.py --tag iter12 --weights nav_iter12.npz \\
        --module drive2win.smooth_mlp --seeds 42
"""
from __future__ import annotations
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input

# ── Tunable constants ─────────────────────────────────────────────────────

THROTTLE          = 0.65   # fixed forward throttle — same every frame
STEER_GAIN        = 1.6     # scale raw network steering output

# Partial homing — network always contributes at least (1 - MAX_BLEND)
MAX_BLEND         = 0.85   # cap: 85%
CP_HOMING_DIST    = 50.0   # metres — begin blending toward checkpoint
CP_HOMING_GAIN    = 7.0    # proportional gain on heading_error

# Wedge / stuck detection
STUCK_SPEED       = 0.30
PURE_STUCK_SPEED  = 0.15
RAY_WEDGE         = 4.0    # metres — wall-contact threshold
STUCK_FRAMES      = 15
PURE_STUCK_FRAMES = 50
REVERSE_FRAMES    = 10

# No-progress detector
# Fires when checkpoint_distance has not changed by PROGRESS_MIN metres
# over the last PROGRESS_WINDOW frames (~3 seconds at 20 Hz).
# Catches ramps, geometry traps, and spinning in place.
PROGRESS_WINDOW   = 35
PROGRESS_MIN      = 1.0

# Orbit escape
CP_ORBIT_DIST     = 8.0
CP_ORBIT_FRAMES   = 25

# Obstacle avoidance
# Only active when front is blocked AND a 45° ray shows a navigable gap.
OBS_DIST          = 8.0


# ── Policy factory ────────────────────────────────────────────────────────

def make_policy(weights_path: str):
    """Load weights and return a callable policy.

    policy(state) -> (throttle, steering)
    state must contain a "sensors" key, or be a flat dict with the same fields.
    """
    w          = nn.load(weights_path)
    output_dim = w["W3"].shape[1]

    if output_dim == 1:
        print(f"  smooth_mlp: steering-only weights ({weights_path})")
    else:
        print(f"  smooth_mlp: legacy 2-output weights ({weights_path})")

    stuck_count       = 0
    reverse_count     = 0
    orbit_frames      = 0
    progress_ref_dist = 999.0
    no_progress_count = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal stuck_count, reverse_count, orbit_frames, \
                 progress_ref_dist, no_progress_count

        sensors = state.get("sensors", state)
        speed   = float(sensors.get("speed", 1.0))
        rays    = sensors.get("rays", [50.0] * 8)
        front   = float(rays[0]) if rays else 50.0
        right   = float(rays[2]) if len(rays) > 2 else 50.0
        left    = float(rays[6]) if len(rays) > 6 else 50.0
        cp_dist = float(sensors.get("checkpoint_distance", 999.0))

        # ── 1. Wedge detection ─────────────────────────────────────────
        wedged     = (speed < STUCK_SPEED
                      and front < RAY_WEDGE
                      and (left < RAY_WEDGE or right < RAY_WEDGE))
        pure_stuck = speed < PURE_STUCK_SPEED
        stuck_count = stuck_count + 1 if (wedged or pure_stuck) else 0
        if stuck_count >= (STUCK_FRAMES if wedged else PURE_STUCK_FRAMES):
            reverse_count = REVERSE_FRAMES
            stuck_count   = 0

        # ── 2. No-progress detection ───────────────────────────────────
        if abs(cp_dist - progress_ref_dist) > PROGRESS_MIN:
            progress_ref_dist = cp_dist
            no_progress_count = 0
        else:
            no_progress_count += 1
        if no_progress_count >= PROGRESS_WINDOW and reverse_count == 0:
            reverse_count     = REVERSE_FRAMES * 2
            no_progress_count = 0
            progress_ref_dist = cp_dist

        # ── 3. Orbit escape ────────────────────────────────────────────
        orbit_frames = orbit_frames + 1 if cp_dist < CP_ORBIT_DIST else 0
        if orbit_frames >= CP_ORBIT_FRAMES and reverse_count == 0:
            reverse_count = REVERSE_FRAMES * 2
            orbit_frames  = 0

        # ── 4. Reverse phase ───────────────────────────────────────────
        if reverse_count > 0:
            reverse_count -= 1
            return (-1.0, -0.8 if right < left else 0.8)

        # ── 5. Network inference ───────────────────────────────────────
        x        = sensors_to_input(sensors)
        raw      = nn.forward(x, w)
        steer_nn = float(np.clip(
            (raw[0] if output_dim == 1 else raw[1]) * STEER_GAIN, -1.0, 1.0))

        # ── 6. Partial checkpoint homing ───────────────────────────────
        if cp_dist < CP_HOMING_DIST:
            heading_norm = float(np.clip(
                sensors.get("heading_error", 0.0) / np.pi, -1.0, 1.0))
            steer_homing = float(np.clip(
                -heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            t     = (CP_HOMING_DIST - cp_dist) / CP_HOMING_DIST
            blend = float(np.clip(t * MAX_BLEND, 0.0, MAX_BLEND))
            steer = steer_nn * (1.0 - blend) + steer_homing * blend
        else:
            steer = steer_nn

        # ── 7. Obstacle avoidance ──────────────────────────────────────
        front_left  = float(rays[7]) if len(rays) > 7 else 50.0
        front_right = float(rays[1]) if len(rays) > 1 else 50.0
        if front < OBS_DIST and max(front_left, front_right) > front:
            heading_err = float(sensors.get("heading_error", 0.0))
            candidates  = [
                (-np.pi / 4, front_left),
                (0.0,        front),
                ( np.pi / 4, front_right),
            ]
            best_score, best_angle = -1.0, 0.0
            for angle, dist in candidates:
                score = dist * max(0.1, float(np.cos(angle + heading_err)))
                if score > best_score:
                    best_score, best_angle = score, angle
            gap_steer = float(np.clip(best_angle / (np.pi / 2), -1.0, 1.0))
            t_obs     = float(np.clip(1.0 - front / OBS_DIST, 0.0, 0.4))
            steer     = steer * (1.0 - t_obs) + gap_steer * t_obs

        return (THROTTLE, steer)

    return policy
