#!/usr/bin/env python3

"""
Extend a motion CSV by repeating the joint-space gait pattern
while continuously advancing the base position and orientation.

Input format (per row):
  x, y, z, qx, qy, qz, qw, j1, j2, ..., jN

Usage:
  python3 scripts/extend_motion.py INPUT_CSV --duration SECONDS [--fps FPS] [--output OUTPUT_CSV]

Arguments:
  INPUT_CSV           Path to the input motion CSV.
  --duration, -t      Desired total duration (in seconds) of the output motion.
  --fps               Frames per second (rows per second), default: 30.
  --output, -o        Optional output CSV path. Defaults to INPUT_CSV with
                      '_extended' appended to the stem in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Sequence


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> List[float]:
    """Hamilton product for quaternions stored as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]


def quat_conj(q: Sequence[float]) -> List[float]:
    """Conjugate / inverse of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_normalize(q: Sequence[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    inv = 1.0 / n
    return [x * inv, y * inv, z * inv, w * inv]


def read_motion(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            rows.append([float(x) for x in raw])
    return rows


def write_motion(path: Path, rows: List[List[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([f"{v:.8f}" for v in r])


def extend_motion(rows: List[List[float]], target_frames: int) -> List[List[float]]:
    """
    Resize a motion sequence to a target number of frames.

    If target_frames <= len(rows), the sequence is truncated.
    If target_frames > len(rows), the sequence is extended by repeating
    the learned gait pattern while advancing the base pose.
    """
    if not rows:
        raise ValueError("Input motion is empty.")

    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")

    n = len(rows)
    d = len(rows[0])
    if d < 7:
        raise ValueError("Expected at least 7 columns (x,y,z,qx,qy,qz,qw, ...).")

    # Truncation-only case.
    if target_frames <= n:
        return rows[:target_frames]

    extra_steps = target_frames - n

    # Basic views into each row
    def pos_of(r: Sequence[float]) -> List[float]:
        return [r[0], r[1], r[2]]

    def quat_of(r: Sequence[float]) -> List[float]:
        return [r[3], r[4], r[5], r[6]]

    def joints_of(r: Sequence[float]) -> List[float]:
        return list(r[7:])

    # Precompute per-step base increments (position and orientation).
    pos_steps: List[List[float]] = []
    quat_steps: List[List[float]] = []
    for i in range(n - 1):
        p0 = pos_of(rows[i])
        p1 = pos_of(rows[i + 1])
        dp = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        pos_steps.append(dp)

        q0 = quat_normalize(quat_of(rows[i]))
        q1 = quat_normalize(quat_of(rows[i + 1]))
        # Relative rotation q_rel such that q1 = q0 * q_rel.
        q_rel = quat_mul(quat_conj(q0), q1)
        quat_steps.append(quat_normalize(q_rel))

    # Find a "phase" index whose joint pose is closest to the last row's joints.
    last_joints = joints_of(rows[-1])

    def joints_distance_sq(a: Sequence[float], b: Sequence[float]) -> float:
        return sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b))

    best_idx = 0
    best_dist = float("inf")
    for i in range(n):
        d2 = joints_distance_sq(joints_of(rows[i]), last_joints)
        if d2 < best_dist:
            best_dist = d2
            best_idx = i

    # Build extended samples.
    extended_rows: List[List[float]] = list(rows)

    pos_prev = pos_of(rows[-1])
    quat_prev = quat_normalize(quat_of(rows[-1]))

    def pattern_index(k: int) -> int:
        # Use the precomputed (n-1) increments, wrapping around from best_idx.
        return (best_idx + k) % (n - 1)

    for k in range(extra_steps):
        inc_idx = pattern_index(k)
        dp = pos_steps[inc_idx]
        q_rel = quat_steps[inc_idx]

        # Advance base position.
        pos_curr = [
            pos_prev[0] + dp[0],
            pos_prev[1] + dp[1],
            pos_prev[2] + dp[2],
        ]

        # Advance base orientation.
        quat_curr = quat_normalize(quat_mul(quat_prev, q_rel))

        # Follow the same joint-space gait pattern, starting just after best_idx.
        src_row = rows[(best_idx + 1 + k) % n]
        joints_curr = joints_of(src_row)

        new_row = pos_curr + quat_curr + joints_curr
        extended_rows.append(new_row)

        pos_prev = pos_curr
        quat_prev = quat_curr

    return extended_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend a motion CSV by repeating its gait pattern while "
            "maintaining continuous base pose."
        )
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        help="Path to the input motion CSV.",
        required=True,
    )
    parser.add_argument(
        "--duration",
        "-t",
        type=float,
        required=True,
        help="Desired total duration of the output motion in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (rows per second). Default: 30.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Output CSV path. Defaults to '<input_stem>_extended.csv' "
            "in the same directory as the input."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv).expanduser()
    if not in_path.is_absolute():
        in_path = in_path.resolve()

    if not in_path.exists():
        raise SystemExit(f"Input motion file not found: {in_path}")

    if args.duration <= 0:
        raise SystemExit("Duration must be positive.")

    if args.fps <= 0:
        raise SystemExit("FPS must be positive.")

    target_frames = int(round(args.fps * args.duration))
    if target_frames <= 0:
        raise SystemExit(
            f"Computed non-positive target frame count ({target_frames}). "
            "Check fps and duration."
        )

    rows = read_motion(in_path)
    extended = extend_motion(rows, target_frames)

    if args.output:
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = in_path.parent / out_path
    else:
        out_path = in_path.with_name(in_path.stem + "_extended.csv")

    write_motion(out_path, extended)
    print(
        f"Wrote extended motion to {out_path} "
        f"({len(rows)} -> {len(extended)} rows, "
        f"duration ≈ {len(extended) / args.fps:.3f}s at {args.fps} fps)."
    )


if __name__ == "__main__":
    main()

