"""Plot training curves from a TensorBoard events file.

Usage:
    python scripts/plot_training_curves.py <tfevents_file> <output_png>

Picks 6 informative scalars for the Lift-Cube run and lays them out in a 2x3 grid.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PANELS = [
    ("Train/mean_reward", "Mean reward"),
    ("Episode_Reward/lifting_object", "Lift reward"),
    ("Episode_Reward/object_goal_tracking_fine_grained", "Goal-tracking (fine, std=0.05)"),
    ("Metrics/object_pose/position_error", "Cube ↔ goal distance (m)"),
    ("Episode_Termination/object_dropping", "Drop rate"),
    ("Loss/value", "Value loss"),
]


def load_scalar(ea: EventAccumulator, key: str):
    if key not in ea.Tags()["scalars"]:
        return None, None
    events = ea.Scalars(key)
    return [e.step for e in events], [e.value for e in events]


def main(tfevents: str, out_png: str):
    ea = EventAccumulator(tfevents, size_guidance={"scalars": 0})
    ea.Reload()

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, PANELS):
        steps, values = load_scalar(ea, key)
        if steps is None:
            ax.set_title(f"{title}\n(missing: {key})")
            ax.axis("off")
            continue
        ax.plot(steps, values, color="C0", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)

    fig.suptitle("Isaac-Lift-Cube-Franka-v0: PPO training (seed 42, 1500 iters, A10G, 24.2 min)", fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: plot_training_curves.py <tfevents> <out_png>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
