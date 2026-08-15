"""Regenerates results/accuracy_curve.png from the committed results/rigor.log and
results/baseline.log -- parses the real per-seed, per-checkpoint numbers those steps
already printed, rather than re-running training, so the plot always matches exactly
what's checked into the logs.

Run with `python -m src.plot_accuracy` (from the project root) or `python
src/plot_accuracy.py` (from `src/`).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RIGOR_LOG = RESULTS_DIR / "rigor.log"
BASELINE_LOG = RESULTS_DIR / "baseline.log"
OUT_PATH = RESULTS_DIR / "accuracy_curve.png"

CHANCE = 1.0 / 12

SEED_HEADER_RE = re.compile(r"^=== seed model=(\d+) train=(\d+) ===$")
STEP_ROW_RE = re.compile(
    r"^\s+step\s+(\d+)/\d+\s+induction\(coincidental\)=(\d+\.\d+)\s+"
    r"induction\(fixed_offset\)=(\d+\.\d+)$"
)
TRANSITION_RE = re.compile(
    r"^  transition step \(halfway chance->final\): (\d+|None)")

RANDOM_INIT_RE = re.compile(r"^  induction accuracy \(coincidental\):\s+(\d+\.\d+)")
TRAINED_1LAYER_HEADER_RE = re.compile(r"^baseline B: 1-layer model")


def parse_rigor_log(path=RIGOR_LOG):
    """Returns (steps, per_seed_accuracies, transition_steps): steps is the shared
    list of checkpoint step numbers, per_seed_accuracies is a list of one accuracy
    list per seed (same order as steps), transition_steps is one int-or-None per
    seed.
    """
    steps = None
    per_seed = []
    transition_steps = []
    current_steps = []
    current_accs = []

    def _flush():
        if current_steps:
            nonlocal steps
            if steps is None:
                steps = list(current_steps)
            else:
                assert steps == current_steps, "checkpoint steps differ across seeds"
            per_seed.append(list(current_accs))

    for line in path.read_text().splitlines():
        if SEED_HEADER_RE.match(line):
            _flush()
            current_steps = []
            current_accs = []
            continue
        m = STEP_ROW_RE.match(line)
        if m:
            current_steps.append(int(m.group(1)))
            current_accs.append(float(m.group(2)))
            continue
        m = TRANSITION_RE.match(line)
        if m:
            transition_steps.append(int(m.group(1)) if m.group(1) != "None" else None)
            continue
    _flush()

    assert len(per_seed) >= 3, "expected at least 3 seeds in rigor.log"
    return steps, per_seed, transition_steps


def parse_baseline_log(path=BASELINE_LOG):
    """Returns (random_init_coincidental, trained_1layer_coincidental)."""
    values = []
    in_trained_1layer = False
    for line in path.read_text().splitlines():
        if TRAINED_1LAYER_HEADER_RE.match(line):
            in_trained_1layer = True
        m = RANDOM_INIT_RE.match(line)
        if m:
            values.append((in_trained_1layer, float(m.group(1))))
    random_init = next(v for is_trained, v in values if not is_trained)
    trained_1layer = next(v for is_trained, v in values if is_trained)
    return random_init, trained_1layer


def plot(steps, per_seed, transition_steps, random_init, trained_1layer, out_path=OUT_PATH):
    accs = np.array(per_seed)  # (n_seeds, n_checkpoints)
    mean = accs.mean(axis=0)
    std = accs.std(axis=0)
    mean_transition = np.mean([t for t in transition_steps if t is not None])

    fig, ax = plt.subplots(figsize=(1600 / 150, 900 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.errorbar(steps, mean, yerr=std, marker="o", capsize=3, color="#1f77b4",
                label=f"2-layer trained, coincidental (mean +/- std, {accs.shape[0]} seeds)")
    ax.axhline(trained_1layer, linestyle="--", linewidth=1.5, color="#d62728", alpha=0.9,
               label=f"1-layer baseline, trained (coincidental) = {trained_1layer:.4f}")
    ax.axhline(random_init, linestyle="--", linewidth=1.5, color="gray", alpha=0.8,
               label=f"2-layer baseline, random init (coincidental) = {random_init:.4f}")
    ax.axhline(CHANCE, linestyle=":", linewidth=1.2, color="black", alpha=0.5,
               label=f"chance = 1/12 = {CHANCE:.4f}")
    ax.axvline(mean_transition, linestyle=":", linewidth=1.2, color="#1f77b4", alpha=0.5)
    ax.annotate(f"mean transition step ~= {mean_transition:.0f}",
                xy=(mean_transition, mean.max() * 0.55), fontsize=10, color="#1f77b4",
                ha="left", va="bottom", rotation=90)

    ax.set_xlabel("training step", fontsize=12)
    ax.set_ylabel("coincidental induction accuracy\n(no positional shortcut available)", fontsize=12)
    ax.set_title(
        "induction accuracy plateaus early, indistinguishable from a baseline\n"
        "that structurally cannot compose",
        fontsize=14,
    )
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    steps, per_seed, transition_steps = parse_rigor_log()
    random_init, trained_1layer = parse_baseline_log()
    plot(steps, per_seed, transition_steps, random_init, trained_1layer)
