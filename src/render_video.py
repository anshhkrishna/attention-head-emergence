"""Builds results/headline.gif from the committed results/attention_checkpoints.npz
(the trained 2-layer model's per-layer attention on a fixed probe sequence, snapshotted
across training) plus the two reference points from results/baseline.log, regenerated
here on the same probe sequence so all three are visually comparable: the 2-layer model
at random initialization, and the 1-layer model trained for the same number of steps.
Both baselines are deterministic reruns of exactly what results/baseline.log already
reports (same seeds, same hyperparameters), not new experiments.

Run with `python src/render_video.py` (from the project root).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from baselines import D_MODEL, MODEL_SEED, NUM_TRAIN_STEPS, TRAIN_RNG_SEED, train_baseline
from data import SEQ_LEN, VOCAB_SIZE
from model import forward, init_params

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS_PATH = RESULTS_DIR / "attention_checkpoints.npz"
RUN_LOG = RESULTS_DIR / "run.log"
OUT_PATH = RESULTS_DIR / "headline.gif"

CHANCE = 1.0 / VOCAB_SIZE
BASELINE_1LAYER_COINCIDENTAL = 0.1777  # trained 1-layer baseline, results/baseline.log
FRAME_MS = 500  # 2 checkpoints/second
HOLD_MS = 2500  # how long the final frame lingers, so the plateau is visible not flashed

STEP_ROW_RE = re.compile(
    r"^step\s+(\d+)/\d+\s+induction\(coincidental\)=(\d+\.\d+)")


def _parse_coincidental_by_step(path=RUN_LOG):
    by_step = {}
    for line in path.read_text().splitlines():
        m = STEP_ROW_RE.match(line)
        if m:
            by_step[int(m.group(1))] = float(m.group(2))
    return by_step


def _attention_for(params, probe_tokens):
    """Per-layer attention matrices for a single probe sequence: (n_layers, seq, seq)."""
    tokens = probe_tokens[None, :]
    _, cache = forward(params, tokens, cache=True)
    return np.stack([layer["A"][0] for layer in cache["layers"]])


def build_frames():
    data = np.load(CHECKPOINTS_PATH)
    steps = data["steps"]
    trained_attn = data["attention"]  # (n_checkpoints, 2, seq, seq)
    probe_tokens = data["probe_tokens"]
    coincidental_by_step = _parse_coincidental_by_step()

    print("regenerating random-init 2-layer baseline attention (no training needed)")
    random_init_params = init_params(2, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=MODEL_SEED)
    random_init_attn = _attention_for(random_init_params, probe_tokens)  # (2, seq, seq)

    print(f"retraining 1-layer baseline for {NUM_TRAIN_STEPS} steps "
          f"(model_seed={MODEL_SEED}, train_rng_seed={TRAIN_RNG_SEED}) to get its attention")
    trained_1layer_params = train_baseline(1, NUM_TRAIN_STEPS, MODEL_SEED, TRAIN_RNG_SEED)
    trained_1layer_attn = _attention_for(trained_1layer_params, probe_tokens)  # (1, seq, seq)

    return steps, trained_attn, coincidental_by_step, random_init_attn, trained_1layer_attn


def render(steps, trained_attn, coincidental_by_step, random_init_attn, trained_1layer_attn,
           out_path=OUT_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(1000 / 150, 1150 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    (ax_l1, ax_l2), (ax_rand, ax_1layer) = axes

    imshow_kwargs = dict(vmin=0.0, vmax=1.0, cmap="viridis", origin="upper")
    im_l1 = ax_l1.imshow(trained_attn[0, 0], **imshow_kwargs)
    im_l2 = ax_l2.imshow(trained_attn[0, 1], **imshow_kwargs)
    ax_rand.imshow(random_init_attn[1], **imshow_kwargs)
    ax_1layer.imshow(trained_1layer_attn[0], **imshow_kwargs)

    ax_l1.set_title("layer 1 (2-layer, training)", fontsize=10)
    ax_l2.set_title("layer 2 (2-layer, training)", fontsize=10)
    ax_rand.set_title("layer 2 (2-layer, random init)", fontsize=9)
    ax_1layer.set_title("1-layer baseline (trained)", fontsize=9)

    for ax in (ax_l1, ax_l2, ax_rand, ax_1layer):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "attention converges to a single fixed offset,\nnot genuine content-based induction",
        fontsize=13, y=0.98,
    )
    step_text = fig.text(0.5, 0.80, "", ha="center", fontsize=9, color="#1f77b4")
    fig.subplots_adjust(top=0.76, bottom=0.03, left=0.04, right=0.97, hspace=0.3, wspace=0.12)

    n_checkpoints = len(steps)
    frames = []
    for idx in range(n_checkpoints):
        step = int(steps[idx])
        im_l1.set_data(trained_attn[idx, 0])
        im_l2.set_data(trained_attn[idx, 1])
        acc = coincidental_by_step[step]
        step_text.set_text(
            f"step {step}/{int(steps[-1])}   coincidental accuracy = {acc:.3f}\n"
            f"(chance = {CHANCE:.3f}, 1-layer baseline = {BASELINE_1LAYER_COINCIDENTAL:.3f})"
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
    plt.close(fig)

    durations = [FRAME_MS] * (n_checkpoints - 1) + [HOLD_MS]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations,
                    loop=0)
    total_s = sum(durations) / 1000
    print(f"wrote {out_path} ({n_checkpoints} frames, ~{total_s:.1f}s)")


if __name__ == "__main__":
    render(*build_frames())
