# attention-head-emergence

a 2-layer attention-only transformer, forward and backward pass hand-written in numpy,
built to watch induction heads appear during training.

**they did not appear.** the composition advantage the architecture is supposed to buy
never shows up at this scale, and a 1-layer model that theoretically cannot do the task
gets essentially the same score.

## the task

sequences of i.i.d. random tokens from a small vocabulary. the only above-chance signal
at any position is whether that token appeared earlier in the sequence, in which case
the right prediction is whatever token followed it last time. everywhere else is chance
by construction (`src/data.py`).

the sequence's second half is an exact repeat of its first, which means a naive
"induction accuracy" quietly mixes two different things: positions solvable by a fixed
positional offset (attend 16 tokens back, no content matching required) and positions
that repeat at no fixed offset. only the second kind actually tests content-based
induction. every number below is that **coincidental** split (`src/baselines.py`).

## what was expected

elhage et al. (2021) argue a single attention layer cannot implement induction at all:
it needs a previous-token head in an early layer feeding an induction head in a later
one. so the 1-layer control should sit at chance while the 2-layer model pulls away from
it, and the pull-away should look like a step change rather than a slope.

## what happened instead

| model | coincidental induction accuracy |
|---|---|
| chance (1/12) | 0.0833 |
| random-init 2-layer | 0.0797 |
| trained 1-layer | 0.1777 |
| trained 2-layer (5 seeds) | 0.1810 +/- 0.0041 |

- the trained 1-layer model, which the composition argument says cannot do content-based
  induction, reaches 0.1777 (`results/baseline.log`, lines 8 and 21).
- the 2-layer model clears chance in 5/5 seeds but clears that 1-layer number in only
  4/5 (`results/rigor.log`, lines 186 to 192). the two sit inside each other's noise.
- **both attention layers converge to the same fixed-offset pattern the 1-layer model
  already learns.** in `results/headline.gif`, after 500 steps layer 1 and layer 2 both
  show an identical diagonal band at a 16-token offset, nearly matching the 1-layer
  baseline's own pattern, even while the model solves the fixed-offset version of the
  task to near 100% (`results/run.log`).
- a leakage check found 0 collisions between the held-out eval batch and any training
  batch across any seed, in 160,000 rows checked (`results/rigor.log` line 196), so this
  is not train/eval overlap.

`results/FINDING.md` has the full reasoning and, more importantly, what this result does
and does not rule out about induction heads at larger scale.

## layout

| file | what it holds |
|---|---|
| `src/data.py` | synthetic induction-task sequence generator |
| `src/model.py` | attention-only transformer, forward and backward pass |
| `src/train.py` | reusable adam training loop |
| `src/baselines.py` | random-init 2-layer and trained 1-layer baselines |
| `src/experiment.py` | full 2-layer training run with attention checkpoints |
| `src/rigor.py` | multi-seed sweep and the train/eval leakage check |
| `src/plot_accuracy.py` | builds `results/accuracy_curve.png` |
| `src/render_video.py` | builds `results/headline.gif` from the checkpoint npz |
| `results/` | committed logs, checkpoint data, chart and video |

## running it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

regenerating every result from scratch, from the project root, each well under a minute
on cpu:

```
python src/baselines.py > results/baseline.log
python src/experiment.py > results/run.log
python src/rigor.py > results/rigor.log
python src/plot_accuracy.py
python src/render_video.py
```
