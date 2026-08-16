# attention-head-emergence

> a 2-layer attention-only transformer in numpy, testing whether induction heads actually compose

## what this tests

a 2-layer attention-only transformer, trained from scratch (numpy, hand-written forward
and backward pass, no pytorch or autograd framework) on a synthetic in-context copying
task: sequences of i.i.d. random tokens from a small vocabulary, where the only
above-chance signal at any position is whether that token appeared earlier in the
sequence, in which case the correct prediction is whatever token followed it last time
("induction"). predicting anywhere else in the sequence is chance by construction
(`src/data.py`).

**claim:** the model's induction behavior appears abruptly during training, as a
step-change rather than a gradual improvement, and the training step where the model's
attention pattern visibly flips to the induction shape coincides with the step where
induction accuracy jumps.

**baseline:** the same architecture at random initialization, and a 1-layer variant of
the same model trained for the same number of steps. the composition argument behind
induction heads (elhage et al., 2021) says a single attention layer cannot implement
induction at all: it needs a previous-token head in an early layer feeding an induction
head in a later one, so the 1-layer model is expected to sit at chance throughout
training.

the task's second half is an exact repeat of its first, so a plain "induction accuracy"
lumps together positions solvable by a fixed positional offset (attend 16 tokens back,
no content matching needed) with positions that repeat by chance at no fixed offset (the
actual test of content-based induction). every number below is the **coincidental**
split, the one with no positional shortcut available (`src/baselines.py`).

## result: the claim does not hold at this scale

neither the abrupt jump nor the composition advantage shows up.

- **both baselines land near or below the trained model.** the random-init 2-layer
  model scores 0.0797 on coincidental induction, indistinguishable from chance
  (1/12 = 0.0833). the trained 1-layer model, which the composition argument says
  cannot do content-based induction at all, actually reaches 0.1777
  (`results/baseline.log`, lines 8 and 21).
- **the trained 2-layer model does not reliably beat that 1-layer baseline.** across 5
  seeds (500 training steps each), final coincidental accuracy is 0.1810 +/- 0.0041,
  clearing chance in 5/5 seeds but clearing the 1-layer baseline's 0.1777 in only 4/5
  (`results/rigor.log`, lines 186 to 192). the two numbers sit within noise of each
  other.
- **both attention layers converge to the same fixed-offset pattern the 1-layer
  baseline already learns**, rather than composing into a distinct induction circuit.
  this is visible in `results/headline.gif`: after 500 training steps, layer 1 and
  layer 2 both show an identical diagonal band at a 16-token offset, matching the
  1-layer baseline's own attention pattern almost exactly, even though the model
  separately solves the fixed-offset version of the task to near 100%
  (`results/run.log`).
- a leakage check confirmed the held-out evaluation batch never appears in any training
  batch across any seed (0 collisions in 160,000 rows checked, `results/rigor.log` line
  196), so this is not an artifact of train/eval overlap.

full reasoning and the honest caveats, in particular what this result does and does
not rule out about induction heads at larger scale, are in `results/FINDING.md`.

## repo layout

- `src/data.py` -- synthetic induction-task sequence generator
- `src/model.py` -- attention-only transformer, forward and backward pass
- `src/train.py` -- reusable adam training loop
- `src/baselines.py` -- random-init 2-layer and trained 1-layer baselines
- `src/experiment.py` -- full training run of the 2-layer model, with attention
  checkpoints
- `src/rigor.py` -- multi-seed sweep and the train/eval leakage check
- `src/plot_accuracy.py` -- builds `results/accuracy_curve.png` from `results/rigor.log`
  and `results/baseline.log`
- `src/render_video.py` -- builds `results/headline.gif` from
  `results/attention_checkpoints.npz`
- `results/` -- committed logs, checkpoint data, and the headline chart/video

## reproducing

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

to regenerate the full set of results from scratch (run from the project root, each
takes well under a minute on cpu):

```
python src/baselines.py > results/baseline.log
python src/experiment.py > results/run.log
python src/rigor.py > results/rigor.log
python src/plot_accuracy.py
python src/render_video.py
```
