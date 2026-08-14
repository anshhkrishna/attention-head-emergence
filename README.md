# attention-head-emergence

> a tiny attention-only transformer, watching induction heads form during training

Status: scaffolded, not yet trained. This README will be rewritten from committed run
logs once the project reaches the writeup step.

## What this tests

A 2-layer attention-only transformer, trained from scratch (NumPy, hand-written forward
and backward pass, no PyTorch or autograd framework) on a synthetic in-context copying
task: sequences of i.i.d. random tokens from a small vocabulary, where the only
above-chance signal at any position is whether that token appeared earlier in the
sequence, in which case the correct prediction is whatever token followed it last time
("induction"). Predicting anywhere else in the sequence is chance by construction.

**Claim:** the model's induction behavior appears abruptly during training, as a
step-change rather than a gradual improvement, and the training step where the model's
attention pattern visibly flips to the induction shape coincides with the step where
induction accuracy jumps.

**Baseline:** the same architecture at random initialization, and a 1-layer variant of
the same model trained for the same number of steps. The composition argument behind
induction heads (Elhage et al., 2021) says a single attention layer cannot implement
induction at all -- it needs a previous-token head in an early layer feeding an
induction head in a later one -- so the 1-layer model is expected to sit at chance
throughout training.

## Repo layout

- `src/data.py` -- synthetic induction-task sequence generator
- `src/model.py` -- attention-only transformer, forward and backward pass
- `src/train.py` -- training loop
- `src/baselines.py` -- random-init and 1-layer baselines
- `src/render_video.py` -- builds `results/headline.gif` from committed checkpoint data
- `results/` -- committed logs, checkpoint data, and the headline chart/video

## Reproducing

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Full reproduction instructions and headline numbers will be added at the writeup step,
once real results exist.
