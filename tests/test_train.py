"""Confirms the reusable training loop runs end to end on a tiny subset without
crashing, and that the loss it produces is finite and trending down.
"""
import numpy as np

from train import train_model


def test_training_loop_runs_and_loss_decreases():
    _, losses = train_model(n_layers=2, num_steps=8, seed_model=100, seed_train=201,
                             batch_size=256, log_every=1000)

    assert len(losses) == 8
    assert all(np.isfinite(l) for l in losses)
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
