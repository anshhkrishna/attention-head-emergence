Measured whether a 2-layer attention-only transformer trained from scratch on a
synthetic induction task develops genuine content-based induction, against a 1-layer
baseline that a published composition argument says structurally cannot. After 500
training steps across 5 seeds, coincidental-position induction accuracy (the split of
the metric with no positional shortcut available) reaches 18.10% plus or minus 0.41%,
next to the trained 1-layer baseline's 17.77%; the 2-layer model clears that baseline
in only 4 of 5 seeds, so it is not reliably better, and the two numbers sit within
noise of each other. The surprising part is visual: animating attention across
training shows both of the 2-layer model's layers converging to the exact same
fixed-offset diagonal pattern the 1-layer baseline already learns on its own, rather
than composing into a distinct induction circuit, even as the model reaches close to
100% accuracy on the fixed-offset version of the task that pattern alone can solve.
