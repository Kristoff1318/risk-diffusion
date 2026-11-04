# Superseded contact accounting

This first reproduction correctly restored the original sampler and feedback
controller but iterated all allocated MuJoCo contact entries. Stale entries
inflated contact duration. Inspection showed an allocated buffer of 100 entries
with only one active contact at the terminal state, while an inactive slot
still held a prior wall contact. The current run checks only `[:ncon]`.

Both runs have 9/10 goal successes and 6/10 episodes with wall contact, but
this attempt's per-step contact durations are superseded. The final code also
uses the exact original parenthesization of the feedback action formula.
