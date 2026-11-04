# Setup-only failure

Exact original-sampler parity passed. Execution stopped before the first
episode because `get_target()` returned a tuple, which has no `.copy()` method.
The final runner explicitly converts the target to a NumPy array. No episode
outcomes were produced in this setup attempt.
