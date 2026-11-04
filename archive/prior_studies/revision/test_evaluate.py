"""Standalone checks: run with ``python risk_diffusion/revision/test_evaluate.py``."""
import pathlib
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from risk_diffusion.revision.evaluate import evaluate, _fractional_cvar
from risk_diffusion.revision.problem import Problem


def main():
    p = Problem(horizon=2, process_sigma=0.0, initial_sigma=0.0, obstacles=((0., 0., .1),))
    u = np.zeros((1, 2, 2)); s = np.array([[-.2, 0.]]); g = np.array([[-.2, 0.]])
    r = evaluate(u, s, g, p, draws=10, seed=3)[0]
    assert r['failures'] == 0 and r['risk'] == 0 and not r['accepted']
    assert np.isclose(r['risk_upper'],1-.05**.1)
    assert r['cvar'] < 0 and r['mean_positive_violation'] == 0

    # Initial state collision is part of the event; multiple circles are supported.
    p2 = Problem(horizon=1, process_sigma=0., initial_sigma=0., obstacles=((0., 0., .2), (3., 3., .2)))
    r = evaluate(np.zeros((1, 1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), p2, draws=7)[0]
    assert r['failures'] == 7 and r['collision'] == 1 and r['goal'] == 1
    assert not r['accepted'] and r['risk_upper'] == 1

    # Exact fractional tail: sorted losses [4,3,2,1], delta*N=2.5.
    assert np.isclose(_fractional_cvar(np.array([1., 2., 3., 4.]), .625), 3.2)
    print('test_evaluate: ok')


if __name__ == '__main__':
    main()
