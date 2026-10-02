"""Bounded, deterministic old-fixture qualification; no sampling or optimizer."""
import hashlib
import json
import math
import pathlib
import sys
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'phase5'))
import panda_p0
import covariance_energy as e

P = pathlib.Path(__file__).resolve().parent
INDICES = [0, 1, 10, 500, 1000, 1500, 1999, 2000]
DETAILS = {}


class Equations(unittest.TestCase):
    def test_augmented_covariance(self):
        f = np.array([[.9, .05], [0, 1.]])
        cov = np.diag([.004**2, .002**2])
        history = [cov.copy()]
        for _ in range(200):
            cov = f@cov@f.T+np.diag([.05**2*.010**2, 0])
            history.append(cov.copy())
        err = []
        for k, l in [(0, 0), (1, 0), (20, 70), (200, 200)]:
            actual = (np.linalg.matrix_power(f, l-k)@history[k])[0, 0]
            err.append(abs(actual-e.cross_cov(k, l)))
        self.assertLess(max(err), 1e-18)
        cmd, exe = e.velocity_variances()
        for k in [0, 1, 50, 199]:
            self.assertAlmostEqual(cmd[k], 4*history[k][0, 0], delta=1e-18)
            actual = np.array([-2., 1.])@history[k]@np.array([-2., 1.])+.010**2
            self.assertAlmostEqual(exe[k], actual, delta=1e-18)
        DETAILS['covariance_max_abs_error'] = max(err)

    def test_dense_interpolation(self):
        # e_(k+1)=a e_k+dt b+dt eta, with exact covariance of [e_k,b].
        k = 50; r = .1
        cb = (1-e.A**k)/e.K*.002**2
        cov = np.array([[e.cross_cov(k, k), cb], [cb, .002**2]])
        h = np.array([1-r+r*e.A, r*e.DT])
        actual = h@cov@h+(r*e.DT)**2*.010**2
        self.assertAlmostEqual(e.dense_variance([501])[0], actual, delta=1e-18)
        self.assertAlmostEqual(e.dense_variance([2000])[0], e.cross_cov(200, 200), delta=1e-18)

    def test_deterministic_and_tail(self):
        p, dp = e.probability(np.array([-1., 0., 1.]), np.zeros(3))
        np.testing.assert_array_equal(p, [1., 0., 0.])
        np.testing.assert_array_equal(dp, 0.)
        p, _ = e.probability(np.array([0., 1., -1.]), np.ones(3))
        np.testing.assert_allclose(p, [.5, .15865525393145707, .8413447460685429])
        p, dp = e.probability(np.array([.3]), np.array([.2]))
        fd = (e.probability(np.array([.300001]), np.array([.2]))[0]-e.probability(np.array([.299999]), np.array([.2]))[0])/2e-6
        np.testing.assert_allclose(dp, fd, rtol=1e-8)

    def test_psd_and_nonfinite(self):
        for c in [-np.eye(7)*1e-4, np.eye(7)*float('nan')]:
            with self.assertRaises(ValueError):
                e.checked_psd(c)
        c = np.eye(7)*1e-6; c[0, 0] = -1e-16
        fixed, count = e.checked_psd(c)
        self.assertEqual(count, 1); self.assertGreaterEqual(np.linalg.eigvalsh(fixed).min(), 0)
        c[0, 1] = 1e-3
        with self.assertRaises(ValueError):
            e.checked_psd(c)

    def test_coincident(self):
        g, j, bad = e.separation(np.zeros((1, 3)), np.ones((1, 3, 7)), .2)
        self.assertTrue(bad[0]); self.assertTrue(np.isfinite(j).all())
        p, dp = e.probability(g, np.zeros(1), bad)
        np.testing.assert_array_equal(p, 1.); np.testing.assert_array_equal(dp, 0.)

    def test_cost_gradient(self):
        q = np.arange(201)[:, None]*np.ones((1, 7))*.001
        q[100, 2] += .01
        d = np.random.default_rng(2026106500).normal(size=q.shape)
        d /= np.linalg.norm(d)
        value, grad = e.cost_gradient(q)
        fd = (e.cost_gradient(q+1e-6*d)[0]-e.cost_gradient(q-1e-6*d)[0])/2e-6
        self.assertAlmostEqual(float(np.sum(grad*d)), fd, delta=1e-7)


class OldFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = panda_p0.Arm()
        fixture = P.parent/'phase5/FROZEN_PATHS.npz'
        z = np.load(fixture)
        cls.q, cls.target = z['q'][0], z['target'][0]
        DETAILS['fixture'] = {'path': 'phase5/FROZEN_PATHS.npz', 'slot': 0,
                              'sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
                              'dense_indices': INDICES}

    def test_fk_clearance_jacobians(self):
        q = self.q[100:101]
        errs = []; parity = []
        for kind in range(2):
            poses, world, own = e.geometry(self.arm, q, kind)
            native = self.arm.fk(q, kind)
            for link in poses:
                np.testing.assert_allclose(poses[link][0], native[link][0], atol=1e-14)
                np.testing.assert_allclose(poses[link][1], native[link][1], atol=1e-14)
            for j in range(7):
                d = np.zeros_like(q); d[0, j] = 1e-6
                _, wp, op = e.geometry(self.arm, q+d, kind)
                _, wm, om = e.geometry(self.arm, q-d, kind)
                for base, plus, minus in [(world, wp, wm), (own, op, om)]:
                    errs.append(float(np.max(abs(base[1][..., j]-(plus[0]-minus[0])/2e-6))))
            parity.extend([float(world[0].min()), float(own[0].min())])
        native, _ = self.arm.margins(q)
        np.testing.assert_allclose(np.array(parity)[[0, 2, 1, 3]], native[0, :4], atol=1e-14)
        self.assertLess(max(errs), 1e-6)
        DETAILS['clearance_jacobian_max_abs_error'] = max(errs)
        DETAILS['native_margin_parity'] = True

    def test_pose_jacobians_and_boxes(self):
        q = self.q[-1:]
        # Offset old pose target to exercise a nonzero rotation log.
        target = self.target.copy()
        target[:, :3] = target[:, :3]@Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
        pose = e.fk_jac(self.arm, q, 0)['panda_hand']
        g, jac, bad = e.pose_box(pose, target)
        errs = []
        for j in range(7):
            d = np.zeros_like(q); d[0, j] = 1e-6
            gp = e.pose_box(e.fk_jac(self.arm, q+d, 0)['panda_hand'], target)[0]
            gm = e.pose_box(e.fk_jac(self.arm, q-d, 0)['panda_hand'], target)[0]
            errs.append(float(np.max(abs(jac[:, j]-(gp-gm)/2e-6))))
        self.assertLess(max(errs), 1e-6); self.assertFalse(bad.any())
        widths = (g[:6]+g[6:])/2
        np.testing.assert_allclose(widths, np.array([.01]*3+[math.radians(2)]*3)/math.sqrt(3))
        # Pi singularity is rejected without NaN normals/derivatives.
        pi_target = self.target.copy()
        pi_target[:, :3] = pose[0][0]@Rotation.from_rotvec([math.pi, 0, 0]).as_matrix()
        _, jj, bb = e.pose_box(pose, pi_target)
        self.assertTrue(bb[3]); self.assertTrue(np.isfinite(jj).all())
        DETAILS['pose_jacobian_max_abs_error'] = max(errs)

    def test_frozen_energy(self):
        beg = time.perf_counter()
        model = e.FrozenEnergy(self.arm, self.q, self.target, INDICES)
        built = time.perf_counter()-beg
        base = model.evaluate(self.q)
        d = np.random.default_rng(2026106501).normal(size=self.q.shape)
        d[0] = 0.; d /= np.linalg.norm(d)
        # Crucially reuse SAME frozen object on both sides.
        plus = model.evaluate(self.q+1e-6*d)
        minus = model.evaluate(self.q-1e-6*d)
        fd = (plus['energy']-minus['energy'])/2e-6
        analytic = float(np.sum(base['gradient']*d))
        err = abs(fd-analytic)
        self.assertLessEqual(err, 1e-4+1e-5*abs(fd))
        production = e.production_event_count(self.arm)
        expected = len(INDICES)*production['geometry_joint_per_state']+5612
        self.assertEqual(base['events'], expected)
        self.assertTrue(base['valid']); self.assertFalse(base['production_grid'])
        DETAILS['bounded_frozen_energy'] = {k: v for k, v in base.items() if k not in ('gradient', 'risk_gradient')}
        DETAILS['bounded_frozen_energy'].update(build_seconds=built,
            total_check_seconds=time.perf_counter()-beg, fd_directional=fd,
            analytic_directional=analytic, derivative_abs_error=err,
            production_event_count=production,
            subset_exceeds_budget=base['risk_sum'] > .05,
            interpretation='subset numeric check only; cannot certify nonlinear risk or full-grid usability')


if __name__ == '__main__':
    beg = time.perf_counter()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Equations, OldFixture)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'pass': result.wasSuccessful(), 'tests_run': result.testsRun,
              'failures': [str(v[0])+': '+v[1] for v in result.failures],
              'errors': [str(v[0])+': '+v[1] for v in result.errors],
              'seconds': time.perf_counter()-beg, 'details': DETAILS,
              'scope': 'CPU deterministic unit and first old-fixture checks only',
              'fresh_tasks_opened': 0, 'optimizer_runs': 0, 'GPU_calls': 0}
    (P/'CPU_QUALIFICATION.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
