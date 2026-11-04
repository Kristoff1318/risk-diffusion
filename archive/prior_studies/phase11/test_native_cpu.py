"""CPU-only invariant-cache/native-anchor equivalence unit tests."""
import json
import pathlib
import sys
import time
import unittest
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'phase5'))
import panda_p0
import covariance_energy as old
import gpu_energy as gpu
import native_energy as native

P = pathlib.Path(__file__).resolve().parent
IX = [0, 1, 10, 500, 1000, 1500, 1999, 2000]
DETAILS = {}


class NativeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.arm = panda_p0.Arm()
        cls.z = np.load(P.parent/'phase5/FROZEN_PATHS.npz')
        cls.context = native.NativeContext(cls.arm, 'cpu', IX)

    def test_all_old_fixture_blocks(self):
        errors = []
        for slot in range(4):
            q, target = self.z['q'][slot], self.z['target'][slot]
            ref = old.FrozenEnergy(self.arm, q, target, IX)
            new = self.context.build(torch.tensor(q, dtype=torch.float64), target)
            self.assertTrue(new.valid)
            for rb, nb in zip(ref.blocks, new.blocks):
                self.assertEqual(rb[0], nb[0])
                for j in [1, 2, 3]:
                    np.testing.assert_allclose(nb[j].numpy(), rb[j], atol=1e-12, rtol=1e-8)
                    errors.append(float(abs(nb[j].numpy()-rb[j]).max()))
                np.testing.assert_array_equal(nb[4].numpy(), rb[4])
                self.assertFalse(any(v.requires_grad for v in nb[1:]))
            for j in [0, 1, 2]:
                np.testing.assert_allclose(new.pose[j].numpy(), ref.pose[j], atol=1e-14, rtol=1e-8)
            qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
            nr = new.evaluate(qt); nr['energy'].backward()
            rr = ref.evaluate(q)
            self.assertEqual(nr['events'], rr['events'])
            self.assertEqual(old.production_event_count(self.arm)['total'], 4323770)
            for k in ['risk_sum', 'energy', 'cost']:
                np.testing.assert_allclose(float(nr[k].detach()), rr[k], atol=1e-6, rtol=1e-8)
            np.testing.assert_allclose(qt.grad.numpy(), rr['gradient'], atol=1e-6, rtol=1e-8)
            for k in rr['components']:
                np.testing.assert_allclose(float(nr['components'][k].detach()), rr['components'][k]['sum_probability'], atol=1e-6, rtol=1e-8)
        DETAILS['max_old_fixture_block_abs_error'] = max(errors)

    def test_fk_point_jacobian(self):
        q = old.dense(self.z['q'][0], IX)
        for kind in range(2):
            poses, pts, jac, _, _ = self.context.geometry(torch.tensor(q, dtype=torch.float64), kind)
            ref = old.fk_jac(self.arm, q, kind)
            for link, (r, t) in poses.items():
                np.testing.assert_allclose(r.numpy(), ref[link][0], atol=1e-14)
                np.testing.assert_allclose(t.numpy(), ref[link][1], atol=1e-14)
            for i, (link, xyz, _) in enumerate(self.arm.spheres[kind]):
                p, j = old.point_jac(ref[link], xyz)
                np.testing.assert_allclose(pts[:, i].numpy(), p, atol=1e-14)
                np.testing.assert_allclose(jac[:, i].numpy(), j, atol=1e-14)

    def test_cache_and_fresh_anchor(self):
        q = torch.tensor(self.z['q'][0], dtype=torch.float64, requires_grad=True)
        target = self.z['target'][0]
        first = self.context.build(q, target)
        altered = q.detach().clone(); altered[100, 0] += .01
        second = self.context.build(altered, target)
        self.assertEqual(first.var.data_ptr(), second.var.data_ptr())
        self.assertEqual(first.indices.data_ptr(), second.indices.data_ptr())
        self.assertNotEqual(first.anchor.data_ptr(), second.anchor.data_ptr())
        self.assertNotEqual(first.blocks[0][2].data_ptr(), second.blocks[0][2].data_ptr())
        self.assertGreater(float(abs(first.blocks[0][1]-second.blocks[0][1]).max()), 0.)
        self.assertFalse(first.anchor.requires_grad)
        np.testing.assert_array_equal(first.anchor.numpy(), self.z['q'][0])

    def test_active_spline_gradient_and_adam_arithmetic(self):
        q = self.z['q'][0]; target = self.z['target'][0]
        ref = old.FrozenEnergy(self.arm, q, target, IX)
        new = self.context.build(torch.tensor(q, dtype=torch.float64), target)
        b, c = gpu.basis(); bt = torch.tensor(b); ct = torch.tensor(c)
        cp = self.z['control_points'][0]
        x0 = np.concatenate((cp[3:26], cp[26:27])); x0[-1, 0] += .03
        x = torch.tensor(x0, requires_grad=True)
        nr = new.evaluate(bt@x+ct[:, None]*torch.tensor(q[0])); nr['energy'].backward()
        rr = ref.evaluate(b@x0+c[:, None]*q[0])
        np.testing.assert_allclose(x.grad.numpy(), b.T@rr['gradient'], atol=1e-6, rtol=1e-8)
        self.assertGreater(float(nr['risk_sum']), .05)
        d0 = np.random.default_rng(2026106502).normal(size=x0.shape); d0 /= np.linalg.norm(d0)
        d = torch.tensor(d0)
        with torch.no_grad():
            plus = new.evaluate(bt@(x+1e-6*d)+ct[:, None]*torch.tensor(q[0]))['energy']
            minus = new.evaluate(bt@(x-1e-6*d)+ct[:, None]*torch.tensor(q[0]))['energy']
        fd = float((plus-minus)/2e-6); analytic = float((x.grad*d).sum())
        self.assertLessEqual(abs(fd-analytic), 1e-4+1e-5*abs(fd))
        self.assertEqual(float(abs(bt[0]).max()), 0.)
        # One parameter-update arithmetic unit check on same fixed gradients;
        # not an optimizer loop/planning run or independent cost/risk claim.
        xr = torch.tensor(x0, requires_grad=True)
        xr.grad = torch.tensor(b.T@rr['gradient'])
        opn = torch.optim.Adam([x], lr=.003); opr = torch.optim.Adam([xr], lr=.003)
        torch.nn.utils.clip_grad_norm_([x], 20.); torch.nn.utils.clip_grad_norm_([xr], 20.)
        np.testing.assert_allclose(x.grad.numpy(), xr.grad.numpy(), atol=1e-10, rtol=1e-8)
        opn.step(); opr.step()
        np.testing.assert_allclose(x.detach().numpy(), xr.detach().numpy(), atol=1e-10, rtol=1e-8)
        for k in ['exp_avg', 'exp_avg_sq']:
            np.testing.assert_allclose(opn.state[x][k].numpy(), opr.state[xr][k].numpy(), atol=1e-10, rtol=1e-8)
        DETAILS['active_spline_FD_relative_error'] = abs(fd-analytic)/max(1., abs(fd))

    def test_edges(self):
        d = torch.tensor([[0., 0., 0.], [1e-12, 0., 0.], [2e-12, 0., 0.]], dtype=torch.float64)
        jac = torch.ones((3, 3, 7), dtype=torch.float64)
        g, j, bad = native.separation(d, jac, .2)
        rg, rj, rb = old.separation(d.numpy(), jac.numpy(), .2)
        np.testing.assert_array_equal(bad.numpy(), rb)
        np.testing.assert_allclose(j.numpy(), rj, atol=1e-14)
        p = gpu.probability(g, torch.zeros_like(g), bad)
        np.testing.assert_array_equal(p.numpy(), old.probability(rg, np.zeros(3), rb)[0])
        margins = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        np.testing.assert_array_equal(gpu.probability(margins, torch.zeros(3)).numpy(), [1., 0., 0.])
        invalid = torch.tensor(self.z['q'][0]); invalid[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.context.build(invalid, self.z['target'][0])
        with self.assertRaises(ValueError):
            self.context.build(torch.tensor(self.z['q'][0], dtype=torch.float32), self.z['target'][0])
        with self.assertRaises(ValueError):
            old.checked_psd(-np.eye(7)*1e-4)


if __name__ == '__main__':
    begin = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeCPU))
    report = {'pass': result.wasSuccessful(), 'tests_run': result.testsRun,
        'seconds': time.perf_counter()-begin, 'details': DETAILS,
        'failures': [str(v[0])+': '+v[1] for v in result.failures],
        'errors': [str(v[0])+': '+v[1] for v in result.errors],
        'device': 'cpu', 'GPU_calls': 0, 'fresh_tasks_opened': 0,
        'scope': 'bounded unit equivalence; no timing benchmark/planning experiment'}
    (P/'NATIVE_CPU_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
