"""Release-authorized old-fixture native GPU gate: parity THEN timing."""
import gc
import hashlib
import json
import pathlib
import sys
import time
import resource
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'phase5'))
import panda_p0
import covariance_energy as cpu
import gpu_energy as hybrid
import native_energy as native

P = pathlib.Path(__file__).resolve().parent
START = time.perf_counter()
REPORT = {'fresh_tasks_opened': 0, 'new_sampling': 0, 'new_inference': 0,
          'numeric_gate_pass': False, 'timing_ran': False, 'full_events': 4323770}


def guard():
    if time.perf_counter()-START > 65:
        raise TimeoutError('65s internal process guard')


def save():
    REPORT['process_seconds'] = time.perf_counter()-START
    REPORT['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    (P/'NATIVE_GPU_CHECKS.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')


def close(a, b, atol=1e-6, rtol=1e-8):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def scalars(r):
    # One transfer preserves selection/nonfinite checks without11 component
    # synchronization points. Optional components are materialized after timing.
    return torch.stack([r[k].detach() for k in ('energy', 'cost', 'risk_sum')]).cpu().tolist()


def run():
    torch.set_num_threads(2)
    torch.cuda.init(); torch.cuda.synchronize()
    REPORT['GPU_name'] = torch.cuda.get_device_name()
    arm = panda_p0.Arm()
    fixture = P.parent/'phase5/FROZEN_PATHS.npz'
    z = np.load(fixture); qs, targets, cp = z['q'], z['target'], z['control_points']
    xs = np.concatenate((cp[:, 3:26], cp[:, 26:27]), axis=1)
    REPORT['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    b, c = hybrid.basis(); bt, ct = [torch.tensor(v, device='cuda:0', dtype=torch.float64) for v in (b, c)]
    np.testing.assert_allclose(np.einsum('tx,pxj->ptj', b, xs)+c[None, :, None]*qs[:, :1], qs, atol=1e-13)
    begin = time.perf_counter(); context = native.NativeContext(arm, 'cuda:0'); torch.cuda.synchronize()
    REPORT['first_context_setup_seconds'] = time.perf_counter()-begin
    REPORT['full_fixture_parity'] = []
    for slot in range(4):
        guard()
        ref = cpu.FrozenEnergy(arm, qs[slot], targets[slot])
        q = torch.tensor(qs[slot], device='cuda:0', dtype=torch.float64, requires_grad=True)
        model = context.build(q, targets[slot])
        max_block = 0.
        for rb, nb in zip(ref.blocks, model.blocks):
            assert rb[0] == nb[0]
            for j in (1, 2, 3):
                close(nb[j], rb[j])
                max_block = max(max_block, float(abs(nb[j].cpu().numpy()-rb[j]).max()))
            np.testing.assert_array_equal(nb[4].cpu().numpy(), rb[4])
            assert all(not v.requires_grad for v in nb[1:])
        for kind in range(2):
            # Compare every FK transform and sphere point Jacobian, beyond
            # aggregate loss equivalence; preserves pair identities/order.
            with torch.no_grad():
                poses, pts, jac, _, _ = context.geometry(model.qbar, kind)
            oracle = cpu.fk_jac(arm, ref.qbar, kind)
            for link, (r, t) in poses.items():
                close(r, oracle[link][0]); close(t, oracle[link][1])
            for i, (link, xyz, _) in enumerate(arm.spheres[kind]):
                p, j = cpu.point_jac(oracle[link], xyz)
                close(pts[:, i], p); close(jac[:, i], j)
            del poses, pts, jac, oracle
        for j in (0, 1, 2):
            close(model.pose[j], ref.pose[j])
        np.testing.assert_array_equal(model.pose[3].cpu().numpy(), ref.pose[3])
        nr = model.evaluate(q); nr['energy'].backward()
        rr = ref.evaluate(qs[slot])
        assert nr['events'] == rr['events'] == 4323770
        for k in ('risk_sum', 'energy', 'cost'):
            close(nr[k], rr[k])
        close(q.grad, rr['gradient'])
        for k in rr['components']:
            close(nr['components'][k], rr['components'][k]['sum_probability'])
            assert nr['event_counts'][k] == rr['components'][k]['events']
        # Retained hybrid CUDA evaluator provides an additional device oracle.
        hr = hybrid.GPUFrozen(ref).evaluate(q.detach())
        for k in ('risk_sum', 'energy', 'cost'):
            close(nr[k], hr[k])
        REPORT['full_fixture_parity'].append({'slot': slot, 'pass': True,
            'max_block_abs_error': max_block, 'risk_sum': float(nr['risk_sum'].detach())})
        del ref, q, model, nr, rr, hr; gc.collect()
        save()
    # Edge branches use same primitives; pose models still invoke the exact
    # retained CPU/SciPy branch, verified below at zero and near-pi targets.
    zero = torch.tensor([-1., 0., 1.], device='cuda:0', dtype=torch.float64)
    close(hybrid.probability(zero, torch.zeros_like(zero)), [1., 0., 0.], atol=0, rtol=0)
    dd = torch.tensor([[0., 0., 0.], [1e-12, 0., 0.], [2e-12, 0., 0.]], device='cuda:0', dtype=torch.float64)
    _, jj, bad = native.separation(dd, torch.ones((3, 3, 7), device='cuda:0', dtype=torch.float64), .2)
    np.testing.assert_array_equal(bad.cpu().numpy(), [True, True, False]); assert bool(torch.isfinite(jj).all())
    endpoint_pose = cpu.fk_jac(arm, qs[0, -1:], 0)['panda_hand'][0][0]
    for angle in (0., np.pi-2e-6, np.pi-.5e-6):
        target = targets[0].copy(); target[:, :3] = endpoint_pose@Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
        model = context.build(torch.tensor(qs[0], device='cuda:0', dtype=torch.float64), target)
        ref_pose = cpu.pose_box(cpu.fk_jac(arm, qs[0, -1:], 0)['panda_hand'], target)
        close(model.pose[0], ref_pose[0]); close(model.pose[1], ref_pose[1])
        np.testing.assert_array_equal(model.pose[3].cpu().numpy(), ref_pose[2])
    REPORT['edge_checks_pass'] = True
    # Active spline FD with a fixed old anchor; never rebuild either side.
    model = context.build(torch.tensor(qs[0], device='cuda:0', dtype=torch.float64), targets[0])
    x = torch.tensor(xs[0], device='cuda:0', dtype=torch.float64); x[-1, 0] += .03; x.requires_grad_(True)
    start = torch.tensor(qs[0, 0], device='cuda:0', dtype=torch.float64)
    r = model.evaluate(bt@x+ct[:, None]*start); r['energy'].backward()
    d = np.random.default_rng(2026106502).normal(size=xs[0].shape); d /= np.linalg.norm(d)
    dt = torch.tensor(d, device='cuda:0')
    with torch.no_grad():
        plus = model.evaluate(bt@(x+1e-6*dt)+ct[:, None]*start)['energy']
        minus = model.evaluate(bt@(x-1e-6*dt)+ct[:, None]*start)['energy']
    fd = float((plus-minus)/2e-6); analytical = float((x.grad*dt).sum())
    assert abs(fd-analytical) <= 1e-4+1e-5*abs(fd)
    REPORT['fixed_model_spline_FD'] = {'pass': True, 'relative_error': abs(fd-analytical)/max(1., abs(fd)), 'initial_derivative': float(abs(bt[0]).max())}
    del x, model, r, plus, minus
    REPORT['optimizer_equivalence'] = []
    # Reference gradients are produced by the full retained NumPy oracle.
    # Each implementation follows its OWN trajectory and must remain equivalent.
    for slot in (0, 1):
        xn = torch.tensor(xs[slot], device='cuda:0', dtype=torch.float64, requires_grad=True)
        xr = xn.detach().clone().requires_grad_(True)
        start = torch.tensor(qs[slot, 0], device='cuda:0', dtype=torch.float64)
        on, ore = torch.optim.Adam([xn], lr=.003), torch.optim.Adam([xr], lr=.003)
        bn = br = None; kn = kr = None; inative = iref = None
        discrepancies = []
        for it in range(8):
            guard(); on.zero_grad(); ore.zero_grad()
            qn = bt@xn+ct[:, None]*start; qr = b@xr.detach().cpu().numpy()+c[:, None]*qs[slot, 0]
            ref = cpu.FrozenEnergy(arm, qr, targets[slot]); rr = ref.evaluate(qr)
            model = context.build(qn, targets[slot]); nr = model.evaluate(qn)
            assert nr['events'] == rr['events'] == 4323770
            for k in ('risk_sum', 'energy', 'cost'):
                close(nr[k], rr[k])
            for k in rr['components']:
                close(nr['components'][k], rr['components'][k]['sum_probability'])
            nr['energy'].backward(); xr.grad = torch.tensor(b.T@rr['gradient'], device='cuda:0')
            close(xn.grad, xr.grad)
            ev, jv, rv = scalars(nr)
            keyn = (0, jv) if rv <= .05 else (1, rv)
            keyr = (0, rr['cost']) if rr['risk_sum'] <= .05 else (1, rr['risk_sum'])
            if kn is None or keyn < kn:
                kn, bn, inative = keyn, xn.detach().clone(), it
            if kr is None or keyr < kr:
                kr, br, iref = keyr, xr.detach().clone(), it
            torch.nn.utils.clip_grad_norm_([xn], 20.); torch.nn.utils.clip_grad_norm_([xr], 20.)
            close(xn.grad, xr.grad); on.step(); ore.step(); close(xn, xr)
            for name in ('exp_avg', 'exp_avg_sq'):
                close(on.state[xn][name], ore.state[xr][name])
            discrepancies.append(float(abs(xn-xr).max().detach()))
            del qn, ref, rr, model, nr
        assert inative == iref; close(bn, br)
        close((bt@bn+ct[:, None]*start)[0], qs[slot, 0], atol=1e-14, rtol=0)
        REPORT['optimizer_equivalence'].append({'slot': slot, 'iterations': 8,
            'selected_iteration_native': inative, 'selected_iteration_reference': iref,
            'max_update_abs_errors': discrepancies, 'pass': True})
        save(); del xn, xr, on, ore, bn, br; gc.collect()
    REPORT['numeric_gate_pass'] = True; save()
    # Speed claim only AFTER all full-grid and optimizer numeric gates passed.
    guard(); torch.cuda.synchronize(); begin_request = time.perf_counter()
    begin = time.perf_counter(); timed_context = native.NativeContext(arm, 'cuda:0'); torch.cuda.synchronize()
    cache_seconds = time.perf_counter()-begin
    traces = []; outputs = {}
    for slot in (0, 1):
        x = torch.tensor(xs[slot], device='cuda:0', dtype=torch.float64, requires_grad=True)
        start = torch.tensor(qs[slot, 0], device='cuda:0', dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=.003); best = None; key = None; trace = []
        for it in range(8):
            guard(); optimizer.zero_grad(); q = bt@x+ct[:, None]*start
            torch.cuda.synchronize(); begin = time.perf_counter()
            model = timed_context.build(q, targets[slot]); torch.cuda.synchronize(); built = time.perf_counter()-begin
            begin = time.perf_counter(); r = model.evaluate(q); torch.cuda.synchronize(); evaluated = time.perf_counter()-begin
            begin = time.perf_counter(); r['energy'].backward(); torch.cuda.synchronize(); backward = time.perf_counter()-begin
            begin = time.perf_counter(); ev, jv, rv = scalars(r)
            if not np.isfinite([ev, jv, rv]).all():
                raise ValueError('nonfinite timed iterate')
            assert r['events'] == 4323770
            candidate_key = (0, jv) if rv <= .05 else (1, rv)
            if key is None or candidate_key < key:
                best, key, chosen = x.detach().clone(), candidate_key, it
            norm = torch.nn.utils.clip_grad_norm_([x], 20.)
            optimizer.step(); torch.cuda.synchronize(); update = time.perf_counter()-begin
            trace.append({'iteration': it, 'anchor_build_including_terminal_transfer_seconds': built,
                'evaluation_seconds': evaluated, 'backward_seconds': backward,
                'checks_selection_clip_Adam_seconds': update, 'energy': ev, 'cost': jv, 'risk_sum': rv,
                'components_device': {k: v.detach() for k, v in r['components'].items()}, 'gradient_norm_device': norm.detach()})
            del q, model, r
        outputs[f'slot{slot}_x'] = best.cpu().numpy()
        outputs[f'slot{slot}_q'] = (bt@best+ct[:, None]*start).cpu().numpy()
        traces.append({'slot': slot, 'iterations': 8, 'selected_iteration': chosen, 'trace': trace})
    torch.cuda.synchronize(); elapsed = time.perf_counter()-begin_request
    # Optional detailed logging outside request timer; no selected decision
    # or finite check is postponed. This is disclosed in reported latency.
    for slot in traces:
        for r in slot['trace']:
            r['components'] = {k: float(v) for k, v in r.pop('components_device').items()}
            r['gradient_norm_before_cap'] = float(r.pop('gradient_norm_device'))
    REPORT['timing_ran'] = True
    REPORT['timing'] = {'two_old_fixture_slots_each_own_start_and_goal': True,
        'cache_setup_in_request_seconds': cache_seconds, 'whole_request_seconds': elapsed,
        'target_seconds': 2., 'pass': elapsed <= 2., 'slots': traces,
        'all_step_timing_synchronizations_charged': True,
        'optional_component_log_materialization_outside_request': True,
        'corrected_paths_not_independently_audited': True}
    np.savez_compressed(P/'NATIVE_OLD_TIMING_PATHS.npz', **outputs)
    REPORT['terminal_gate'] = 'PASS: native old-fixture equivalence and runtime gate' if elapsed <= 2. else 'FAIL: native runtime exceeds2s gate'


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        import traceback
        REPORT.update(terminal_gate='FAIL: native qualification exception; no speed claim', error=repr(error), traceback=traceback.format_exc())
    finally:
        save()
    print(json.dumps({k: v for k, v in REPORT.items() if k != 'timing'}, indent=2))
    sys.exit(0 if REPORT['numeric_gate_pass'] and REPORT.get('timing_ran') else 1)
