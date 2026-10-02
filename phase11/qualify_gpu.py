"""Frozen old-fixture qualification/timing only. No inference/new sampling."""
import gc
import hashlib
import json
import pathlib
import resource
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'phase5'))
import panda_p0
import covariance_energy as cpu
import gpu_energy as gpu

P = pathlib.Path(__file__).resolve().parent
START = time.perf_counter()
REPORT = {'fresh_tasks_opened': 0, 'new_inference': 0, 'new_sampling': 0,
          'architecture': 'complete CPU frozen-model builder plus CUDA float64 evaluation',
          'comparison': 'old-fixture numeric qualification and timing, not a planning comparison'}


def save():
    REPORT['process_seconds'] = time.perf_counter()-START
    REPORT['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    (P/'GPU_QUALIFICATION.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')


def guard():
    if time.perf_counter()-START > 60:
        raise TimeoutError('60s process guard')


def run():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('released GPU unavailable')
    torch.cuda.init(); torch.cuda.synchronize()
    REPORT['GPU_name'] = torch.cuda.get_device_name()
    arm = panda_p0.Arm()
    path = P.parent/'phase5/FROZEN_PATHS.npz'
    z = np.load(path)
    paths, targets, cp = z['q'], z['target'], z['control_points']
    REPORT['fixture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    b, c = gpu.basis()
    controls = np.concatenate((cp[:, 3:26], cp[:, 26:27]), axis=1)
    recon = np.einsum('tx,pxj->ptj', b, controls)+c[None, :, None]*paths[:, :1]
    np.testing.assert_allclose(recon, paths, atol=1e-13)
    REPORT['spline_reconstruction_max_abs'] = float(abs(recon-paths).max())
    bt = torch.as_tensor(b, device='cuda', dtype=torch.float64)
    ct = torch.as_tensor(c, device='cuda', dtype=torch.float64)
    source = P.parent/'phase6/RUN_256_10.json'
    old = json.loads(source.read_text())
    old_rows = old['rows']
    REPORT['independent_risk_source'] = {'path': 'phase6/RUN_256_10.json',
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'reuse': 'existing independent DOUBLE-regime whole-arm hard-event measurements; no new draws'}
    REPORT['full_grid_fixtures'] = []
    for slot in range(4):
        guard(); begin = time.perf_counter()
        model = cpu.FrozenEnergy(arm, paths[slot], targets[slot])
        built = time.perf_counter()-begin
        expected = cpu.production_event_count(arm)['total']
        device = gpu.GPUFrozen(model)
        torch.cuda.synchronize()
        q = device.cast(paths[slot]).requires_grad_(True)
        r = device.evaluate(q); r['energy'].backward(); torch.cuda.synchronize()
        actual = gpu.numeric(r)
        assert actual['events'] == expected == 4323770
        evidence = next(v for v in old_rows if v['path'] == slot and v['regime'] == 'double')
        row = {'slot': slot, 'CPU_build_seconds': built,
               'complete_build_transfer_evaluate_backward_seconds': time.perf_counter()-begin,
               'surrogate': actual, 'independent_existing_risk': evidence,
               'within_surrogate_budget': actual['risk_sum'] <= .05,
               'valid': model.valid}
        if slot == 0:
            rc = model.evaluate(paths[slot])
            for key in ['energy', 'risk_sum', 'cost']:
                np.testing.assert_allclose(actual[key], rc[key], atol=1e-6, rtol=1e-8)
            np.testing.assert_allclose(q.grad.cpu().numpy(), rc['gradient'], atol=1e-6, rtol=1e-8)
            for key in rc['components']:
                np.testing.assert_allclose(actual['components'][key], rc['components'][key]['sum_probability'], atol=1e-6, rtol=1e-8)
            REPORT['full_grid_backend_parity'] = {'pass': True,
                'gradient_max_abs': float(abs(q.grad.cpu().numpy()-rc['gradient']).max())}
            # Active risk spline probe, SAME frozen object on every side.
            x = device.cast(controls[slot]).clone(); x[-1, 0] += .03; x.requires_grad_(True)
            start = device.cast(paths[slot, 0])
            qx = bt@x+ct[:, None]*start
            rr = device.evaluate(qx); rr['energy'].backward(); torch.cuda.synchronize()
            xc = x.detach().cpu().numpy(); qc = b@xc+c[:, None]*paths[slot, 0]
            cr = model.evaluate(qc)
            np.testing.assert_allclose(x.grad.cpu().numpy(), b.T@cr['gradient'], atol=1e-6, rtol=1e-8)
            direction = np.random.default_rng(2026106502).normal(size=xc.shape)
            direction /= np.linalg.norm(direction)
            d = device.cast(direction)
            with torch.no_grad():
                plus = device.evaluate(bt@(x+1e-6*d)+ct[:, None]*start)['energy']
                minus = device.evaluate(bt@(x-1e-6*d)+ct[:, None]*start)['energy']
            fd = float((plus-minus)/2e-6)
            analytic = float((x.grad*d).sum())
            assert abs(fd-analytic) <= 1e-4+1e-5*abs(fd)
            assert float(rr['risk_sum']) > .05
            REPORT['active_frozen_spline_gradient'] = {'pass': True, 'fd_directional': fd,
                'analytic_directional': analytic, 'relative_error': abs(fd-analytic)/max(1., abs(fd)),
                'CPU_GPU_control_gradient_max_abs': float(abs(x.grad.cpu().numpy()-b.T@cr['gradient']).max()),
                'initial_state_derivative_max_abs': float(abs(b[0]).max()),
                'frozen_model_rebuilt': False}
            del rc, rr, cr, x, qx, plus, minus
        REPORT['full_grid_fixtures'].append(row); save()
        del q, r, device, model; gc.collect(); torch.cuda.empty_cache()
    REPORT['all_healthy_fixtures_unusable'] = all(not v['within_surrogate_budget'] or not v['valid'] for v in REPORT['full_grid_fixtures'])
    if REPORT['all_healthy_fixtures_unusable']:
        REPORT['terminal_gate'] = 'failure: all four old healthy fixtures unusable under unchanged surrogate'
        return
    # Explicit full eight steps for timing, not enforcing the 2s target secretly.
    request_start = time.perf_counter(); traces = []; output = {}
    for slot in (0, 1):
        x = torch.as_tensor(controls[slot], device='cuda', dtype=torch.float64).clone().requires_grad_(True)
        start = torch.as_tensor(paths[slot, 0], device='cuda', dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=.003)
        best = None; best_key = None; trace = []
        for iteration in range(8):
            guard(); begin = time.perf_counter()
            optimizer.zero_grad(); q = bt@x+ct[:, None]*start
            model = cpu.FrozenEnergy(arm, q.detach().cpu().numpy(), targets[slot])
            device = gpu.GPUFrozen(model)
            r = device.evaluate(q)
            if not all(torch.isfinite(r[k]) for k in ('energy', 'cost', 'risk_sum')):
                raise ValueError('nonfinite old-fixture optimizer')
            values = gpu.numeric(r)
            key = (0, values['cost']) if values['risk_sum'] <= .05 else (1, values['risk_sum'])
            if best_key is None or key < best_key:
                best_key = key; best = x.detach().clone(); returned = iteration
            r['energy'].backward()
            norm = float(torch.nn.utils.clip_grad_norm_([x], 20.))
            optimizer.step(); torch.cuda.synchronize()
            trace.append({'iteration': iteration, 'seconds': time.perf_counter()-begin,
                          'gradient_norm_before_cap': norm, 'values': values})
            del q, r, model, device; gc.collect()
        selected = (bt@best+ct[:, None]*start).cpu().numpy()
        np.testing.assert_allclose(selected[0], paths[slot, 0], atol=1e-14)
        output[f'slot{slot}_q'] = selected
        output[f'slot{slot}_x'] = best.cpu().numpy()
        traces.append({'slot': slot, 'iterations': len(trace), 'selected_iteration': returned, 'trace': trace})
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-request_start
    REPORT['eight_step_timing'] = {'two_distinct_old_fixture_slots': True,
        'each_own_start_and_goal': True, 'request_seconds': elapsed,
        'target_seconds': 2., 'within_timing_gate': elapsed <= 2., 'slots': traces,
        'no_independent_risk_claim_for_corrected_paths': True}
    np.savez_compressed(P/'OLD_TIMING_PATHS.npz', **output)
    REPORT['terminal_gate'] = 'pass: old qualification gates' if elapsed <= 2. else 'failure: unchanged prototype exceeds 2s two-candidate eight-step correction gate'


if __name__ == '__main__':
    try:
        run()
        REPORT['numeric_checks_pass'] = True
    except Exception as exc:
        import traceback
        REPORT.update(numeric_checks_pass=False, terminal_gate='failure: qualification exception', error=repr(exc), traceback=traceback.format_exc())
    finally:
        save()
    print(json.dumps({k: v for k, v in REPORT.items() if k not in ('full_grid_fixtures', 'eight_step_timing')}, indent=2))
    sys.exit(0 if REPORT['numeric_checks_pass'] else 1)
