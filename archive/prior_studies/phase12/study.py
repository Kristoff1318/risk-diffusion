"""Exactly one frozen four-task confirmation; original fully batched validator."""
import time
START = time.perf_counter()
import pathlib, sys, json, hashlib, types, resource, traceback
import numpy as np
import torch
from scipy.stats import beta
P = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(P.parent/'phase7'), str(P.parent/'phase11'), str(P.parent/'phase10')]
import core
import native_energy
import gpu_energy
import initial_joint_floor

CFG = json.loads((P/'PROTOCOL.json').read_text())
ROWS = [{'task': t, 'official_ID': CFG['IDs'][t], 'arm': a, 'status': 'not_started',
         'accepted': False, 'selected': None, 'candidates': []}
        for t in range(4) for a in CFG['arms']]
REPORT = {'terminal': False, 'complete': False, 'scope': 'fresh frozen4-task5% comparison',
          'rows': ROWS, 'initialization': {}, 'tasks': [], 'prepared': [], 'physics': CFG['physics'],
          'new_training': 0, 'validator': 'unchanged phase7 core.audit; no adaptive/lazy adapter'}
PATHS = {}


class BudgetStop(RuntimeError):
    pass


def guard(seconds=1):
    if time.perf_counter()-START+seconds > CFG['internal_guard_seconds']:
        raise BudgetStop('frozen remaining operation would exceed internal guard')


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def save():
    REPORT['process_seconds'] = time.perf_counter()-START
    REPORT['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    dump(P/'MAIN_RESULTS.json', REPORT)
    if PATHS:
        np.savez_compressed(P/'PATHS.npz', **PATHS)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pack(name, flags):
    np.savez_compressed(P/'outcomes'/name, flags=np.packbits(flags, axis=0), n=len(flags))


def stats(flags):
    k = int(flags.any(1).sum())
    return {'n': len(flags), 'union': k, 'risk': k/len(flags), 'CI95': core.p0.interval(k, len(flags)),
            'counts': dict(zip(core.p0.EVENTS, flags.sum(0).astype(int).tolist()))}


def upper(k, n):
    return float(beta.ppf(.975, k+1, n-k)) if k < n else 1.


def surrogate(context, nom, target):
    with torch.no_grad():
        model = context.build(nom, target)
        if not model.valid:
            return {'valid': False, 'status': 'singular local model; cannot optimize', 'events': 4323770}
        result = gpu_energy.numeric(model.evaluate(nom))
        result['valid'] = True
        return result


def correct(c, context, inputs, start, target, target_np, method):
    begun = time.perf_counter(); logs = []; values = []
    zeros = (c.cast(np.zeros((1, 2001, 7))), c.cast(np.zeros((1, 200, 7))), c.cast(np.zeros((1, 200, 7))))
    for slot in range(2):
        remaining = max(0., 2-(time.perf_counter()-begun))
        if remaining <= 0:
            values.append(inputs[slot].detach()); logs.append({'slot': slot, 'iterations': 0, 'timeout': True, 'status': 'request_ceiling_reached'})
            continue
        if method == 'dense':
            xx, log = core.refine(c, inputs[slot], start, target, steps=8, lr=.003, delta=1., err=zeros, wallcap=remaining)
            log['slot'] = slot; logs.append(log); values.append(xx)
            continue
        x = inputs[slot].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=.003); trace = []; best = None; best_key = None; timeout = False; selected = None
        for it in range(8):
            guard(.1)
            if time.perf_counter()-begun >= 2:
                timeout = True; break
            opt.zero_grad(); nom = c.nom(x, start)
            model = context.build(nom, target_np)
            if not model.valid:
                trace.append({'iteration': it, 'status': 'singular_local_model'}); break
            r = model.evaluate(nom)
            ev, jv, rv = torch.stack([r[k].detach() for k in ('energy', 'cost', 'risk_sum')]).cpu().tolist()
            if not np.isfinite([ev, jv, rv]).all():
                trace.append({'iteration': it, 'status': 'nonfinite'}); break
            key = (0, jv) if rv <= .05 else (1, rv)
            if best_key is None or key < best_key:
                best = x.detach().clone(); best_key = key; selected = it
            r['energy'].backward(); norm = float(torch.nn.utils.clip_grad_norm_([x], 20.))
            opt.step(); torch.cuda.synchronize()
            trace.append({'iteration': it, 'energy': ev, 'cost': jv, 'risk_sum': rv,
                          'gradient_norm_before_cap': norm, 'request_seconds': time.perf_counter()-begun, 'events': r['events']})
        values.append(best if best is not None else inputs[slot].detach())
        logs.append({'slot': slot, 'iterations': len([v for v in trace if 'energy' in v]),
                     'trace': trace, 'returned_iteration': selected, 'timeout': timeout,
                     'selection_rule': 'lowestJ withR<=.05 else lowestR; completed evaluated iterates'})
    torch.cuda.synchronize(); elapsed = time.perf_counter()-begun
    return torch.stack(values), {'seconds': elapsed, 'ceiling_seconds': 2., 'ceiling_overrun': elapsed > 2,
                                'slots': logs, 'steps_compute_not_assumed_equal': True}


def run():
    torch.set_num_threads(2)
    for name, h in CFG['source_hashes'].items():
        assert sha(P.parent/name) == h, name
    base = core.p0.BASE; index = json.loads((base/'assets/archive-extraction.json').read_text())
    dataset = base/'assets'/index['target_dataset_prefix']/CFG['dataset_filename']
    indices_file = base/'assets'/index['target_model_prefix']/'val_subset_indices.pt'
    for path in [dataset, indices_file]:
        guard(8); assert sha(path) == CFG['asset_hashes'][str(path)], str(path)
    before = time.perf_counter()
    pm = types.SimpleNamespace(Unpickler=core.p0.RestrictedCPU, __name__='restricted')
    indices = torch.load(indices_file, map_location='cpu', pickle_module=pm)
    ids = [int(indices[i]) for i in CFG['positions']]
    assert ids == CFG['IDs']
    with open(dataset, 'rb') as stream:
        fields = core.p0.RestrictedCPU(stream).load()['fields']
    keys = ['q_start', 'q_goal', 'qs', 'ee_goal_pose', 'ee_goal_orientation', 'ee_goal_position', 'control_points']
    data = {k: fields[k][ids].clone().cpu() for k in keys}; del fields
    REPORT['initialization']['data_load_and_select_seconds'] = time.perf_counter()-before
    np.savez_compressed(P/'TASKS.npz', **{k: v.numpy() for k, v in data.items()}, official_ids=np.array(ids), positions=np.array(CFG['positions']))
    guard(2); before = time.perf_counter(); c = core.Core(); torch.cuda.synchronize()
    REPORT['initialization']['core_cuda_assets_seconds'] = time.perf_counter()-before
    floors = initial_joint_floor.diagnose({'tasks': [{'official_ID': ids[i], 'q_start': data['q_start'][i].tolist()} for i in range(4)],
        'joint_lower': c.arm.lo.tolist(), 'joint_upper': c.arm.hi.tolist(), 'initial_error_std_rad': .004, 'risk_budgets': [.05]})
    dump(P/'INITIAL_JOINT_FLOORS.json', floors)
    eligible = []
    for t in range(4):
        guard(1); st = c.cast(data['q_start'][t]); goal = c.cast(data['q_goal'][t]); target = c.cast(data['ee_goal_pose'][t])
        ref = c.nom(c.cast(data['control_points'][t]), st)
        nc = core.nominal_check(c, ref, target)
        with torch.no_grad():
            endpoints = c.margins_tensor(torch.stack((st, goal))).cpu().numpy()
        floor = floors['rows'][t]['initial_joint_event_probability']
        valid = bool(nc['finite'] and nc['union'] == 0 and (endpoints >= 0).all())
        allowed = valid and floor <= .05
        REPORT['tasks'].append({'task': t, 'official_ID': ids[t], 'position': CFG['positions'][t],
            'initial_joint_floor': floor, 'ruled_out_by_initial_event': floor > .05,
            'descriptor_valid': valid, 'reference_nominal': nc, 'endpoint_margins': endpoints.tolist(), 'eligible': allowed})
        if allowed:
            eligible.append(t)
        else:
            for row in ROWS:
                if row['task'] == t:
                    row['status'] = 'initial_risk_infeasible' if floor > .05 else 'invalid_native_descriptor'
    save()
    if len(eligible) < 3:
        REPORT['stop_reason'] = 'frozen descriptor/initial-event health gate: fewer than3 eligible of4'; return
    guard(2); before = time.perf_counter(); model, norm, context_fn, meta = core.load_mpd(); torch.cuda.synchronize()
    REPORT['initialization']['MPD_model_seconds'] = time.perf_counter()-before
    queries = {k: v.to('cuda') for k, v in data.items()}
    before = time.perf_counter(); native = native_energy.NativeContext(c.arm, 'cuda:0'); torch.cuda.synchronize()
    REPORT['initialization']['native_invariant_cache_seconds'] = time.perf_counter()-before
    prepared = {}
    for t in eligible:
        st = c.cast(data['q_start'][t]); target = c.cast(data['ee_goal_pose'][t]); target_np = data['ee_goal_pose'][t].numpy()
        seed = 2026106000+t*100
        guard(7); before = time.perf_counter(); xx, gen = core.classical(c, st, target, seed, steps=80); torch.cuda.synchronize()
        prep = {'task': t, 'family': 'classical', 'seed': seed, 'seconds': time.perf_counter()-before, 'generation_and_nominal': gen, 'nominal_checks': []}
        prepared[t, 'classical'] = xx
        for slot in range(2):
            q = c.nom(xx[slot], st); nc = core.nominal_check(c, q, target); prep['nominal_checks'].append(nc)
            PATHS[f't{t}_classical_prepared_c{slot}_x'] = xx[slot].cpu().numpy(); PATHS[f't{t}_classical_prepared_c{slot}_q'] = q.detach().cpu().numpy()
        REPORT['prepared'].append(prep); save()
        guard(5); before = time.perf_counter(); raw, gen = core.mpd(model, norm, context_fn, queries, t, seed)
        xx = []; logs = []
        for slot in range(2):
            guard(2); value, log = core.refine(c, raw[slot], st, target, steps=80, lr=.025, wallcap=15)
            xx.append(value); logs.append(log)
            PATHS[f't{t}_mpd_raw_c{slot}_x'] = raw[slot].cpu().numpy()
        torch.cuda.synchronize(); xx = torch.stack(xx)
        prep = {'task': t, 'family': 'mpd', 'seed': seed, 'seconds': time.perf_counter()-before,
                'generation': gen, 'nominal_refinements': logs, 'nominal_checks': []}
        prepared[t, 'mpd'] = xx
        for slot in range(2):
            q = c.nom(xx[slot], st); nc = core.nominal_check(c, q, target); prep['nominal_checks'].append(nc)
            PATHS[f't{t}_mpd_prepared_c{slot}_x'] = xx[slot].cpu().numpy(); PATHS[f't{t}_mpd_prepared_c{slot}_q'] = q.detach().cpu().numpy()
        REPORT['prepared'].append(prep); save()
    health = {f: sum(any(n['finite'] and n['union'] == 0 for n in p['nominal_checks']) for p in REPORT['prepared'] if p['family'] == f) for f in ['classical', 'mpd']}
    REPORT['prepared_health_tasks_of4'] = health
    if min(health.values()) < 3:
        REPORT['stop_reason'] = 'frozen nominal-health gate: either source healthy on fewer than3 of4 tasks'
        for r in ROWS:
            if r['status'] == 'not_started': r['status'] = 'not_run_nominal_health_gate'
        return
    # Reconcile the still-frozen counts before starting costly validation.
    remaining_forecast = len(eligible)/4*(203.84+4.37+4.)
    REPORT['remaining_validation_forecast_seconds'] = remaining_forecast
    guard(remaining_forecast)
    for t in eligible:
        st = c.cast(data['q_start'][t]); target = c.cast(data['ee_goal_pose'][t]); target_np = data['ee_goal_pose'][t].numpy()
        before = time.perf_counter(); guard(2)
        ad = core.errors(1024, 2026106200+t, 2); ad_draw_seconds = time.perf_counter()-before
        before = time.perf_counter(); au = core.errors(4096, 2026106300+t, 2); au_draw_seconds = time.perf_counter()-before
        before = time.perf_counter(); pd = core.errors(2048, 2026106400+t, 2); pd_draw_seconds = time.perf_counter()-before
        for family in ('classical', 'mpd'):
            prep = next(p for p in REPORT['prepared'] if p['task'] == t and p['family'] == family)
            prep['diagnostics'] = []; prep['prepared_draw_seconds_shared_task'] = pd_draw_seconds
            for slot in range(2):
                guard(3.2); q = c.nom(prepared[t, family][slot], st)
                rr, flags = core.audit(c, q, target, pd, 16)
                rr['cost'] = float(c.cost(q)); rr['surrogate'] = surrogate(native, q, target_np)
                prep['diagnostics'].append(rr); pack(f't{t}_{family}_prepared_c{slot}_diagnostic.npz', flags)
            save()
        order = CFG['arms'][t%4:]+CFG['arms'][:t%4]
        for arm in order:
            family, method = arm.split('_'); row = next(r for r in ROWS if r['task'] == t and r['arm'] == arm)
            row['status'] = 'started'; row['delta'] = .05
            guard(2.2); before = time.perf_counter()
            xx, correction = correct(c, native, prepared[t, family], st, target, target_np, method)
            row['correction'] = correction; row['correction_seconds'] = time.perf_counter()-before
            row['candidate_surrogates'] = []; paths = []
            before = time.perf_counter(); admitted = []
            for slot in range(2):
                guard(1.7); q = c.nom(xx[slot], st); paths.append(q)
                nc = core.nominal_check(c, q, target)
                ac, flags = core.audit(c, q, target, ad, 16)
                ac['selection_upper_alpha025'] = upper(ac['union'], ac['n'])
                ac['nominal_command_valid'] = bool((q.diff(dim=0).abs()/.05 <= c.vmax).all())
                ac['cost'] = float(c.cost(q)); ac['admitted'] = bool(ac['finite'] and ac['nominal_command_valid'] and ac['selection_upper_alpha025'] <= .05)
                ac['path_hash'] = hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest()
                row['candidates'].append({'slot': slot, 'nominal': nc, 'admission': ac})
                pack(f't{t}_{arm}_c{slot}_admission.npz', flags)
                if ac['admitted']: admitted.append(slot)
                PATHS[f't{t}_{arm}_c{slot}_q'] = q.detach().cpu().numpy(); PATHS[f't{t}_{arm}_c{slot}_x'] = xx[slot].cpu().numpy()
            row['admission_seconds'] = time.perf_counter()-before
            row['selected'] = min(admitted, key=lambda s: row['candidates'][s]['admission']['cost']) if admitted else None
            row['accepted'] = bool(admitted)
            prep = next(p for p in REPORT['prepared'] if p['task'] == t and p['family'] == family)
            row['fresh_shared_source_preparation_seconds'] = prep['seconds']
            row['admission_draw_seconds_shared_task'] = ad_draw_seconds
            row['audit_draw_seconds_shared_task'] = au_draw_seconds
            row['standalone_warm_request_seconds'] = prep['seconds']+row['correction_seconds']+row['admission_seconds']+ad_draw_seconds
            row['actual_shared_generation_count_this_arm'] = 0
            row['source_generation_count_once_per_family_task'] = 1
            row['hypothetical_cold_compute_request_seconds'] = row['standalone_warm_request_seconds']+REPORT['initialization']['core_cuda_assets_seconds']+(REPORT['initialization']['MPD_model_seconds'] if family == 'mpd' else 0)+(REPORT['initialization']['native_invariant_cache_seconds'] if method == 'native' else 0)
            save()
            before = time.perf_counter()
            for slot, q in enumerate(paths):
                count = 4096 if slot == row['selected'] else 2048
                guard(5.8 if count == 4096 else 3.2)
                ar, flags = core.audit(c, q, target, tuple(v[:count] for v in au), 16)
                ar['prefix2048'] = stats(flags[:2048]); ar['cost'] = float(c.cost(q)); ar['cache_reused'] = False
                ar['surrogate'] = surrogate(native, q, target_np)
                row['candidates'][slot]['audit'] = ar
                pack(f't{t}_{arm}_c{slot}_audit.npz', flags)
                save()
            row['audit_seconds'] = time.perf_counter()-before; row['status'] = 'completed'
            dump(P/'outcomes'/f't{t}_{arm}.json', row); save()
            print(json.dumps({'task': t, 'ID': CFG['IDs'][t], 'arm': arm, 'accepted': row['accepted'],
                'correction_seconds': row['correction_seconds'], 'admission_seconds': row['admission_seconds'],
                'audit_seconds': row['audit_seconds'], 'process_seconds': time.perf_counter()-START}), flush=True)
        del ad, au, pd
    REPORT['complete'] = all(r['status'] in ('completed', 'initial_risk_infeasible', 'invalid_native_descriptor') for r in ROWS)


if __name__ == '__main__':
    (P/'outcomes').mkdir(exist_ok=False)
    try:
        run()
    except Exception as exc:
        REPORT.update(stop_reason='resource_guard' if isinstance(exc, BudgetStop) else 'exception', error=repr(exc), traceback=traceback.format_exc())
        for r in ROWS:
            if r['status'] == 'not_started': r['status'] = 'not_run_'+REPORT['stop_reason']
    finally:
        REPORT['terminal'] = True; save()
    print(json.dumps({k: v for k, v in REPORT.items() if k not in ('rows', 'tasks', 'prepared')}, indent=2), flush=True)
