"""CPU synthesis of saved gate outcome; no FK, inference or new sampling."""
import csv, hashlib, json, pathlib
P = pathlib.Path(__file__).resolve().parent
r = json.loads((P/'MAIN_RESULTS.json').read_text())
cfg = json.loads((P/'PROTOCOL.json').read_text())
rows = []
for p in r['prepared']:
    for slot, n in enumerate(p['nominal_checks']):
        rows.append({'official_ID': cfg['IDs'][p['task']], 'task': p['task'], 'family': p['family'],
            'slot': slot, 'nominal_valid': n['finite'] and n['union'] == 0, 'cost_J': n['cost'],
            'failure_components': ';'.join(k for k, v in n['counts'].items() if v),
            'position_m': n['max_position'], 'orientation_rad': n['max_orientation'],
            'joint_margin_rad': n['minimum_margins'][4], 'world_mpd_margin_m': n['minimum_margins'][0],
            'world_curobo_margin_m': n['minimum_margins'][1], 'self_mpd_margin_m': n['minimum_margins'][2],
            'self_curobo_margin_m': n['minimum_margins'][3], 'source_generation_and_nominal_seconds': p['seconds'],
            'admission_executions': 0, 'audit_executions': 0, 'stochastic_risk_measured': False})
with (P/'PREPARED_PROPOSALS.csv').open('w') as stream:
    w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
tasks = []
for t in r['tasks']:
    item = {'task': t['task'], 'official_ID': t['official_ID'], 'position': t['position'],
        'descriptor_valid': t['descriptor_valid'], 'initial_joint_floor': t['initial_joint_floor'],
        'descriptor_failure_components': ';'.join(k for k, v in t['reference_nominal']['counts'].items() if v),
        'classical_valid_slots': sum(a['nominal_valid'] for a in rows if a['task'] == t['task'] and a['family'] == 'classical'),
        'MPD_valid_slots': sum(a['nominal_valid'] for a in rows if a['task'] == t['task'] and a['family'] == 'mpd'),
        'risk_admission_status': 'NOT_RUN_NOMINAL_HEALTH_GATE' if t['eligible'] else 'NOT_RUN_INVALID_DESCRIPTOR',
        'stochastic_admission_executions': 0, 'stochastic_audit_executions': 0,
        'prepared_diagnostic_executions': 0, 'correction_steps': 0}
    tasks.append(item)
with (P/'PER_TASK_STATUS.csv').open('w') as stream:
    w = csv.DictWriter(stream, fieldnames=list(tasks[0]), lineterminator='\n'); w.writeheader(); w.writerows(tasks)
assert r['stop_reason'].startswith('frozen nominal-health gate')
assert not any(v['candidates'] for v in r['rows'])
assert len(rows) == 12
assert all(n['iterations'] == 80 for p in r['prepared'] for n in p.get('nominal_refinements', p.get('generation_and_nominal', {}).get('nominal_refinements', [])))
initial = r['initialization']
md = ['# Frozen fresh confirmation: terminal nominal-health gate failure', '',
'The authorized study stopped at its prespecified nominal-health gate. Classical supplied a healthy prepared candidate on2/4 fixed tasks; MPD on3/4. Both sources were required to reach3/4. No task replacement, extra IK seed, extra nominal step, correction or stochastic validation followed. GPU slot released.', '',
'This is a completed gate decision, not a completed risk-method comparison. MAIN_RESULTS.complete=false records that the16 planned method requests did not run. Their accepted=false placeholders MUST NOT be read as stochastic rejections. The initial analytic joint floor rules out none of the four tasks; machine-underflow zeros are not exact zero risk.', '',
'## Fixed cohort and observed preparation', '',
'|Task ID / position|Descriptor|Initial joint-event probability|Classical healthy slots|MPD healthy slots|Terminal method status|',
'|---|---|---:|---:|---:|---|']
for t in tasks:
    floor_label = f"{t['initial_joint_floor']:.6g}" if t['initial_joint_floor'] else 'numeric underflow'
    classical_label = f"{t['classical_valid_slots']}/2" if t['descriptor_valid'] else 'not generated'
    mpd_label = f"{t['MPD_valid_slots']}/2" if t['descriptor_valid'] else 'not generated'
    md.append(f"|{t['official_ID']} / {t['position']}|{'valid' if t['descriptor_valid'] else t['descriptor_failure_components']}|{floor_label}|{classical_label}|{mpd_label}|{t['risk_admission_status']}|")
md += ['', 'Task175075 has a native cuRobo self-collision already at its nominal start/reference (minimum clearance-3.522mm). It fails the frozen descriptor validity gate. This is NOT a computed Gaussian collision-risk lower bound. The other three descriptors pass; none was substituted.', '',
'Every generated slot is retained below, including failures. J is nominal effort; nominal validity is a deterministic zero-error hard-event check, not a chance constraint or admission. Generic risk/CI fields in raw nominal_check results come from a one-execution deterministic validator and are NOT Monte Carlo risk estimates.', '',
'|Task|Source / slot|Nominal valid|J|Failing hard components|Source two-slot generation +80-step preparation, seconds|',
'|---|---|---|---:|---|---:|']
for a in rows:
    md.append(f"|{a['official_ID']}|{a['family']} / {a['slot']}|{a['nominal_valid']}|{a['cost_J']:.6f}|{a['failure_components'] or 'none'}|{a['source_generation_and_nominal_seconds']:.6f}|")
md += ['', 'The source time is shared by its two slots and two intended variants; repeated table entries are not extra executed time. All12 nominal refinements completed80 steps (960 gradient calls). Classical ran three fresh batched80-step full-pose IK procedures with two seeds each. MPD ran three fresh two-sample DDIM15 requests (45 denoiser calls) and six nominal refinements. The six MPD and two classical paths passed the nominal hard checks; all other classical slots remain saved.', '',
'## Actual stochastic counts and comparison limits', '',
'Admission:0 executions; corrected unselected audits:0; selected audits:0; prepared-input stochastic diagnostics:0; correction steps:0 for either method. No optimizer risk scenarios or risk draw streams were generated. There were16 deterministic full-event checks (four references plus12 prepared slots) and separate native endpoint checks.', '',
'Thus per-task stochastic acceptance, audited risk/CI, risk-correction cost/latency, surrogate calibration and dense-versus-native gain/loss are NOT MEASURED. No native energy was evaluated on fresh paths after the gate failed. No historical CVaR timing superiority or nonlinear safety claim follows.', '',
'The evidence concerns nominal initialization health: repaired MPD supplies six valid slots on the three eligible cases, whereas classical supplies two. This is a small fixed-cohort observation with an invalid fourth descriptor, not a successful risk-aware planning confirmation. It does not establish no risk headroom; that question was not reached.', '',
'## Timing and resources', '',
f"Source metadata load/select:{initial['data_load_and_select_seconds']:.6f}s; Core/CUDA/native assets initialization:{initial['core_cuda_assets_seconds']:.6f}s; genuine MPD model initialization:{initial['MPD_model_seconds']:.6f}s; native invariant cache:{initial['native_invariant_cache_seconds']:.6f}s.", '',
f"Fresh generation + nominal preparation total:{sum(p['seconds'] for p in r['prepared']):.6f}s. Per-source warm timings are in the table; no end-to-end risk request was completed, so no cold/admission request latency is manufactured. Initializations and offline1.77GB metadata load are separately charged.", '',
f"Runner process:{r['process_seconds']:.6f}s; timed command18.42s. Peak allocated GPU{r['peak_GPU_allocated_bytes']:,}bytes; RSS{r['peak_RSS_KiB']:,}KiB. No timeout, nonfinite or iteration shortening triggered the gate; the failure is nominal health. The original fully batched validator was used unchanged, no adaptive adapter.", '',
'Pre-outcome full-plan forecast290.21s fitted the309s remainder with unchanged sample counts and39s report/commit reserve. The prescribed gate stopped early; no counts were reduced to fit outcomes. Additional booking57.42s =18.42s measured command +39s estimated ancillary reserve (includes protocol/code/hash/report/preservation/commit work). Cumulative198.42/450s; remaining251.58s. Reserve is conservative bookkeeping, not a measured ancillary total.', '',
'Protocol committed7d53254 before outcomes; original asset/code hashes are checked and preserved. Original/default code, prior results and phase10/phase11 specifications remain unchanged. New sources/paths/tasks/nominal traces are preserved inphase12; no copied external assets.', '',
'## Decision', '',
'Stop this variant at the frozen gate. Do not replace175075, repair classical with extra steps/seeds, weaken model events, or run a surviving MPD-only comparison under this protocol. Any different study requires a new scientific question and explicit authorization. The covariance-energy risk benefit remains unconfirmed on fresh tasks. GPU slot released.']
(P/'RESULTS.md').write_text('\n'.join(md)+'\n')
status = {'terminal': True, 'stage_completed': 'initial/descriptor checks and nominal preparation',
    'terminal_gate': 'FAIL_NOMINAL_HEALTH', 'comparison_executed': False, 'planned_requests': 16,
    'completed_risk_requests': 0, 'fixed_tasks': 4, 'descriptor_valid_tasks': 3,
    'classical_healthy_tasks_of4': 2, 'MPD_healthy_tasks_of4': 3,
    'all_tasks_retained': True, 'risk_sample_counts_actual': {'admission': 0, 'unselected_audit': 0, 'selected_audit': 0, 'prepared_diagnostic': 0},
    'fresh_model_generation_requests': 3, 'fresh_MPD_samples': 6, 'denoiser_calls': 45,
    'classical_IK_seed_trajectories': 6, 'nominal_gradient_calls': 960,
    'GPU_slot_released': True, 'no_automatic_follow_on': True}
dump = lambda f, v: f.write_text(json.dumps(v, indent=2)+'\n')
dump(P/'STATUS.json', status)
ledger = {'prior_cumulative_booked_seconds': 141, 'remaining_authorized_seconds_before_run': 309,
    'forecast_main_seconds': 251.21, 'forecast_total_additional_seconds': 290.21,
    'hard_outer_timeout_seconds': 270, 'internal_guard_seconds': 267,
    'actual_timed_command_wall_seconds': 18.42, 'estimated_ancillary_reserve_seconds': 39,
    'ancillary_reserve_is_estimate_not_measured': True, 'additional_booked_seconds': 57.42,
    'cumulative_booked_seconds': 198.42, 'total_cap_seconds': 450, 'remaining_seconds': 251.58,
    'artifact_total_cap_bytes': 50000000, 'phase11_bytes': sum(f.stat().st_size for f in (P.parent/'phase11').iterdir() if f.is_file()),
    'phase12_bytes_before_final_ledger': sum(f.stat().st_size for f in P.rglob('*') if f.is_file()),
    'sample_counts_reduced_after_outcomes': False, 'stopped_on_prespecified_gate': True,
    'GPU_slot_released': True, 'protocol_commit_before_outcomes': '7d53254'}
dump(P/'RESOURCE_LEDGER.json', ledger)
print(json.dumps(status, indent=2)); print(json.dumps(ledger, indent=2))
