"""Replay saved commands to independently check execution and active-contact flags."""
import argparse
import json
from pathlib import Path

from nominal_baseline import ROOT, load_environment, np, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='results/nominal_baseline_20261104')
    args = parser.parse_args()
    run = ROOT / args.run
    protocol = json.loads((run / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert sha(path) == expected, f'Source changed: {path}'
    env = load_environment('maze2d-large-v1')
    results = json.loads((run / 'results.json').read_text())
    checks = []
    for row in results:
        data = np.load(run / f'episode_{row["seed"]}.npz')
        env.seed(row['seed'])
        start = env.reset()
        assert np.array_equal(start, row['start'])
        states, flags = [start.copy()], []
        stale_wall_steps, max_command_error = 0, 0.
        for t, recorded in enumerate(data['commands']):
            state = env.state_vector().copy()
            ref = data['reference'][min(t + 1, len(data['reference']) - 1)].copy()
            if t >= len(data['reference']) - 1:
                ref[2:] = 0.
            expected = ref[:2] - state[:2] + (ref[2:] - state[2:])
            max_command_error = max(max_command_error, float(np.max(np.abs(expected - recorded))))
            obs, _, _, _ = env.step(recorded)
            states.append(obs.copy())
            def is_wall(c):
                names = [env.model.geom_id2name(int(g)) or '' for g in (c.geom1, c.geom2)]
                return c.dist <= .002 and any(n.startswith('wall_') for n in names)
            active = any(is_wall(env.sim.data.contact[i]) for i in range(env.sim.data.ncon))
            full_buffer = any(is_wall(c) for c in env.sim.data.contact)
            stale_wall_steps += int(full_buffer and not active)
            flags.append(active)
        error = float(np.max(np.abs(np.asarray(states) - data['execution'])))
        assert error == 0., error
        assert max_command_error == 0., max_command_error
        assert np.array_equal(flags, data['wall_contact_flags'])
        assert sum(flags) == row['wall_contact_steps']
        checks.append(dict(seed=row['seed'], trajectory_max_error=error,
                           controller_max_error=max_command_error,
                           active_wall_steps=sum(flags), stale_buffer_extra_steps=stale_wall_steps))
    assert any(r['stale_buffer_extra_steps'] > 0 for r in checks)
    report = dict(passed=True, protocol_source_hashes_unchanged=True,
                  exact_original_controller_replay=True,
                  exact_native_trajectory_replay=True, active_contacts_match=True,
                  stale_buffer_regression_demonstrated=True, episodes=checks)
    (run / 'replay_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == '__main__':
    main()
