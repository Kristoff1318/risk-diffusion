"""Check a relocated copy with access to the old project and archives blocked."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def worker():
    root = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if 'rb-diffuser' not in p]
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    def deny_external_project(event, args):
        if event in ('open', 'os.listdir', 'os.scandir', 'os.chdir') and args:
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                p = os.fsdecode(p)
                if 'rb-diffuser' in p or '/archive/' in p:
                    raise RuntimeError(f'External project/archive access blocked: {p}')

    sys.addaudithook(deny_external_project)
    from nominal_baseline import load_policy, sample, execute, np, torch, load_environment
    import diffuser
    from chance_guidance import TrackingChanceModel
    torch.set_num_threads(2)
    run = root / 'results/nominal_baseline'
    row = json.loads((run / 'results.json').read_text())[0]
    saved = np.load(run / 'episode_0.npz')
    env = load_environment('maze2d-large-v1')
    env.seed(0)
    start = env.reset()
    goal = np.asarray(env.get_target(), dtype=float)
    policy = load_policy()
    paths, _ = sample(policy, start, goal, 0)
    assert np.array_equal(paths.observations[0], saved['reference'])
    assert np.array_equal(paths.actions[0], saved['generated_actions'])
    metrics, arrays = execute(env, paths.observations[0], start, goal)
    assert np.array_equal(arrays['execution'], saved['execution'])
    assert np.array_equal(arrays['wall_contact_flags'], saved['wall_contact_flags'])
    assert metrics['goal_reached'] == row['goal_reached']
    model = TrackingChanceModel(env, np.load(root / 'assets/maze2d-large-v1/normalization.npz'),
                                execution_steps=64)
    physical = torch.tensor([[[.402, 1., 0., 0.]]*65], dtype=torch.float64)
    normalized = 2.*(physical-model.observation_minimum)/model.observation_range-1.
    z = torch.cat([torch.zeros((1,65,2),dtype=torch.float64),normalized],-1).requires_grad_(True)
    start_tensor = torch.tensor([[.402, 1., 0., 0.]], dtype=torch.float64)
    goal_tensor = torch.tensor([[.902, 1.]], dtype=torch.float64)
    energy = model.energy(z, start_tensor, goal_tensor, .05).sum()
    gradient = torch.autograd.grad(energy, z)[0][0, 1, 2].item()
    # The wall's small propagated standard deviation makes its CDF steep.
    # Use a small float64 perturbation to resolve the local derivative.
    step = 1.e-7
    plus, minus = z.detach().clone(), z.detach().clone()
    plus[0, 1, 2] += step
    minus[0, 1, 2] -= step
    finite_difference = ((model.energy(plus, start_tensor, goal_tensor, .05) -
                          model.energy(minus, start_tensor, goal_tensor, .05)) / (2 * step)).item()
    error = abs(gradient - finite_difference) / max(abs(gradient), abs(finite_difference), 1.e-12)
    assert abs(gradient) > 1.e-8 and error < 1.e-5, (gradient, finite_difference, error)
    env.close()
    result = dict(passed=True, relocated_copy=True, no_old_project_or_archive_access=True,
                  local_diffuser_path=str(Path(diffuser.__file__).relative_to(root)),
                  seed0_reference_exact=True, seed0_actions_exact=True,
                  seed0_execution_exact=True, seed0_contacts_exact=True,
                  chance_energy_gradient_relative_error=error,
                  chance_model_scope='same state-tracking controller as nominal; native risk audits reported separately')
    (root / 'portability_worker.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='maze2d-portability-') as temp:
        copy = Path(temp) / 'project'
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns(
            '.git', 'archive', '__pycache__', '*.egg-info', '.pytest_cache'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
        completed = subprocess.run([sys.executable, '-I', str(copy / 'verify_portability.py'),
                                    '--worker'], cwd=temp, env=env, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        result = json.loads((copy / 'portability_worker.json').read_text())
    out = root / 'results/nominal_baseline/portability_verification.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
