"""Load only repository-local Diffuser code and pretrained assets."""
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / 'src'
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
import diffuser
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.guides.policies import Policy
from diffuser.models import GaussianDiffusion, TemporalUnet

ASSETS = PROJECT / 'assets/maze2d-large-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy():
    module = Path(diffuser.__file__).resolve()
    if SOURCE.resolve() not in module.parents:
        raise RuntimeError(f'Expected repository-local Diffuser, got {module}')
    manifest = json.loads((ASSETS / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(ASSETS / name) != expected:
            raise ValueError(f'Local pretrained asset hash mismatch: {name}')
    config = json.loads((ASSETS / 'config.json').read_text())
    model_args = dict(config['model'])
    model_args['dim_mults'] = tuple(model_args['dim_mults'])
    model = TemporalUnet(**model_args).to('cuda')
    diffusion = GaussianDiffusion(model, **config['diffusion']).to('cuda')
    checkpoint = ASSETS / f'state_{config["epoch"]}.pt'
    diffusion.load_state_dict(torch.load(checkpoint, map_location='cuda')['ema'])
    diffusion.eval()
    normalizer = object.__new__(DatasetNormalizer)
    normalizer.action_dim, normalizer.observation_dim = 2, 4
    # The checkpoint was trained in normalized coordinates, approximately
    # [-1, 1] per channel. These saved dataset minima/maxima restore exactly
    # that scaling for [x, y, vx, vy] and [action_x, action_y]. They are not
    # risk parameters, covariance, rollout results, or a second trained model.
    with np.load(ASSETS / 'normalization.npz') as limits:
        normalizer.normalizers = {
            key: LimitsNormalizer(np.stack([limits[key + '_mins'], limits[key + '_maxs']]))
            for key in ('actions', 'observations')
        }
    return Policy(diffusion, normalizer)
