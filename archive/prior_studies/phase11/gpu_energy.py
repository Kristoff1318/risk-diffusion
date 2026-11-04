"""CUDA float64 evaluator of the unchanged CPU-built frozen surrogate."""
import math
import numpy as np
import torch
from scipy.interpolate import BSpline
import covariance_energy as cpu


def basis():
    knots = np.pad(np.linspace(0, 1, 25), 5, 'edge')
    all_b = BSpline(knots, np.eye(29), 5)(np.linspace(0, 1, 201))
    return np.column_stack((all_b[:, 3:26], all_b[:, 26:].sum(1))), all_b[:, :3].sum(1)


def probability(margin, sigma, invalid=None):
    positive = sigma > 0
    safe = torch.where(positive, sigma, torch.ones_like(sigma))
    p = torch.where(positive, .5*torch.erfc(margin/safe/math.sqrt(2)),
                    (margin < 0).to(margin.dtype))
    if invalid is not None:
        p = torch.where(invalid, torch.ones_like(p), p)
    return p


class GPUFrozen:
    def __init__(self, model):
        self.cast = lambda x: torch.as_tensor(np.asarray(x), device='cuda', dtype=torch.float64)
        self.valid = model.valid
        self.production_grid = model.production_grid
        self.indices = torch.as_tensor(model.indices, device='cuda', dtype=torch.long)
        self.qbar = self.cast(model.qbar)
        self.var = self.cast(model.var)
        self.blocks = [(name, self.cast(g), self.cast(j), self.cast(sd),
                        torch.as_tensor(bad, device='cuda', dtype=torch.bool))
                       for name, g, j, sd, bad in model.blocks]
        g, j, sd, bad = model.pose
        self.pose = self.cast(g), self.cast(j), self.cast(sd), torch.as_tensor(bad, device='cuda', dtype=torch.bool)
        self.anchor = self.cast(model.anchor)
        self.lo, self.hi, self.vmax = (self.cast(v) for v in (model.arm.lo, model.arm.hi, model.arm.vmax))
        self.velocity_sd = [self.cast(np.sqrt(v))[:, None] for v in cpu.velocity_variances()]

    def evaluate(self, q):
        if not self.valid:
            raise ValueError('singular frozen model cannot be optimized')
        k = self.indices//10
        r = (self.indices%10).to(q.dtype)/10.
        qd = (1-r[:, None])*q[k]+r[:, None]*q[torch.minimum(k+1, torch.full_like(k, 200))]
        shift = qd-self.qbar
        components = {}; counts = {}

        def record(name, p):
            components[name] = p.sum(); counts[name] = p.numel()

        for name, g, jac, sd, bad in self.blocks:
            record(name, probability(g+torch.einsum('tpj,tj->tp', jac, shift), sd, bad))
        sd = torch.sqrt(self.var[:, None])
        record('joint_lower', probability(qd-self.lo, sd))
        record('joint_upper', probability(self.hi-qd, sd))
        ff = q.diff(dim=0)/cpu.DT
        for name, vs in zip(('command_velocity', 'executed_velocity'), self.velocity_sd):
            record(name+'_upper', probability(self.vmax-ff, vs))
            record(name+'_lower', probability(self.vmax+ff, vs))
        g, jac, sd, bad = self.pose
        record('terminal_box', probability(g+jac@(q[-1]-self.anchor[-1]), sd, bad))
        risk = sum(components.values())
        acc = ff.diff(dim=0)/cpu.DT
        cost = cpu.DT*ff.square().sum()+.005*cpu.DT*acc.square().sum()
        energy = cost+cpu.PENALTY*torch.relu((risk-cpu.DELTA)/cpu.DELTA).square()
        return {'risk_sum': risk, 'cost': cost, 'energy': energy,
                'components': components, 'event_counts': counts, 'events': sum(counts.values())}


def numeric(result):
    return {k: ({name: float(value.detach()) for name, value in v.items()}
                if k == 'components' else v if isinstance(v, (int, dict)) else float(v.detach()))
            for k, v in result.items()}
