"""Float64 device-native anchor builder; unchanged frozen affine energy.

Only invariant topology/covariance is cached. Every build uses a detached
fresh anchor. Existing CPU and hybrid implementations remain the oracle.
The tiny terminal rotation-log model deliberately retains CPU/SciPy semantics.
No GPU is selected implicitly; construct with device='cpu' for CPU tests.
"""
import numpy as np
import torch
import covariance_energy as cpu
import gpu_energy as reference


def separation(d, jac, radius):
    length = torch.linalg.vector_norm(d, dim=-1)
    invalid = length <= cpu.SEPARATION_EPS
    safe = torch.where(invalid, torch.ones_like(length), length)
    normal = torch.where(invalid[..., None], torch.zeros_like(d), d/safe[..., None])
    return length-radius, torch.einsum('...c,...cj->...j', normal, jac), invalid


class NativeContext:
    """Immutable-in-use invariant tensors; no pose/clearance/Jacobian cache."""
    def __init__(self, arm, device='cpu', indices=None):
        self.arm = arm
        self.device = torch.device(device)
        self.cast = lambda x: torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float64)
        ix = np.arange(2001) if indices is None else np.asarray(indices, dtype=int)
        if (ix < 0).any() or (ix > 2000).any() or len(np.unique(ix)) != len(ix):
            raise ValueError('invalid dense indices')
        self.indices = torch.as_tensor(ix, device=self.device, dtype=torch.long)
        self.production_grid = np.array_equal(ix, np.arange(2001))
        self.k = self.indices//10
        self.next_k = torch.minimum(self.k+1, torch.full_like(self.k, 200))
        self.r = (self.indices%10).to(torch.float64)/10.
        var = cpu.dense_variance(ix)
        cov, self.psd_roundoff_clips = cpu.checked_psd(var[:, None, None]*np.eye(7))
        # This prototype has fixed isotropic covariance. Generic covariances
        # must not silently use the scalar projection shortcut.
        if not np.array_equal(cov, var[:, None, None]*np.eye(7)):
            raise ValueError('scalar projection requires exact isotropic schedule')
        self.var = self.cast(var)
        self.device = self.var.device  # Resolve e.g. 'cuda' to actual 'cuda:0'.
        self.lo, self.hi, self.vmax = [self.cast(v) for v in (arm.lo, arm.hi, arm.vmax)]
        self.velocity_sd = [self.cast(np.sqrt(v))[:, None] for v in cpu.velocity_variances()]
        self.centers, self.radii = self.cast(arm.centers), self.cast(arm.radii)
        self.models = []
        for kind in range(2):
            pending = list(arm.trees[kind]); ancestors = {'panda_link0': np.zeros(7, bool)}
            ordered = []
            while pending:
                progress = False
                for item in pending[:]:
                    parent, child, r0, t0, idx, typ, val, axis = item
                    if parent not in ancestors:
                        continue
                    active = ancestors[parent].copy()
                    if idx is not None:
                        if not np.allclose(axis, [0, 0, 1]):
                            raise ValueError('native FK requires z revolute axis')
                        active[idx] = True
                    ancestors[child] = active
                    ordered.append((parent, child, self.cast(r0), self.cast(t0), idx,
                                    typ, self.cast(axis*val)))
                    pending.remove(item); progress = True
                if not progress:
                    raise ValueError('disconnected native tree')
            spheres = arm.spheres[kind]
            groups = []
            for link in dict.fromkeys(s[0] for s in spheres):
                ids = [j for j, s in enumerate(spheres) if s[0] == link]
                groups.append((link, torch.as_tensor(ids, device=self.device),
                               self.cast([spheres[j][1] for j in ids]),
                               torch.as_tensor(ancestors[link], device=self.device)))
            pairs = arm.pairs[kind]
            self.models.append({'tree': ordered, 'groups': groups,
                'sphere_radii': self.cast([s[2] for s in spheres]),
                'ii': torch.as_tensor(pairs[:, 0].astype(int), device=self.device),
                'jj': torch.as_tensor(pairs[:, 1].astype(int), device=self.device),
                'threshold': self.cast(pairs[:, 2]), 'sphere_count': len(spheres)})

    def dense(self, q):
        return (1-self.r[:, None])*q[self.k]+self.r[:, None]*q[self.next_k]

    def fk(self, q, kind):
        n = len(q)
        eye = torch.eye(3, dtype=torch.float64, device=self.device)
        poses = {'panda_link0': (eye.expand(n, 3, 3), torch.zeros((n, 3), dtype=torch.float64, device=self.device))}
        axes = torch.zeros((n, 7, 3), dtype=torch.float64, device=self.device)
        origins = torch.zeros_like(axes)
        # Each global arm joint axis/origin is computed once. Cached per-link
        # ancestor masks ensure non-ancestor axes never enter its Jacobian.
        for parent, child, r0, t0, idx, typ, offset in self.models[kind]['tree']:
            rp, tp = poses[parent]
            r = rp@r0
            t = tp+torch.einsum('nij,j->ni', rp, t0)
            if idx is not None:
                axes[:, idx] = r[:, :, 2]; origins[:, idx] = t
                co, si = torch.cos(q[:, idx]), torch.sin(q[:, idx])
                rz = torch.zeros((n, 3, 3), dtype=torch.float64, device=self.device)
                rz[:, 0, 0] = co; rz[:, 0, 1] = -si
                rz[:, 1, 0] = si; rz[:, 1, 1] = co; rz[:, 2, 2] = 1.
                r = r@rz
            elif typ == 'prismatic':
                t = t+torch.einsum('nij,j->ni', r, offset)
            poses[child] = r, t
        return poses, axes, origins

    def geometry(self, q, kind):
        model = self.models[kind]
        poses, axes, origins = self.fk(q, kind)
        pts = torch.empty((len(q), model['sphere_count'], 3), dtype=torch.float64, device=self.device)
        jac = torch.empty((*pts.shape, 7), dtype=torch.float64, device=self.device)
        for link, ids, xyz, active in model['groups']:
            r, t = poses[link]
            p = torch.einsum('nij,sj->nsi', r, xyz)+t[:, None]
            pts[:, ids] = p
            j = torch.linalg.cross(axes[:, None].expand(-1, len(ids), -1, -1),
                                   p[:, :, None]-origins[:, None], dim=-1)
            jac[:, ids] = (j*active[None, None, :, None]).transpose(-1, -2)
        rr = model['sphere_radii']
        world = separation(pts[:, :, None]-self.centers[None, None],
            jac[:, :, None].expand(-1, -1, len(self.centers), -1, -1),
            rr[None, :, None]+self.radii[None, None])
        ii, jj = model['ii'], model['jj']
        own = separation(pts[:, ii]-pts[:, jj], jac[:, ii]-jac[:, jj], model['threshold'])
        return poses, pts, jac, world, own

    @torch.no_grad()
    def build(self, anchor, target):
        if anchor.shape != (201, 7) or anchor.dtype != torch.float64 or anchor.device != self.device:
            raise ValueError('anchor must be201x7 float64 on context device')
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError('nonfinite anchor')
        model = NativeFrozen(self)
        model.anchor = anchor.detach().clone()
        model.qbar = self.dense(model.anchor)
        model.blocks = []
        bad_flags = []
        for kind, label in enumerate(('mpd', 'curobo')):
            _, _, _, world, own = self.geometry(model.qbar, kind)
            for name, (g, jac, bad) in ((f'world_{label}', world), (f'self_{label}', own)):
                g = g.reshape(len(self.indices), -1)
                jac = jac.reshape(len(self.indices), -1, 7)
                bad = bad.reshape(g.shape)
                variance = self.var[:, None]*jac.square().sum(-1)
                if not bool(torch.isfinite(jac).all() & torch.isfinite(g).all() & torch.isfinite(variance).all()):
                    raise ValueError('nonfinite geometry/Jacobian/projected variance')
                # The scalar covariance and sum-of-squares are nonnegative;
                # no new variance floor or probability clipping is introduced.
                model.blocks.append((name, g, jac, torch.sqrt(variance), bad))
                bad_flags.append(bad.any())
        # Preserve the existing exact SciPy branch behavior. Only one7-vector
        # and tiny12x7 terminal model cross the device boundary, not geometry.
        endpoint = model.anchor[-1:].cpu().numpy()
        pose = cpu.fk_jac(self.arm, endpoint, 0)['panda_hand']
        g, jac, bad = cpu.pose_box(pose, np.asarray(target))
        sd = np.sqrt(cpu.cross_cov(200, 200)*np.sum(jac*jac, axis=1))
        model.pose = self.cast(g), self.cast(jac), self.cast(sd), torch.as_tensor(bad, device=self.device)
        model.valid = not bool(torch.stack(bad_flags).any()) and not bad.any()
        return model


class NativeFrozen(reference.GPUFrozen):
    """Reuses the SAME evaluator; bypasses only its CPU-to-CUDA constructor."""
    def __init__(self, context):
        self.context = context
        self.cast = context.cast
        for name in ('indices', 'var', 'lo', 'hi', 'vmax', 'velocity_sd', 'production_grid', 'psd_roundoff_clips'):
            setattr(self, name, getattr(context, name))
