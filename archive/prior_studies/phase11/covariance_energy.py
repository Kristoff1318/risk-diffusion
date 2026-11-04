"""CPU frozen affine distance-risk energy; no optimizer or sampling.

Geometry Jacobians/projected variances are fixed in FrozenEnergy. Rebuilding
the object changes the surrogate. This is NOT a nonlinear risk certificate.
"""
import math
import numpy as np
from scipy.special import ndtr
from scipy.spatial.transform import Rotation

DT = .05
H = 200
K = 2.
A = .9
DELTA = .05
PENALTY = 500.
SEPARATION_EPS = 1e-12


def cross_cov(k, l):
    m = min(k, l)
    return (A**(k+l)*.004**2 + (1-A**k)*(1-A**l)/K**2*.002**2
            + DT**2*A**abs(k-l)*(1-A**(2*m))/(1-A*A)*.010**2)


def dense_variance(indices):
    out = []
    for i in indices:
        k, r = divmod(int(i), 10)
        r /= 10.
        v = cross_cov(k, k) if k == H else (
            (1-r)**2*cross_cov(k, k)+r*r*cross_cov(k+1, k+1)
            + 2*r*(1-r)*cross_cov(k, k+1))
        out.append(v)
    return np.array(out)


def velocity_variances():
    pos = np.array([cross_cov(k, k) for k in range(H)])
    cmd = K*K*pos
    exe = cmd + .002**2 + .010**2 - 2*K*(1-A**np.arange(H))/K*.002**2
    return cmd, exe


def checked_psd(c):
    c = np.asarray(c, dtype=float)
    if not np.isfinite(c).all() or c.shape[-2:] != (7, 7):
        raise ValueError('nonfinite or wrong covariance shape')
    tol = 1e-14 + 1e-10*np.max(np.abs(c))
    if np.max(np.abs(c-c.swapaxes(-1, -2))) > tol:
        raise ValueError('asymmetric covariance')
    w, v = np.linalg.eigh((c+c.swapaxes(-1, -2))/2)
    if np.min(w) < -tol:
        raise ValueError('indefinite covariance')
    clipped = int((w < 0).sum())
    return (v*np.maximum(w, 0)[..., None, :])@v.swapaxes(-1, -2), clipped


def probability(margin, sigma, invalid=None):
    margin, sigma = np.broadcast_arrays(np.asarray(margin), np.asarray(sigma))
    if not np.isfinite(margin).all() or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('invalid margin or standard deviation')
    p = (margin < 0).astype(float)
    dp = np.zeros_like(p)
    nz = sigma > 0
    z = margin[nz]/sigma[nz]
    p[nz] = ndtr(-z)
    dp[nz] = -np.exp(-.5*z*z)/(math.sqrt(2*math.pi)*sigma[nz])
    if invalid is not None:
        p = np.where(invalid, 1., p)
        dp = np.where(invalid, 0., dp)
    return p, dp


def separation(d, jac, radius):
    length = np.linalg.norm(d, axis=-1)
    invalid = length <= SEPARATION_EPS
    normal = np.zeros_like(d)
    np.divide(d, length[..., None], out=normal, where=~invalid[..., None])
    return length-radius, np.einsum('...c,...cj->...j', normal, jac), invalid


def fk_jac(arm, q, kind):
    """Native FK plus world-frame revolute axes/origins and ancestor masks."""
    n = len(q)
    root = (np.broadcast_to(np.eye(3), (n, 3, 3)), np.zeros((n, 3)),
            np.zeros((n, 7, 3)), np.zeros((n, 7, 3)), np.zeros(7, bool))
    poses = {'panda_link0': root}
    pending = list(arm.trees[kind])
    while pending:
        progress = False
        for item in pending[:]:
            parent, child, r0, t0, idx, typ, val, axis = item
            if parent not in poses:
                continue
            rp, tp, axes, origins, active = poses[parent]
            r = rp@r0
            t = tp+np.einsum('nij,j->ni', rp, t0)
            axes, origins, active = axes.copy(), origins.copy(), active.copy()
            if idx is not None:
                if not np.allclose(axis, [0, 0, 1]):
                    raise ValueError('native FK requires z revolute axis')
                axes[:, idx] = r[:, :, 2]
                origins[:, idx] = t
                active[idx] = True
                co, si = np.cos(q[:, idx]), np.sin(q[:, idx])
                rz = np.zeros((n, 3, 3))
                rz[:, 0, 0] = co; rz[:, 0, 1] = -si
                rz[:, 1, 0] = si; rz[:, 1, 1] = co; rz[:, 2, 2] = 1
                r = r@rz
            elif typ == 'prismatic':
                t = t+np.einsum('nij,j->ni', r, axis*val)
            poses[child] = r, t, axes, origins, active
            pending.remove(item); progress = True
        if not progress:
            raise ValueError('disconnected native tree')
    return poses


def point_jac(pose, local):
    r, t, axes, origins, active = pose
    p = np.einsum('nij,j->ni', r, local)+t
    jac = np.cross(axes, p[:, None]-origins)*active[None, :, None]
    return p, jac.swapaxes(-1, -2)


def geometry(arm, q, kind):
    poses = fk_jac(arm, q, kind)
    points, jac = zip(*(point_jac(poses[link], xyz) for link, xyz, _ in arm.spheres[kind]))
    points, jac = np.stack(points, 1), np.stack(jac, 1)
    rr = np.array([s[2] for s in arm.spheres[kind]])
    world = separation(points[:, :, None]-arm.centers[None, None],
                       np.broadcast_to(jac[:, :, None], (len(q), len(rr), len(arm.centers), 3, 7)),
                       rr[None, :, None]+arm.radii[None, None])
    ii, jj = arm.pairs[kind][:, :2].astype(int).T
    own = separation(points[:, ii]-points[:, jj], jac[:, ii]-jac[:, jj],
                     arm.pairs[kind][:, 2])
    return poses, world, own


def pose_box(pose, target):
    r, t, axes, _, active = pose
    pos, jp = point_jac(pose, np.zeros(3))
    phi = Rotation.from_matrix(target[:, :3].T@r[0]).as_rotvec()
    theta = np.linalg.norm(phi)
    invalid = theta >= math.pi-1e-6
    hat = np.array([[0., -phi[2], phi[1]], [phi[2], 0., -phi[0]], [-phi[1], phi[0], 0.]])
    coeff = 1/12+theta*theta/720 if theta < 1e-4 else (1-.5*theta/math.tan(theta/2))/theta**2
    inv_left = np.eye(3)-.5*hat+coeff*(hat@hat)
    jo = inv_left@target[:, :3].T@axes[0].T
    jo *= active[None]
    err = np.concatenate((pos[0]-target[:, 3], phi))
    jac = np.concatenate((jp[0], jo))
    tol = np.array([.01]*3+[math.radians(2)]*3)/math.sqrt(3)
    bad = np.array([False]*3+[invalid]*3)
    return (np.concatenate((tol-err, tol+err)), np.concatenate((-jac, jac)),
            np.concatenate((bad, bad)))


def dense(q, indices):
    i = np.asarray(indices)
    k = i//10; r = (i%10)/10.
    return (1-r[:, None])*q[k]+r[:, None]*q[np.minimum(k+1, H)]


def cost_gradient(q):
    vel = np.diff(q, axis=0)/DT
    acc = np.diff(vel, axis=0)/DT
    cost = DT*np.sum(vel*vel)+.005*DT*np.sum(acc*acc)
    gv = 2*DT*vel
    ga = 2*.005*DT*acc/DT
    gv[:-1] -= ga; gv[1:] += ga
    grad = np.zeros_like(q)
    grad[:-1] -= gv/DT; grad[1:] += gv/DT
    return float(cost), grad


class FrozenEnergy:
    def __init__(self, arm, anchor, target, indices=None):
        self.arm = arm
        self.anchor = np.asarray(anchor, dtype=float).copy()
        if self.anchor.shape != (201, 7) or not np.isfinite(self.anchor).all():
            raise ValueError('invalid nominal path')
        self.indices = np.arange(2001) if indices is None else np.asarray(indices, dtype=int)
        if (self.indices < 0).any() or (self.indices > 2000).any() or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError('invalid dense state indices')
        self.production_grid = np.array_equal(self.indices, np.arange(2001))
        self.qbar = dense(self.anchor, self.indices)
        self.var = dense_variance(self.indices)
        cov, self.psd_roundoff_clips = checked_psd(self.var[:, None, None]*np.eye(7))
        self.blocks = []
        for kind, label in enumerate(('mpd', 'curobo')):
            _, world, own = geometry(arm, self.qbar, kind)
            for name, (g, jac, bad) in ((f'world_{label}', world), (f'self_{label}', own)):
                g = g.reshape(len(self.indices), -1)
                jac = jac.reshape(len(self.indices), -1, 7)
                bad = bad.reshape(g.shape)
                if not np.isfinite(jac).all():
                    raise ValueError('nonfinite geometry Jacobian')
                variance = np.einsum('tpj,tjk,tpk->tp', jac, cov, jac)
                tol = 1e-14+1e-10*np.max(np.abs(cov))
                if (variance < -tol).any():
                    raise ValueError('negative projected variance')
                self.psd_roundoff_clips += int((variance < 0).sum())
                self.blocks.append((name, g, jac, np.sqrt(np.maximum(variance, 0)), bad))
        terminal = fk_jac(arm, self.anchor[-1:], 0)['panda_hand']
        g, jac, bad = pose_box(terminal, np.asarray(target))
        self.pose = g, jac, np.sqrt(cross_cov(H, H)*np.sum(jac*jac, axis=1)), bad
        self.valid = not any(b[-1].any() for b in self.blocks) and not bad.any()

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (201, 7) or not np.isfinite(q).all():
            raise ValueError('invalid evaluation path')
        qd = dense(q, self.indices); shift = qd-self.qbar
        gd = np.zeros_like(qd); grad = np.zeros_like(q); stats = {}; risk = 0.

        def record(name, p):
            nonlocal risk
            value = float(p.sum()); risk += value
            stats[name] = {'events': int(p.size), 'sum_probability': value}

        for name, g, jac, sd, bad in self.blocks:
            p, dp = probability(g+np.einsum('tpj,tj->tp', jac, shift), sd, bad)
            gd += np.einsum('tp,tpj->tj', dp, jac); record(name, p)
        sd = np.sqrt(self.var[:, None])
        for suffix, margin, sign in (('lower', qd-self.arm.lo, 1), ('upper', self.arm.hi-qd, -1)):
            p, dp = probability(margin, sd)
            gd += sign*dp; record('joint_'+suffix, p)
        k = self.indices//10; r = (self.indices%10)/10.
        np.add.at(grad, k, (1-r[:, None])*gd)
        np.add.at(grad, np.minimum(k+1, H), r[:, None]*gd)
        ff = np.diff(q, axis=0)/DT; gv = np.zeros_like(ff)
        for name, variance in zip(('command_velocity', 'executed_velocity'), velocity_variances()):
            for suffix, margin, sign in (('upper', self.arm.vmax-ff, -1), ('lower', self.arm.vmax+ff, 1)):
                p, dp = probability(margin, np.sqrt(variance[:, None]))
                gv += sign*dp; record(name+'_'+suffix, p)
        grad[:-1] -= gv/DT; grad[1:] += gv/DT
        g, jac, sd, bad = self.pose
        p, dp = probability(g+jac@(q[-1]-self.anchor[-1]), sd, bad)
        grad[-1] += dp@jac; record('terminal_box', p)
        cost, gc = cost_gradient(q)
        excess = max(0., (risk-DELTA)/DELTA)
        energy = cost+PENALTY*excess*excess
        ge = gc+2*PENALTY*excess/DELTA*grad
        return {'risk_sum': risk, 'cost': cost, 'energy': energy,
                'gradient': ge, 'risk_gradient': grad, 'components': stats,
                'events': sum(v['events'] for v in stats.values()),
                'valid': self.valid, 'production_grid': self.production_grid,
                'psd_roundoff_clips': self.psd_roundoff_clips}


def production_event_count(arm):
    per_state = sum(len(s)*len(arm.centers) for s in arm.spheres)+sum(len(p) for p in arm.pairs)+14
    return {'geometry_joint_per_state': per_state, 'states': 2001,
            'velocity_events': 200*7*2*2, 'terminal_box_events': 12,
            'total': 2001*per_state+5600+12}
