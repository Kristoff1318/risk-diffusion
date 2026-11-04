"""Opt-in bridge to the current injected whole-arm validator.

No historical core is imported or patched. Caller supplies its CURRENT core
API/instance, immutable declared model identity, and completion synchronization.
Default remains fully scheduled fixed-N. No sequential early-admission mode.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time
import numpy as np
from . import Bank, Candidate, FrozenBankBackend, Request, validate


def path_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class PhysicalPath:
    nominal: np.ndarray


def freeze_candidate(slot, nominal, *, cost, eligible, expected_path_hash=None):
    """Copy the exact executed float64 path; preserve externally frozen metadata."""
    q = np.asarray(nominal)
    if q.shape != (201, 7) or q.dtype != np.float64:
        raise ValueError('executed nominal path must be201x7 float64')
    owned = np.array(q, copy=True, order='C'); owned.setflags(write=False)
    h = path_hash(owned)
    if expected_path_hash is not None and h != expected_path_hash:
        raise ValueError('candidate bytes do not match declared path hash')
    return Candidate(slot, float(cost), eligible, h, PhysicalPath(owned))


def bank_hash(errors):
    h = hashlib.sha256()
    for array in errors:
        h.update(str(array.shape).encode()); h.update(str(array.dtype).encode())
        h.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return h.hexdigest()


class PhysicalBackend:
    """Full original noise(N,seed,scale), then original errors/tracking slices.

    core_api.errors(N,seed,scale) and core_api.audit(instance,nom,target,errors,
    kernel_rollouts) must be the qualified CURRENT implementations. Their
    physical model must remain fixed for the call. The bridge owns path/goal/
    bank snapshots, verifies their hashes, and never mutates the supplied core.
    Private per-session tensors are reused only for the same immutable path.
    """
    def __init__(self, core_api, instance, target, *, model_id, synchronize,
                 noise_scale=2., kernel_rollouts=16, partial_feedback=False,
                 observe=None):
        if not model_id or not callable(synchronize) or kernel_rollouts <= 0:
            raise ValueError('model identity, sync callback and positive kernel chunk required')
        target = np.asarray(target)
        if target.shape != (3, 4) or not np.isfinite(target).all():
            raise ValueError('finite3x4 target required')
        self.api, self.instance = core_api, instance
        self.target = np.array(target, dtype=np.float64, copy=True); self.target.setflags(write=False)
        self.target_hash = path_hash(self.target)
        self.model_id, self.synchronize = model_id, synchronize
        self.scale, self.kernel_rollouts = noise_scale, kernel_rollouts
        self.partial_feedback, self.observe = partial_feedback, observe
        self.metrics = {}; self.active = False

    def _sync(self):
        self.metrics['synchronization_calls'] += 1
        self.synchronize()

    def _verify(self, candidate):
        if not isinstance(candidate.payload, PhysicalPath):
            raise ValueError('physical bridge requires frozen path payload')
        if path_hash(candidate.payload.nominal) != candidate.path_hash:
            raise ValueError('immutable candidate path changed')
        if path_hash(self.target) != self.target_hash:
            raise ValueError('immutable target changed')

    def open(self, request, candidates):
        if self.active:
            raise RuntimeError('no concurrent/mutable validation sessions')
        if request.model_id != self.model_id or request.horizon != 1024 or request.alpha_per_candidate != Fraction(1, 40):
            raise ValueError('current bridge preservesN1024/alpha.025 and declared model exactly')
        self.metrics = {'noise_tracking_interpolation_seconds': 0., 'physical_validator_seconds': 0.,
            'source_audit_calls': 0, 'geometry_rollout_chunks': 0, 'kernel_rollouts': self.kernel_rollouts,
            'synchronization_calls': 0, 'sample_slices': [], 'padding_trials': 0,
            'actual_CUDA_kernel_launches': 'not profiled; geometry chunks are not launch counts'}
        for candidate in candidates:
            self._verify(candidate)
        before = time.perf_counter()
        errors = self.api.errors(request.horizon, request.seed, self.scale)
        if len(errors) != 3:
            raise ValueError('current errors must return e,executed_velocity,command errors')
        shapes = [(1024, 2001, 7), (1024, 200, 7), (1024, 200, 7)]
        errors = tuple(np.array(a, copy=True, order='C') for a in errors)
        for a, shape in zip(errors, shapes):
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('wrong original full-horizon error shapes or nonfinite bank')
            a.setflags(write=False)
        self.metrics['noise_tracking_interpolation_seconds'] = time.perf_counter()-before
        identity = bank_hash(errors); self.metrics['full_error_bank_sha256'] = identity
        trajectories = {c.slot: self.instance.cast(np.array(c.payload.nominal, copy=True)) for c in candidates}
        target = self.instance.cast(np.array(self.target, copy=True)); self._sync()
        bridge = self

        def evaluate(candidate, sliced):
            bridge._verify(candidate)
            bridge._sync(); began = time.perf_counter()
            _, flags = bridge.api.audit(bridge.instance, trajectories[candidate.slot], target,
                                      sliced, bridge.kernel_rollouts)
            bridge._sync()
            bridge.metrics['physical_validator_seconds'] += time.perf_counter()-began
            n = len(sliced[0])
            bridge.metrics['source_audit_calls'] += 1
            bridge.metrics['geometry_rollout_chunks'] += math.ceil(n/bridge.kernel_rollouts)
            bridge._verify(candidate)
            if flags.shape != (n, 9):
                raise ValueError('expected nine whole-execution event columns')
            if bridge.observe is not None:
                bridge.observe(candidate.slot, np.array(flags, copy=True))
            return flags

        backend = FrozenBankBackend(lambda req: Bank(errors, req.horizon),
            lambda bank, a, b: tuple(v[a:b] for v in bank), evaluate,
            event_count=9, partial_feedback=self.partial_feedback)
        base_session = backend.open(request, candidates)
        self.active = True

        class PhysicalSession:
            partial_feedback = bridge.partial_feedback
            def evaluate(self, candidate, start, stop):
                bridge.metrics['sample_slices'].append({'slot': candidate.slot, 'start': start, 'stop': stop})
                return base_session.evaluate(candidate, start, stop)
            def close(self):
                try:
                    base_session.close(); bridge._sync()
                    for c in candidates:
                        bridge._verify(c)
                    if bank_hash(errors) != identity:
                        raise ValueError('immutable physical bank changed during validation')
                finally:
                    bridge.active = False
        return PhysicalSession()


def validate_current(candidates, request, backend, *, mode='fixed', chunk_size=16):
    """Original fixed behavior by default; lazy allocation requires explicit opt-in."""
    result = validate(candidates, request, backend, mode=mode, chunk_size=chunk_size)
    result['physical_backend'] = dict(backend.metrics)
    if result['certificate'] is not None:
        result['certificate']['full_error_bank_sha256'] = backend.metrics['full_error_bank_sha256']
    return result
