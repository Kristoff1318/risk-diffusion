"""CPU contract tests; no robot planning, CUDA queries or risk experiment."""
from fractions import Fraction
import types
import unittest
import numpy as np
from validation_adapter import Request
from validation_adapter.physical_bridge import PhysicalBackend, freeze_candidate, validate_current


class BridgeTests(unittest.TestCase):
    def backend(self, streams=(False, False), observe=None):
        calls = []; slices = []; syncs = []
        def errors(n, seed, scale):
            calls.append((n, seed, scale))
            ids = np.arange(n, dtype=float)[:, None, None]
            return tuple(np.broadcast_to(ids, (n, t, 7)) for t in (2001, 200, 200))
        def audit(instance, nom, target, e, chunk):
            slot = int(nom[0, 0]); ids = e[0][:, 0, 0].astype(int)
            slices.append((slot, ids.tolist(), chunk))
            flags = np.zeros((len(ids), 9), bool); flags[:, 0] = streams[slot]
            return {}, flags
        api = types.SimpleNamespace(errors=errors, audit=audit)
        core = types.SimpleNamespace(cast=lambda a: np.array(a, copy=True))
        backend = PhysicalBackend(api, core, np.zeros((3, 4)), model_id='model',
            synchronize=lambda: syncs.append(1), partial_feedback=True, observe=observe)
        return backend, calls, slices, syncs

    def candidates(self):
        return [freeze_candidate(i, np.full((201, 7), float(i)), cost=i+1., eligible=True) for i in (0, 1)]

    def test_default_and_lazy_decision_and_original_indices(self):
        req = Request('request', 'model', 123, Fraction(1, 20))
        backend, calls, slices, syncs = self.backend((True, False))
        fixed = validate_current(self.candidates(), req, backend)
        self.assertEqual(fixed['mode'], 'fixed'); self.assertEqual(fixed['selected'], 1)
        self.assertEqual(fixed['executed'], 2048); self.assertEqual(calls, [(1024, 123, 2.)])
        self.assertEqual([len(s[1]) for s in slices], [1024, 1024])
        original_identity = fixed['physical_backend']['full_error_bank_sha256']
        lazy = validate_current(self.candidates(), req, backend, mode='lazy', chunk_size=16)
        self.assertEqual(lazy['selected'], 1); self.assertEqual(lazy['executed'], 48+1024)
        self.assertEqual(lazy['certificate']['n'], 1024)
        self.assertEqual(original_identity, lazy['physical_backend']['full_error_bank_sha256'])
        self.assertEqual(lazy['physical_backend']['padding_trials'], 0)
        self.assertEqual(lazy['physical_backend']['source_audit_calls'], 67)
        self.assertEqual(len(syncs), fixed['physical_backend']['synchronization_calls']+lazy['physical_backend']['synchronization_calls'])

    def test_hash_and_mutation_fail_closed(self):
        with self.assertRaises(ValueError):
            freeze_candidate(0, np.zeros((201, 7)), cost=1., eligible=True, expected_path_hash='wrong')
        cs = self.candidates(); cs[0].payload.nominal.setflags(write=True); cs[0].payload.nominal[0, 0] += 1
        backend, *_ = self.backend()
        r = validate_current(cs, Request('request', 'model', 1, Fraction(1, 20)), backend)
        self.assertIsNone(r['selected']); self.assertEqual(r['rejection'], 'backend-error')
        self.assertEqual(r['executed'], 0)

    def test_wrong_model_N_alpha_and_no_BC(self):
        for req in [Request('r', 'other', 1, Fraction(1, 20)), Request('r', 'model', 1, Fraction(1, 20), horizon=100),
                    Request('r', 'model', 1, Fraction(1, 20), alpha_per_candidate=Fraction(1, 100))]:
            backend, *_ = self.backend()
            r = validate_current(self.candidates(), req, backend)
            self.assertEqual(r['rejection'], 'backend-error'); self.assertIsNone(r['selected'])
        backend, *_ = self.backend()
        with self.assertRaises(ValueError):
            validate_current(self.candidates(), Request('r', 'model', 1, Fraction(1, 20)), backend, mode='B')

    def test_expensive_feedback_fallback(self):
        backend, *_ = self.backend()
        backend.partial_feedback = False
        r = validate_current(self.candidates(), Request('r', 'model', 1, Fraction(1, 20)), backend, mode='lazy')
        self.assertEqual(r['selected'], 0); self.assertEqual(r['executed'], 1024)
        self.assertEqual(r['batch_calls'], 1)


if __name__ == '__main__':
    unittest.main()
