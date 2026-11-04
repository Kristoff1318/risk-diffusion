"""CPU regression tests; saved streams are read-only oracles, never production imports."""
import hashlib
import itertools
import json
import pathlib
import random
import time
import unittest
from fractions import Fraction as F

from validation_adapter import (Bank, Batch, Candidate, FrozenBankBackend, Request,
                                acceptance_threshold, validate, whole_execution_failures)

SOURCE = pathlib.Path('/home/kristoff/Documents/experiments/research/risk_diffusion')


def candidates(costs=(1.,2.), eligible=(True,True)):
    return tuple(Candidate(s,costs[s],eligible[s],f'path-{s}') for s in (0,1))


def bit_backend(streams, partial=True, factory_calls=None, slices=None):
    def factory(request):
        if factory_calls is not None:
            factory_calls.append((request.request_id,request.seed,request.horizon))
        return Bank(tuple(range(request.horizon)), request.horizon)
    def evaluate(candidate, indices):
        if slices is not None:
            slices.append((candidate.slot, tuple(indices)))
        return [[int(streams[candidate.slot][index])] for index in indices]
    return FrozenBankBackend(factory, lambda bank,a,b: bank[a:b],evaluate,
                             event_count=1,partial_feedback=partial)


class AdapterTests(unittest.TestCase):
    def test_default_is_fully_scheduled_fixedN(self):
        req=Request('default','model',1,F(1,2),horizon=10)
        streams=[[False]*10]*2
        calls=[]
        result=validate(candidates(eligible=(False,True)),req,bit_backend(streams,slices=calls))
        self.assertEqual(result['mode'],'fixed')
        self.assertEqual(result['executed'],20)
        self.assertEqual(result['selected'],1)
        self.assertEqual(len(calls),2)
        self.assertEqual(result['certificate']['n'],10)
        self.assertNotIn('confidence_interval',result)

    def test_thresholds_match_independentCP(self):
        from scipy.stats import beta
        for p,expected in [(F(1,20),37),(F(1,10),83)]:
            k=acceptance_threshold(1024,p,F(1,40))
            self.assertEqual(k,expected)
            self.assertLessEqual(beta.ppf(.975,k+1,1024-k),float(p))
            self.assertGreater(beta.ppf(.975,k+2,1023-k),float(p))

    def test_boundary_ties_eligibility_and_preferred_resolution(self):
        req=Request('boundaries','model',2,F(1,2),horizon=10)
        k=acceptance_threshold(10,req.epsilon,F(1,40))
        self.assertEqual(k,1)
        # Count k passes; k+1 rejects, with failures at front/back boundaries.
        streams=[[False]*10,[True]*10]
        for count in (k,k+1,0,10):
            streams.extend([[True]*count+[False]*(10-count),
                            [False]*(10-count)+[True]*count])
        for first,second in itertools.product(streams,repeat=2):
            for costs in ((1.,1.),(2.,1.)):
                for eligible in ((True,True),(False,True),(True,False),(False,False)):
                    cs=candidates(costs,eligible)
                    fixed=validate(cs,req,bit_backend([first,second]))
                    for chunk in (1,3,6,16):
                        lazy=validate(cs,req,bit_backend([first,second]),mode='lazy',chunk_size=chunk)
                        self.assertEqual(lazy['selected'],fixed['selected'])
                        self.assertGreaterEqual(lazy['executed'],lazy['examined'])
                        if lazy['selected'] is not None:
                            self.assertEqual(lazy['certificate']['n'],10)
        # Preferred remains unresolved until N; later cannot be returned early.
        log=[]
        out=validate(candidates((1.,1.)),req,bit_backend([[False]*10]*2,slices=log),mode='lazy',chunk_size=3)
        self.assertEqual(out['selected'],0)
        self.assertEqual([len(indices) for _,indices in log],[3,3,3,1])
        self.assertEqual({s for s,_ in log},{0})
        reversed_slots=tuple(reversed(candidates((1.,1.))))
        for mode in ('fixed','lazy'):
            out=validate(reversed_slots,req,bit_backend([[False]*10]*2),mode=mode,chunk_size=3)
            self.assertEqual(out['selected'],0)

    def test_no_cheap_feedback_falls_back_to_fullN(self):
        req=Request('batch','model',3,F(1,2),horizon=10)
        result=validate(candidates(),req,bit_backend([[True]*10,[False]*10],partial=False),mode='lazy',chunk_size=3)
        self.assertEqual(result['feedback'],'full-horizon')
        self.assertEqual(result['executed'],20)
        self.assertEqual(result['examined'],12)
        self.assertEqual(result['selected'],1)
        self.assertEqual(result['batch_calls'],2)

    def test_partial_final_chunks_and_repeated_request_world_identity(self):
        calls=[];worlds=[];slices=[]
        req=Request('worlds','model',8675309,F(1,2),horizon=10)
        def factory(request):
            rng=random.Random(request.seed)
            bank=tuple(rng.random() for _ in range(request.horizon))
            worlds.append(bank);calls.append(request.seed)
            return Bank(bank,len(bank))
        def evaluate(candidate,draws):
            slices.append((candidate.slot,tuple(draws)))
            # Cheaper rejects; later succeeds. Both receive same frozen worlds.
            return [[int(candidate.slot==0)] for _ in draws]
        backend=FrozenBankBackend(factory,lambda bank,a,b:bank[a:b],evaluate,event_count=1,partial_feedback=True)
        first=validate(candidates(),req,backend,mode='lazy',chunk_size=3)
        first_slices=slices.copy();slices.clear()
        second=validate(candidates(),req,backend,mode='lazy',chunk_size=3)
        self.assertEqual(first['selected'],second['selected'])
        self.assertEqual(worlds[0],worlds[1])
        self.assertEqual(first_slices,slices)
        self.assertEqual(calls,[req.seed,req.seed])
        self.assertEqual(first_slices[0][1],worlds[0][:3])
        self.assertEqual(first_slices[1][1],worlds[0][:3])
        self.assertEqual(first_slices[-1][1],worlds[0][9:10])
        session=backend.open(req,candidates())
        session.evaluate(candidates()[0],0,3)
        with self.assertRaises(ValueError): session.evaluate(candidates()[0],0,3)
        with self.assertRaises(ValueError): session.evaluate(candidates()[0],4,5)
        session.close()
        with self.assertRaises(RuntimeError): session.evaluate(candidates()[1],0,3)

    def test_union_and_fail_closed(self):
        self.assertEqual(whole_execution_failures([[1,1,0],[0,0,0]],3),(True,False))
        with self.assertRaises(ValueError): whole_execution_failures([[1,0]],3)
        with self.assertRaises(ValueError): whole_execution_failures([[2]],1)
        with self.assertRaises(TypeError): whole_execution_failures([True,False],1)
        req=Request('error','model',4,F(1,2),horizon=10)
        class BadSession:
            partial_feedback=True
            def evaluate(self,*args): raise RuntimeError('backend failed after unknown execution')
            def close(self): pass
        class BadBackend:
            def open(self,*args): return BadSession()
        out=validate(candidates(),req,BadBackend(),mode='lazy',chunk_size=3)
        self.assertEqual(out['rejection'],'backend-error')
        self.assertIsNone(out['selected'])
        self.assertFalse(out['executed_counts_exact'])
        self.assertEqual(out['attempted'],3)
        self.assertEqual(out['batch_calls'],1)
        with self.assertRaises(ValueError): validate(candidates(),req,bit_backend([[0]*10]*2),mode='B')
        with self.assertRaises(ValueError): validate(candidates(),Request('bad','model',1,F(1,2),10,F(1,20)),BadBackend())

    def test_bad_backend_lengths_and_close_fail_closed(self):
        req=Request('malformed','model',5,F(1,2),horizon=10)
        class Session:
            partial_feedback=True
            def evaluate(self,c,a,b): return Batch((False,))
            def close(self): pass
        class Backend:
            def open(self,*args): return Session()
        out=validate(candidates(),req,Backend(),mode='lazy',chunk_size=3)
        self.assertEqual(out['rejection'],'backend-error')
        self.assertFalse(out['executed_counts_exact'])
        self.assertEqual(out['executed'],0)
        class CloseSession:
            partial_feedback=False
            def evaluate(self,c,a,b): return Batch((False,)*(b-a))
            def close(self): raise RuntimeError('close failed')
        class CloseBackend:
            def open(self,*args): return CloseSession()
        out=validate(candidates(),req,CloseBackend())
        self.assertEqual(out['rejection'],'backend-error')
        self.assertIsNone(out['selected'])
        self.assertEqual(out['executed'],20)
        class OpenBackend:
            def open(self,*args): raise RuntimeError('factory failed')
        out=validate(candidates(),req,OpenBackend())
        self.assertEqual(out['rejection'],'backend-error')
        self.assertGreaterEqual(out['open_seconds'],0.)

    def test_impossible_boundary_and_large_chunks(self):
        req=Request('impossible','model',6,F(1,20),horizon=10)
        self.assertEqual(acceptance_threshold(10,req.epsilon,F(1,40)),-1)
        for streams in ([[False]*10]*2,[[True]*10]*2):
            fixed=validate(candidates(),req,bit_backend(streams))
            lazy=validate(candidates(),req,bit_backend(streams),mode='lazy',chunk_size=30)
            self.assertIsNone(fixed['selected']);self.assertIsNone(lazy['selected'])
            self.assertEqual(lazy['executed'],0)
            self.assertEqual(fixed['executed'],20)

    def test_exhaustive_smallN_oracle(self):
        req=Request('exhaustive','model',5,F(1,2),horizon=8)
        k=acceptance_threshold(8,req.epsilon,F(1,40))
        sequences=list(itertools.product((False,True),repeat=8))
        # Independent total-failure oracle for every stream paired with pass/fail.
        for stream in sequences:
            for other in ([False]*8,[True]*8):
                for cs in (candidates(),candidates((1.,1.)),candidates((2.,1.))):
                    bits=[stream,other]
                    passing=[c for c in cs if sum(bits[c.slot])<=k]
                    expected=min(passing,key=lambda c:(c.cost,c.slot)).slot if passing else None
                    result=validate(cs,req,bit_backend(bits),mode='lazy',chunk_size=3)
                    self.assertEqual(result['selected'],expected)

    def test_all_192_saved_streams(self):
        import numpy as np
        files=sorted((SOURCE/'phase9/outcomes').glob('t*.json'))
        self.assertEqual(len(files),96)
        for file in files:
            row=json.loads(file.read_text());streams=[]
            for slot,c in enumerate(row['candidates']):
                with np.load(SOURCE/'phase9/outcomes'/f"{row['key']}_c{slot}_admission.npz") as z:
                    n=int(z['n']);packed=z['flags'];self.assertEqual(n,1024)
                    # Explicit byte decode oracle, independent of unpackbits.
                    flags=[[bool(int(packed[i//8,j]) & (1<<(7-i%8))) for j in range(9)] for i in range(n)]
                stream=whole_execution_failures(flags,9)
                self.assertEqual(sum(stream),c['admission']['union']);streams.append(stream)
            cs=tuple(Candidate(s,c['admission']['cost'],bool(c['admission']['finite'] and c['admission']['nominal_command_valid']),c['admission']['path_hash']) for s,c in enumerate(row['candidates']))
            req=Request(row['key'],'phase9-declared-double-noise-K2',2026105200+row['task'],F(str(row['delta'])))
            factories=[]
            fixed=validate(cs,req,bit_backend(streams,factory_calls=factories))
            self.assertEqual(fixed['selected'],row['selected']);self.assertEqual(fixed['executed'],2048)
            self.assertEqual(len(factories),1)
            for partial,chunk in ((True,1),(True,16),(True,31),(True,1024),(False,16)):
                factories=[]
                out=validate(cs,req,bit_backend(streams,partial,factories),mode='lazy',chunk_size=chunk)
                self.assertEqual(out['selected'],row['selected'])
                self.assertEqual(out['rejection'],None if row['selected'] is not None else 'no-candidate-certified')
                self.assertEqual(len(factories),1)
                if out['certificate']:
                    self.assertEqual(out['certificate']['failures'],sum(streams[out['selected']]))
                    self.assertEqual(out['certificate']['n'],1024)


if __name__=='__main__':
    unittest.main()
