"""Opt-in decision-equivalent fixed-horizon lazy validation; standard library only.
No planner, robot, RNG, GPU, or sequential B/C implementation is imported here.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import math
import time
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class Candidate:
    slot: int
    cost: float
    eligible: bool
    path_hash: str
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Request:
    request_id: str
    model_id: str
    seed: int
    epsilon: Fraction
    horizon: int = 1024
    alpha_per_candidate: Fraction = Fraction(1, 40)


@dataclass(frozen=True)
class Batch:
    """Exactly one whole-execution union failure per actually executed trial."""
    failures: tuple[bool, ...]


class Session(Protocol):
    partial_feedback: bool

    def evaluate(self, candidate: Candidate, start: int, stop: int) -> Batch: ...
    def close(self) -> None: ...


class Backend(Protocol):
    def open(self, request: Request, candidates: tuple[Candidate, ...]) -> Session: ...


@dataclass(frozen=True)
class Bank:
    """Full-horizon world draws. Payload layout belongs to the backend."""
    payload: Any
    sample_count: int


def whole_execution_failures(event_rows, event_count: int) -> tuple[bool, ...]:
    """Reduce columns, never flatten timesteps or count correlated events as trials."""
    failures = []
    if event_count <= 0:
        raise ValueError('event_count must be positive')
    for row in event_rows:
        values = tuple(row)
        if len(values) != event_count or any(v not in (0, 1) for v in values):
            raise ValueError('Expected one row of declared binary events per whole execution')
        failures.append(any(values))
    return tuple(failures)


class FrozenBankBackend:
    """Generate once per request at the original horizon, then slice without RNG.

    factory(request)->Bank; slice_bank(payload,start,stop)->draw slice;
    evaluate_events(candidate,draw_slice)->rows of whole-execution event columns.
    Callback implementations own physical validation and synchronization.
    Set partial_feedback=False when intermediate feedback is unavailable/expensive.
    """
    def __init__(self, factory: Callable, slice_bank: Callable,
                 evaluate_events: Callable, *, event_count: int,
                 partial_feedback: bool = False):
        self.factory = factory
        self.slice_bank = slice_bank
        self.evaluate_events = evaluate_events
        self.event_count = event_count
        self.partial_feedback = partial_feedback

    def open(self, request, candidates):
        bank = self.factory(request)
        if not isinstance(bank, Bank) or bank.sample_count != request.horizon:
            raise ValueError('Factory must supply the full original-horizon bank')
        backend = self
        class FrozenSession:
            partial_feedback = backend.partial_feedback
            def __init__(self):
                self.closed = False
                # Candidate prefixes share worlds, not mutable stream cursors.
                self.next_index = {}

            def evaluate(self, candidate, start, stop):
                if self.closed:
                    raise RuntimeError('Session already closed')
                if start != self.next_index.get(candidate.slot, 0):
                    raise ValueError('Repeated, skipped or reordered sample indices')
                if not 0 <= start < stop <= request.horizon:
                    raise ValueError('Sample slice outside frozen horizon')
                draw_slice = backend.slice_bank(bank.payload, start, stop)
                rows = backend.evaluate_events(candidate, draw_slice)
                flags = whole_execution_failures(rows, backend.event_count)
                if len(flags) != stop-start:
                    raise ValueError('Backend returned wrong whole-execution trial count')
                self.next_index[candidate.slot] = stop
                return Batch(flags)

            def close(self):
                self.closed = True
        return FrozenSession()


@lru_cache(maxsize=64)
def acceptance_threshold(horizon: int, epsilon: Fraction, alpha: Fraction) -> int:
    """Largest k with BinomialCDF(k;N,epsilon)<=alpha, exact rational comparison.

    Equivalent to the one-sided fixed-N Clopper-Pearson upper test. No floating
    tolerance can admit excess null mass. -1 means certification impossible.
    """
    p = Fraction(epsilon)
    alpha = Fraction(alpha)
    if horizon <= 0 or not 0 < p < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid horizon, epsilon or alpha')
    a, b = p.numerator, p.denominator
    denominator = b**horizon
    mass = (b-a)**horizon
    cumulative, boundary = 0, -1
    for k in range(horizon+1):
        cumulative += mass
        if cumulative*alpha.denominator > alpha.numerator*denominator:
            break
        boundary = k
        if k < horizon:
            mass = mass*(horizon-k)*a//((k+1)*(b-a))
    return boundary


def validate(candidates: Sequence[Candidate], request: Request, backend: Backend,
             *, mode: str = 'fixed', chunk_size: int = 16) -> dict:
    """Return fixed-N certificate/selection and accounting, never a stopped CP CI.

    fixed (default): fully evaluate both slots even if metadata ineligible.
    lazy (opt-in): cost/slot order; stop on irreversible fixed-N rejection;
    certify only at N; skip later slots after preferred candidate certified.
    If partial feedback is false, lazy executes one full-N batch per candidate.
    Backend exceptions fail closed with partial accounting; no candidate returned.
    """
    began = time.perf_counter()
    candidates = tuple(candidates)
    if mode not in ('fixed', 'lazy') or chunk_size <= 0:
        raise ValueError('mode must be fixed/lazy and chunk_size positive')
    if len(candidates) != 2 or sorted(c.slot for c in candidates) != [0, 1]:
        raise ValueError('Exactly two fixed slots 0 and 1 are required')
    if any(math.isnan(c.cost) or type(c.eligible) is not bool for c in candidates):
        raise ValueError('Costs must be ordered (not NaN); eligibility must be explicit bool')
    if any(not c.path_hash for c in candidates):
        raise ValueError('Frozen candidate path hashes are required')
    if type(request.horizon) is not int or request.horizon <= 0:
        raise ValueError('Horizon must be a positive integer')
    if not request.request_id or not request.model_id:
        raise ValueError('Request and declared-model identities are required')
    epsilon = Fraction(str(request.epsilon))
    alpha = Fraction(str(request.alpha_per_candidate))
    if not 0 < alpha <= Fraction(1,40):
        raise ValueError('Two-candidate alpha must not exceed .025')
    limit = acceptance_threshold(request.horizon, epsilon, alpha)
    records = {c.slot: dict(slot=c.slot, cost=c.cost, eligible=c.eligible,
                           path_hash=c.path_hash, status='unresolved', examined=0,
                           executed=0, attempted=0, failures=0, batch_calls=0, backend_seconds=0.)
               for c in candidates}
    output = dict(request_id=request.request_id, model_id=request.model_id,
                  seed=request.seed, horizon=request.horizon, epsilon=str(epsilon),
                  alpha_per_candidate=str(alpha), mode=mode, selected=None,
                  selected_cost=None, rejection=None, certificate=None,
                  candidates=[], examined=0, executed=0, batch_calls=0,
                  open_seconds=0., backend_seconds=0., close_seconds=0.,
                  threshold_seconds=time.perf_counter()-began, executed_counts_exact=True)
    session, admitted = None, []
    try:
        opened = time.perf_counter()
        try:
            session = backend.open(request, candidates)
        finally:
            output['open_seconds'] = time.perf_counter()-opened
        partial = bool(session.partial_feedback)
        output['feedback'] = 'chunked' if mode == 'lazy' and partial else 'full-horizon'
        step = chunk_size if mode == 'lazy' and partial else request.horizon
        ordered = sorted(candidates, key=lambda c: c.slot if mode == 'fixed' else (c.cost,c.slot))
        for candidate in ordered:
            record = records[candidate.slot]
            if mode == 'lazy' and not candidate.eligible:
                record['status'] = 'ineligible'
                continue
            if mode == 'lazy' and limit < 0:
                record['status'] = 'futility'
                continue
            for start in range(0, request.horizon, step):
                stop = min(start+step, request.horizon)
                tick = time.perf_counter()
                record['batch_calls'] += 1
                record['attempted'] += stop-start
                try:
                    batch = session.evaluate(candidate, start, stop)
                finally:
                    record['backend_seconds'] += time.perf_counter()-tick
                if not isinstance(batch, Batch) or len(batch.failures) != stop-start:
                    raise ValueError('Batch must return every actually executed trial in this slice')
                if any(type(flag) is not bool for flag in batch.failures):
                    raise ValueError('Batch failures must be whole-execution boolean union bits')
                record['executed'] += stop-start
                for flag in batch.failures:
                    record['examined'] += 1
                    record['failures'] += int(flag)
                    if mode == 'lazy' and record['failures'] > limit:
                        record['status'] = 'futility'
                        break
                if record['status'] == 'futility':
                    break
            if record['status'] == 'unresolved':
                if not candidate.eligible:
                    record['status'] = 'ineligible'
                elif record['failures'] <= limit:
                    record['status'] = 'certified'
                    admitted.append(candidate)
                else:
                    record['status'] = 'fixed-N-rejection'
            if mode == 'lazy' and admitted:
                break
        if admitted:
            winner = min(admitted, key=lambda c:(c.cost,c.slot))
            record = records[winner.slot]
            output.update(selected=winner.slot, selected_cost=winner.cost,
                          certificate=dict(test='fixed-N-one-sided-CP', n=request.horizon,
                                           failures=record['failures'], boundary=limit,
                                           epsilon=str(epsilon), alpha=str(alpha),
                                           slot=winner.slot, path_hash=winner.path_hash,
                                           model_id=request.model_id, seed=request.seed,
                                           sample_indices=[0,request.horizon]))
        else:
            output['rejection'] = 'no-candidate-certified'
    except Exception as error:
        output.update(selected=None, selected_cost=None, certificate=None,
                      rejection='backend-error', error=repr(error), executed_counts_exact=False)
    finally:
        if session is not None:
            tick = time.perf_counter()
            try:
                session.close()
            except Exception as error:
                output.update(selected=None, selected_cost=None, certificate=None,
                              rejection='backend-error', close_error=repr(error))
            output['close_seconds'] = time.perf_counter()-tick
    output['candidates'] = [records[s] for s in (0,1)]
    for metric in ('examined','executed','attempted','batch_calls','backend_seconds'):
        output[metric] = sum(r[metric] for r in records.values())
    output['total_seconds'] = time.perf_counter()-began
    output['selection_and_adapter_seconds'] = max(0.,output['total_seconds']-output['open_seconds']-output['backend_seconds']-output['close_seconds'])
    return output
