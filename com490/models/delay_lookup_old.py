"""
Query-time delay-distribution lookups + per-path confidence scoring.

Two lookup tables are consumed:

  - StopDelayLookup
        keyed by (line, bpuic, dow, hour)
        used when no real-time observation of the bus is available

  - SegmentTravelTimeLookup
        keyed by (line, source_bpuic, target_bpuic, dow, hour)
        used when the bus's current stop and current observed time
        are known: we sample a travel time and add it to the observed
        time to predict actual arrival at the transfer stop

A `bus_state_fn(trip_id, query_time_s) -> (curr_bpuic, curr_observed_time_s) | None`
is plugged in at scoring time. How it's implemented (Trino query against
istdaten with `arr_time <= query_time`, a pre-loaded snapshot, mocked for
tests) is the caller's choice — this module only consumes its output.

A `trip_to_line` dict maps timetable trip_id -> line string. Build it
once from `iceberg.sbb.trips JOIN iceberg.sbb.routes` filtered to the
relevant pub_date and pass it in.

dow follows Spark's F.dayofweek convention: 1=Sunday .. 7=Saturday.
Use `day_to_dow("monday")` to convert from a day name.
"""

# +
"""
Query-time delay-distribution lookups + per-path confidence scoring.

Stop delays use DISCRETE distributions (PMF over 1-minute buckets).
Transfer probability is computed via exact convolution — no Monte Carlo,
no quantile interpolation artifacts.

Segment travel times keep quantile-based distributions (7 quantiles).
When real-time bus position is known, travel time is sampled and compared
to the departure distribution via Monte Carlo.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd

# ── Segment TT quantile config (unchanged) ────────────────────────────────
QUANTILE_PROBS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
_QCOLS = [f'q{int(q*100):02d}' for q in QUANTILE_PROBS]

# ── Discrete delay bucket config (stop delays) ────────────────────────────
BUCKET_MIN_MIN = -10
BUCKET_MAX_MIN =  60
BUCKETS_MIN    = np.arange(BUCKET_MIN_MIN, BUCKET_MAX_MIN + 1, dtype=int)  # 71
BUCKETS_S      = BUCKETS_MIN * 60  # same in seconds

def _bucket_col(prefix, b):
    return f'{prefix}_m{abs(b):02d}' if b < 0 else f'{prefix}_p{b:02d}'

ARR_BUCKET_COLS = [_bucket_col('arr', b) for b in BUCKETS_MIN]
DEP_BUCKET_COLS = [_bucket_col('dep', b) for b in BUCKETS_MIN]

_DOW = {'sunday': 1, 'monday': 2, 'tuesday': 3, 'wednesday': 4,
        'thursday': 5, 'friday': 6, 'saturday': 7}

BusStateFn = Callable[[int, float], Optional[Tuple[int, float]]]


def day_to_dow(day_name: str) -> int:
    return _DOW[day_name.strip().lower()]


def _hour_of(seconds_since_midnight) -> int:
    return int(seconds_since_midnight // 3600) % 24


# ── Discrete delay distribution (stop delays) ─────────────────────────────

class DiscreteDelayDistribution:
    """
    Empirical delay distribution stored as a PMF over 1-minute buckets
    [BUCKET_MIN_MIN, BUCKET_MAX_MIN]. No interpolation, no tail clipping.
    """
    __slots__ = ('pmf', 'n')

    def __init__(self, pmf, n):
        self.pmf = np.asarray(pmf, dtype=float)
        s = self.pmf.sum()
        if s > 0:
            self.pmf = self.pmf / s
        self.n = int(n)

    @classmethod
    def zero(cls):
        """On-time: all mass at delay = 0."""
        pmf = np.zeros(len(BUCKETS_MIN))
        pmf[-BUCKET_MIN_MIN] = 1.0   # index of bucket 0
        return cls(pmf, 0)

    def sample(self, n_samples, rng=None):
        """Sample n_samples delay values in seconds (for real-time path)."""
        rng = rng or np.random.default_rng()
        return rng.choice(BUCKETS_S, size=n_samples, p=self.pmf).astype(float)

    def cdf(self, x_s):
        """P(delay ≤ x_s) — used for deadline check."""
        idx = int(np.floor(x_s / 60)) - BUCKET_MIN_MIN + 1
        idx = max(0, min(idx, len(self.pmf)))
        return float(self.pmf[:idx].sum())

    @property
    def median(self):
        idx = np.searchsorted(np.cumsum(self.pmf), 0.5)
        return float(BUCKETS_S[min(idx, len(BUCKETS_S) - 1)])


def _row_to_discrete(row, prefix, bucket_cols):
    pmf = np.array([row[c] for c in bucket_cols], dtype=float)
    return DiscreteDelayDistribution(pmf, n=row[f'{prefix}_n'])


# ── Quantile distribution (segment travel times) ──────────────────────────

class DelayDistribution:
    """
    Quantile-based distribution for segment travel times.
    Kept for the real-time bus-position path only.
    """
    __slots__ = ('q', 'n', 'mean', 'std')

    def __init__(self, quantile_values, n, mean=0.0, std=0.0):
        self.q    = np.asarray(quantile_values, dtype=float)
        self.n    = int(n)
        self.mean = float(mean)
        self.std  = float(std)

    @classmethod
    def zero(cls):
        return cls(np.zeros(len(QUANTILE_PROBS)), 0)

    def sample(self, n_samples, rng=None):
        rng = rng or np.random.default_rng()
        u = rng.uniform(size=n_samples)
        return np.interp(u, QUANTILE_PROBS, self.q)

    def cdf(self, x):
        return float(np.interp(x, self.q, QUANTILE_PROBS))

    @property
    def median(self):
        return float(self.q[3])


def _row_to_dist(row, prefix):
    qs = [row[f'{prefix}_{c}'] for c in _QCOLS]
    return DelayDistribution(
        qs, n=row[f'{prefix}_n'],
        mean=row.get(f'{prefix}_mean', 0.0) or 0.0,
        std=row.get(f'{prefix}_std', 0.0) or 0.0,
    )


# ── Lookups ───────────────────────────────────────────────────────────────

class StopDelayLookup:
    """
    Lookup of discrete per-stop delay distributions,
    keyed by (line, bpuic, dow, hour). Three-tier fallback.
    """
    def __init__(self, primary_path, fallback_no_line_path=None,
                 fallback_global_path=None, min_n=50):
        self._primary = (
            pd.read_parquet(primary_path)
              .set_index(['line', 'bpuic', 'dow', 'hour']).sort_index()
        )
        self._no_line = (
            pd.read_parquet(fallback_no_line_path)
              .set_index(['bpuic', 'dow', 'hour']).sort_index()
            if fallback_no_line_path else None
        )
        self._global = (
            pd.read_parquet(fallback_global_path)
              .set_index(['dow', 'hour']).sort_index()
            if fallback_global_path else None
        )
        self.min_n = min_n

    def arrival(self, line, bpuic, dow, hour) -> DiscreteDelayDistribution:
        row = self._row(line, bpuic, dow, hour, 'arr')
        return _row_to_discrete(row, 'arr', ARR_BUCKET_COLS) if row is not None \
               else DiscreteDelayDistribution.zero()

    def departure(self, line, bpuic, dow, hour) -> DiscreteDelayDistribution:
        row = self._row(line, bpuic, dow, hour, 'dep')
        return _row_to_discrete(row, 'dep', DEP_BUCKET_COLS) if row is not None \
               else DiscreteDelayDistribution.zero()

    def _row(self, line, bpuic, dow, hour, prefix):
        if line is not None:
            key = (str(line), int(bpuic), int(dow), int(hour))
            if key in self._primary.index:
                row = self._primary.loc[key]
                if row[f'{prefix}_n'] >= self.min_n:
                    return row
        if self._no_line is not None:
            key = (int(bpuic), int(dow), int(hour))
            if key in self._no_line.index:
                row = self._no_line.loc[key]
                if row[f'{prefix}_n'] >= self.min_n:
                    return row
        if self._global is not None:
            key = (int(dow), int(hour))
            if key in self._global.index:
                return self._global.loc[key]
        return None


class SegmentTravelTimeLookup:
    """
    Quantile-based travel-time lookup,
    keyed by (line, source_bpuic, target_bpuic, is_weekend, hour).
    """
    def __init__(self, primary_path, fallback_no_line_path=None, min_n=50):
        self._primary = (
            pd.read_parquet(primary_path)
              .set_index(['line', 'source_bpuic', 'target_bpuic', 'is_weekend', 'hour'])
              .sort_index()
        )
        self._no_line = (
            pd.read_parquet(fallback_no_line_path)
              .set_index(['source_bpuic', 'target_bpuic', 'is_weekend', 'hour']).sort_index()
            if fallback_no_line_path else None
        )
        self.min_n = min_n

    def get(self, line, source_bpuic, target_bpuic, dow, hour) -> Optional[DelayDistribution]:
        is_weekend = 1 if int(dow) in (1, 7) else 0
        if line is not None:
            key = (str(line), int(source_bpuic), int(target_bpuic), is_weekend, int(hour))
            if key in self._primary.index:
                row = self._primary.loc[key]
                if row['tt_n'] >= self.min_n:
                    return _row_to_dist(row, 'tt')
        if self._no_line is not None:
            key = (int(source_bpuic), int(target_bpuic), is_weekend, int(hour))
            if key in self._no_line.index:
                row = self._no_line.loc[key]
                if row['tt_n'] >= self.min_n:
                    return _row_to_dist(row, 'tt')
        return None


# ── Exact convolution (static path) ───────────────────────────────────────

def p_connection_discrete(arr_dist, dep_dist, slack_s, walk_s=0.0):
    """
    Exact P(connection) via convolution of two discrete delay distributions.

    Transfer succeeds when:
        arr_actual + walk_s <= dep_actual
    <=> arr_delay_min - dep_delay_min <= (slack_s - walk_s) / 60

    where slack_s = dep_sched_s - arr_sched_s  (seconds)
          walk_s  = walking time between stops  (seconds)

    Uses np.convolve(arr_pmf, dep_pmf[::-1]):
        result[k] = P[arr_delay - dep_delay = k - (n-1)] minutes
    """
    slack_min = (slack_s - walk_s) / 60.0
    n = len(BUCKETS_MIN)            # 71
    diff_pmf = np.convolve(arr_dist.pmf, dep_dist.pmf[::-1])
    cutoff = int(np.floor(slack_min)) + n
    cutoff = max(0, min(cutoff, len(diff_pmf)))
    return float(diff_pmf[:cutoff].sum())


# ── Per-path confidence ────────────────────────────────────────────────────

def path_confidence(events,
                    stop_delays,
                    travel_times,
                    trip_to_line,
                    dow,
                    query_time_s,
                    deadline_s=None,
                    buffer_s=0.0,
                    bus_state_fn=None,
                    n_samples=10000,
                    rng=None):
    """
    End-to-end confidence for a path from Router.reconstruct_route.

    Static path (no real-time info):
        Uses discrete PMF distributions + exact convolution.
        Deterministic — no Monte Carlo variance.

    Real-time path (bus position known + segment TT available):
        Uses quantile TT distribution sampled via Monte Carlo,
        combined with discrete departure distribution.
    """
    rng = rng or np.random.default_rng()
    overall = 1.0
    breakdown = []
    last_alight = None
    walk_accum_s = 0.0
    final_alight = None
    final_sched_ts = None

    for i, ev in enumerate(events):
        ts, x, y, z = ev
        is_alight = (x is None  and y is not None and z is not None)
        is_board  = (x is not None and y is not None and z is None)
        is_walk   = (y is None  and z is not None)

        if is_alight:
            last_alight   = {'trip_id': y, 'line': trip_to_line.get(y),
                             'stop': z, 'ts': ts}
            walk_accum_s  = 0.0
            final_alight  = last_alight
            final_sched_ts = ts

        elif is_walk:
            prev_ts = events[i - 1][0] if i > 0 else ts
            walk_accum_s += (ts - prev_ts)
            final_sched_ts = ts

        elif is_board and last_alight is not None:
            board_trip = y
            board_line = trip_to_line.get(board_trip)
            board_stop = x

            rt_state = bus_state_fn(last_alight['trip_id'], query_time_s) \
                       if bus_state_fn is not None else None

            if rt_state is not None:
                curr_bpuic, curr_time_s = rt_state
                tt = travel_times.get(
                    last_alight['line'], curr_bpuic,
                    last_alight['stop'], dow, _hour_of(curr_time_s)
                )
                if tt is not None:
                    # Real-time path: Monte Carlo
                    arr_samples = curr_time_s + tt.sample(n_samples, rng) + walk_accum_s
                    dep_dist    = stop_delays.departure(
                        board_line, board_stop, dow, _hour_of(ts))
                    dep_samples = ts + dep_dist.sample(n_samples, rng)
                    p = float(np.mean(arr_samples + buffer_s <= dep_samples))
                else:
                    rt_state = None   # fall through to static path

            if rt_state is None:
                # Static path: exact convolution
                slack_s  = float(ts - last_alight['ts'])
                arr_dist = stop_delays.arrival(
                    last_alight['line'], last_alight['stop'],
                    dow, _hour_of(last_alight['ts']))
                dep_dist = stop_delays.departure(
                    board_line, board_stop, dow, _hour_of(ts))
                p = p_connection_discrete(
                    arr_dist, dep_dist, slack_s,
                    walk_s=walk_accum_s + buffer_s)

            breakdown.append({
                'type':          'transfer',
                'incoming_trip': last_alight['trip_id'],
                'incoming_line': last_alight['line'],
                'alight_stop':   last_alight['stop'],
                'alight_ts':     last_alight['ts'],
                'walk_s':        walk_accum_s,
                'outgoing_trip': board_trip,
                'outgoing_line': board_line,
                'board_stop':    board_stop,
                'board_ts':      ts,
                'p':             p,
            })
            overall      *= p
            last_alight   = None
            walk_accum_s  = 0.0

    if deadline_s is not None and final_alight is not None:
        walk_to_dest = float(final_sched_ts - final_alight['ts'])
        slack_s      = deadline_s - final_alight['ts'] - walk_to_dest
        arr_dist     = stop_delays.arrival(
            final_alight['line'], final_alight['stop'],
            dow, _hour_of(final_alight['ts']))
        p_final      = arr_dist.cdf(slack_s)
        breakdown.append({
            'type':         'arrival',
            'final_trip':   final_alight['trip_id'],
            'final_line':   final_alight['line'],
            'final_stop':   final_alight['stop'],
            'sched_arr_ts': final_sched_ts,
            'deadline_s':   deadline_s,
            'p':            p_final,
        })
        overall *= p_final

    return overall, breakdown
# -


