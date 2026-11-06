"""
Build the two empirical-distribution parquet tables consumed by
com490.models.delay_lookup at query time.

There are two tables, produced by two functions:

  1. build_stop_delays(...)
       Per-(line, bpuic, dow, hour) empirical distribution of delay
       at a stop in seconds (arr_actual - arr_time, with departure
       symmetric). Used at query time when we have no real-time
       observation of the bus on this leg (it has not yet started,
       or sparse upstream data): we sample a delay and add it to
       the scheduled arrival/departure at the transfer stop.

  2. build_segment_travel_times(...)
       Per-(line, source_bpuic, target_bpuic, dow, hour) empirical
       distribution of actual travel time in seconds, computed as
       arr_actual(target) - dep_actual(source) within the same
       (trip_id, operating_day). Used at query time when we *do*
       have a real-time observation: we sample a travel time and
       add it to the bus's current observed time to predict its
       arrival at the transfer stop.

Conventions
-----------
- dow follows Spark's F.dayofweek: 1=Sunday .. 7=Saturday.
- hour is the F.hour of the source-side scheduled time
  (arr_time for stop delays, dep_time for segment travel times).
- line is the istdaten `line_text` column (which equals
  timetable `route_short_name` for the Lausanne dataset).
"""

# +
"""
Build the two empirical-distribution parquet tables consumed by
com490.models.delay_lookup at query time.

  1. build_stop_delays(...)
       Per-(line, bpuic, dow, hour) DISCRETE distribution of delay
       stored as a probability mass function over 1-minute buckets
       [-10min, +60min]. Exact frequencies, no quantile interpolation.

  2. build_segment_travel_times(...)
       Per-(line, source_bpuic, target_bpuic, is_weekend, hour)
       empirical distribution of actual travel time stored as 7
       quantiles (P05..P95). Used with CDF lookup at query time.
"""
import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F

# ── Discrete bucket config (stop delays only) ──────────────────────────────
BUCKET_MIN_MIN = -10   # minutes
BUCKET_MAX_MIN =  60   # minutes
BUCKETS        = list(range(BUCKET_MIN_MIN, BUCKET_MAX_MIN + 1))  # 71 buckets

def _bucket_col(prefix, b):
    return f'{prefix}_m{abs(b):02d}' if b < 0 else f'{prefix}_p{b:02d}'

def _delay_seconds(actual_col, scheduled_col):
    return (
        F.col(actual_col).cast('timestamp').cast('long')
        - F.col(scheduled_col).cast('timestamp').cast('long')
    )

def _bucket_agg_exprs(delay_col, prefix):
    """Count observations per 1-minute bucket using conditional sums."""
    exprs = [
        F.count(F.when(F.col(delay_col).isNotNull(), 1)).alias(f'{prefix}_n')
    ]
    exprs += [
        F.sum(
            F.when(F.round(F.col(delay_col) / 60).cast('int') == b, 1).otherwise(0)
        ).alias(_bucket_col(prefix, b))
        for b in BUCKETS
    ]
    return exprs

# ── Quantile config (segment travel times only) ────────────────────────────
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

def _agg_exprs_quantile(prefix, col):
    exprs = [
        F.count(F.when(F.col(col).isNotNull(), 1)).alias(f'{prefix}_n'),
        F.mean(col).alias(f'{prefix}_mean'),
        F.stddev(col).alias(f'{prefix}_std'),
    ]
    exprs += [
        F.expr(f'percentile_approx({col}, {q})').alias(f'{prefix}_q{int(q*100):02d}')
        for q in QUANTILES
    ]
    return exprs


# ── Table 1: per-stop DISCRETE delay distributions ─────────────────────────

def _normalize_buckets(df, prefixes, min_n):
    """Normalize bucket counts to probabilities and filter sparse bins."""
    df = df.filter(
        F.greatest(*[F.col(f'{p}_n') for p in prefixes]) >= min_n
    )
    for prefix in prefixes:
        total = F.col(f'{prefix}_n').cast('double')
        for b in BUCKETS:
            col = _bucket_col(prefix, b)
            df = df.withColumn(col, F.col(col) / total)
    return df


def build_stop_delays(spark,
                      istdaten_parquet_path,
                      output_path,
                      fallback_no_dow_path=None,
                      fallback_no_line_path=None,
                      min_n=50,
                      delay_clip_min=(-10, 60)):
    """
    Three-tier hierarchy:
      primary             : (line, stop, dow, hour)
      fallback_no_dow     : (line, stop, hour)    ← drop dow
      fallback_no_line    : (stop, hour)          ← drop line
    """
    lo, hi = delay_clip_min
    df = (
        spark.read.parquet(istdaten_parquet_path)
        .withColumn('arr_delay', _delay_seconds('arr_actual', 'arr_time'))
        .withColumn('dep_delay', _delay_seconds('dep_actual', 'dep_time'))
        .withColumn('dow',  F.dayofweek('operating_day').cast('int'))
        .withColumn('hour', F.hour('arr_time').cast('int'))
        .filter(
            F.round(F.col('arr_delay') / 60).cast('int').between(lo, hi) &
            F.round(F.col('dep_delay') / 60).cast('int').between(lo, hi)
        )
    )

    aggs = _bucket_agg_exprs('arr_delay', 'arr') + _bucket_agg_exprs('dep_delay', 'dep')

    primary = _normalize_buckets(
        df.groupBy('line_text', 'bpuic', 'dow', 'hour').agg(*aggs),
        prefixes=['arr', 'dep'], min_n=min_n
    ).withColumnRenamed('line_text', 'line')
    primary.write.parquet(output_path, mode='overwrite')

    if fallback_no_dow_path is not None:
        _normalize_buckets(
            df.groupBy('line_text', 'bpuic', 'hour').agg(*aggs),     # ← drop dow
            prefixes=['arr', 'dep'], min_n=min_n
        ).withColumnRenamed('line_text', 'line') \
         .write.parquet(fallback_no_dow_path, mode='overwrite')

    if fallback_no_line_path is not None:
        _normalize_buckets(
            df.groupBy('bpuic', 'hour').agg(*aggs),                  # ← drop line too
            prefixes=['arr', 'dep'], min_n=min_n
        ).write.parquet(fallback_no_line_path, mode='overwrite')


# ── Table 1.2: per-stop GUMBEL delay distributions ─────────────────────────

def _gumbel_param_exprs(col_name, prefix):
    """
    Returns aggregation expressions to estimate Gumbel parameters
    using the Method of Moments.
    """
    euler_mascheroni = 0.5772156649
    sqrt_6_over_pi = np.sqrt(6) / np.pi  # ~0.7796968
    
    mean_col = F.mean(col_name)
    std_col = F.stddev(col_name)
    count_col = F.count(col_name)
    
    # Scale parameter beta = (stddev * sqrt(6)) / pi
    beta_expr = std_col * sqrt_6_over_pi
    
    return [
        count_col.alias(f'{prefix}_n'),
        mean_col.alias(f'{prefix}_mean'),
        std_col.alias(f'{prefix}_std'),
        beta_expr.alias(f'{prefix}_beta'),
        # Location parameter mu = mean - gamma * beta
        (mean_col - (beta_expr * euler_mascheroni)).alias(f'{prefix}_mu')
    ]

def _filter_and_clean_estimates(df, prefixes, min_n):
    """
    Filters rows with sufficient data and handles fallback edge cases 
    (e.g., variance is 0 or negative due to numerical instability).
    """
    for pfx in prefixes:
        # A valid Gumbel scale parameter must be > 0. 
        # If stddev is 0 (or null), we fallback or fill with a tiny value/null.
        df = df.filter(F.col(f'{pfx}_n') >= min_n).withColumn(
            f'{pfx}_beta', 
            F.when(F.col(f'{pfx}_beta') > 0, F.col(f'{pfx}_beta')).otherwise(F.lit(1e-4))
        )
    return df

def build_stop_delays_gumbel(spark,
                             istdaten_parquet_path,
                             output_path,
                             fallback_no_dow_path=None,
                             fallback_no_line_path=None,
                             min_n=50,
                             delay_clip_min=(-10, 60)):
    """
    Three-tier hierarchy:
      primary             : (line, stop, dow, hour)
      fallback_no_dow     : (line, stop, hour)    ← drop dow
      fallback_no_line    : (stop, hour)          ← drop line
    """
    lo, hi = delay_clip_min

    # 1. Read and preprocess transit data (keeping raw seconds for precision)
    df = (
        spark.read.parquet(istdaten_parquet_path)
        .withColumn('arr_delay', _delay_seconds('arr_actual', 'arr_time'))
        .withColumn('dep_delay', _delay_seconds('dep_actual', 'dep_time'))
        .withColumn('dow',  F.dayofweek('operating_day').cast('int'))
        .withColumn('hour', F.hour('arr_time').cast('int'))
        .filter(
            F.round(F.col('arr_delay') / 60).cast('int').between(lo, hi) &
            F.round(F.col('dep_delay') / 60).cast('int').between(lo, hi)
        )
    )

    # 2. Define the new Gumbel aggregation expressions
    gumbel_aggs = _gumbel_param_exprs('arr_delay', 'arr') + _gumbel_param_exprs('dep_delay', 'dep')

    # 3. Primary Granularity: Line + Stop + DayOfWeek + Hour
    primary = df.groupBy('line_text', 'bpuic', 'dow', 'hour').agg(*gumbel_aggs)
    primary = _filter_and_clean_estimates(primary, prefixes=['arr', 'dep'], min_n=min_n)
    primary = primary.withColumnRenamed('line_text', 'line')
    primary.write.parquet(output_path, mode='overwrite')

    # 4. Fallback 1: Line + Stop + Hour (drop dow)
    if fallback_no_dow_path is not None:
        fallback_no_dow = df.groupBy('line_text', 'bpuic', 'hour').agg(*gumbel_aggs)
        fallback_no_dow = _filter_and_clean_estimates(fallback_no_dow, prefixes=['arr', 'dep'], min_n=min_n)
        fallback_no_dow = fallback_no_dow.withColumnRenamed('line_text', 'line')
        fallback_no_dow.write.parquet(fallback_no_dow_path, mode='overwrite')

    # 5. Fallback 2: Stop + Hour (drop line too)
    if fallback_no_line_path is not None:
        fallback_no_line = df.groupBy('bpuic', 'hour').agg(*gumbel_aggs)
        fallback_no_line = _filter_and_clean_estimates(fallback_no_line, prefixes=['arr', 'dep'], min_n=min_n)
        fallback_no_line.write.parquet(fallback_no_line_path, mode='overwrite')

# ── Table 2: per-segment QUANTILE travel-time distributions ────────────────

def build_segment_travel_times(spark,
                               istdaten_parquet_path,
                               output_path,
                               fallback_no_line_path=None,
                               min_n=50,
                               max_stop_gap=20,
                               travel_time_clip=(10, 7200)):
    lo, hi = travel_time_clip
    df = spark.read.parquet(istdaten_parquet_path)

    trip_window = Window.partitionBy('operating_day', 'trip_id').orderBy('arr_time')
    df = df.withColumn('stop_seq', F.row_number().over(trip_window))

    a = df.alias('a')
    b = df.alias('b')

    pairs = (
        a.join(b, [
            F.col('a.trip_id')       == F.col('b.trip_id'),
            F.col('a.operating_day') == F.col('b.operating_day'),
            F.col('a.stop_seq')       < F.col('b.stop_seq'),
            (F.col('b.stop_seq') - F.col('a.stop_seq')) <= max_stop_gap,
        ])
        .filter(F.col('a.dep_actual').isNotNull() & F.col('b.arr_actual').isNotNull())
        .withColumn('travel_time',
            F.col('b.arr_actual').cast('timestamp').cast('long') -
            F.col('a.dep_actual').cast('timestamp').cast('long')
        )
        .filter(F.col('travel_time').between(lo, hi))
        .select(
            F.col('a.line_text').alias('line'),
            F.col('a.bpuic').alias('source_bpuic'),
            F.col('b.bpuic').alias('target_bpuic'),
            F.when(F.dayofweek(F.col('a.operating_day')).isin([1, 7]), 1)
             .otherwise(0).cast('int').alias('is_weekend'),
            F.hour(F.col('a.dep_time')).cast('int').alias('hour'),
            'travel_time',
        )
    )

    aggs = _agg_exprs_quantile('tt', 'travel_time')

    primary = (
        pairs.groupBy('line', 'source_bpuic', 'target_bpuic', 'is_weekend', 'hour')
             .agg(*aggs)
             .filter(F.col('tt_n') >= min_n)
    )
    primary.write.parquet(output_path, mode='overwrite')

    if fallback_no_line_path is not None:
        (
            pairs.groupBy('source_bpuic', 'target_bpuic', 'is_weekend', 'hour')
                 .agg(*aggs)
                 .filter(F.col('tt_n') >= min_n)
                 .write.parquet(fallback_no_line_path, mode='overwrite')
        )
# -