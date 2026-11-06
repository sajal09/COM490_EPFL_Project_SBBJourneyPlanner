# +
"""
Feature importance analysis for delay distribution grouping keys.

Trains a lightweight Decision Tree Regressor on a sample of istdaten
to determine which temporal features best explain delay variance.
Results inform the grouping keys used in build_stop_delays and
build_segment_travel_times.

Features tested:
  - hour       : raw hour of day (0-23), 24 possible values
  - is_rush_hour: binary — hours 7-9 and 17-19
  - dow        : day of week (1-7), 7 possible values
  - is_weekend : binary — Saturday/Sunday
  - month      : month of year (1-12), captures seasonal patterns
"""

from pyspark.sql import Window
from pyspark.sql import functions as F
import pandas as pd


def _build_stop_delay_sample(spark, istdaten_parquet_path,
                              sample_fraction=0.1,
                              delay_clip=(-600, 3600)):
    lo, hi = delay_clip
    return (
        spark.read.parquet(istdaten_parquet_path)
        .withColumn('delay',
            F.col('arr_actual').cast('timestamp').cast('long') -
            F.col('arr_time').cast('timestamp').cast('long')
        )
        .filter(F.col('delay').between(lo, hi))
        .withColumn('hour',         F.hour('arr_time').cast('int'))
        .withColumn('dow',          F.dayofweek('operating_day').cast('int'))
        .withColumn('month',        F.month('operating_day').cast('int'))
        .withColumn('is_weekend',   F.when(
            F.dayofweek('operating_day').isin([1, 7]), 1).otherwise(0))
        .withColumn('is_rush_hour', F.when(
            F.hour('arr_time').between(7, 9) |
            F.hour('arr_time').between(17, 19), 1).otherwise(0))
        .sample(fraction=sample_fraction, seed=42)
        .select('hour', 'dow', 'month', 'is_weekend', 'is_rush_hour', 'delay')
        .dropna()
        .toPandas()
    )


def _build_segment_tt_sample(spark, istdaten_parquet_path,
                              sample_fraction=0.1,
                              travel_time_clip=(10, 7200),
                              max_stop_gap=3):
    lo, hi = travel_time_clip
    df = spark.read.parquet(istdaten_parquet_path)

    trip_window = Window.partitionBy('operating_day', 'trip_id').orderBy('arr_time')
    df = df.withColumn('stop_seq', F.row_number().over(trip_window))

    a = df.alias('a')
    b = df.alias('b')

    return (
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
        .withColumn('hour',         F.hour(F.col('a.dep_time')).cast('int'))
        .withColumn('dow',          F.dayofweek(F.col('a.operating_day')).cast('int'))
        .withColumn('month',        F.month(F.col('a.operating_day')).cast('int'))
        .withColumn('is_weekend',   F.when(
            F.dayofweek(F.col('a.operating_day')).isin([1, 7]), 1).otherwise(0))
        .withColumn('is_rush_hour', F.when(
            F.hour(F.col('a.dep_time')).between(7, 9) |
            F.hour(F.col('a.dep_time')).between(17, 19), 1).otherwise(0))
        .sample(fraction=sample_fraction, seed=42)
        .select('hour', 'dow', 'month', 'is_weekend', 'is_rush_hour', 'travel_time')
        .dropna()
        .toPandas()
    )


def _run_dt(pdf, target_col, feature_cols, max_depth=6):
    from sklearn.tree import DecisionTreeRegressor
    X = pdf[feature_cols].values
    y = pdf[target_col].values
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X, y)
    return (
        pd.DataFrame({'feature': feature_cols, 'importance': dt.feature_importances_})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


TEMPORAL_FEATURES = ['hour', 'is_rush_hour', 'dow', 'is_weekend', 'month']


def analyze_delay_features(spark, istdaten_parquet_path, sample_fraction=0.1):
    """
    Run DT feature importance on both stop delays and segment travel times.
    Returns (fi_stop, fi_seg) DataFrames with feature importances.

    Interpretation guide
    --------------------
    importance > 0.20 : strong signal, keep in grouping key
    importance 0.05-0.20 : moderate, worth keeping if bins stay large enough
    importance < 0.05 : weak, consider dropping to increase bin sizes
    """
    print('=== Stop delay feature importance ===')
    print(f'Sampling {sample_fraction:.0%} of istdaten...')
    stop_pdf = _build_stop_delay_sample(spark, istdaten_parquet_path, sample_fraction)
    print(f'  {len(stop_pdf):,} rows')
    fi_stop = _run_dt(stop_pdf, 'delay', TEMPORAL_FEATURES)
    print(fi_stop.to_string(index=False))

    print('\n=== Segment travel time feature importance ===')
    print(f'Sampling {sample_fraction:.0%} of istdaten...')
    seg_pdf = _build_segment_tt_sample(spark, istdaten_parquet_path, sample_fraction)
    print(f'  {len(seg_pdf):,} rows')
    fi_seg = _run_dt(seg_pdf, 'travel_time', TEMPORAL_FEATURES)
    print(fi_seg.to_string(index=False))

    return fi_stop, fi_seg

# ── New rigorous version: gradient boosting + permutation importance ──
# Tests temporal, spatial, and interaction features. No leakage.

def analyze_delay_features_full(spark, istdaten_parquet_path,
                                 sample_fraction=0.05,
                                 test_size=0.3,
                                 n_perm_repeats=5,
                                 random_state=42):
    """
    Train a HistGradientBoostingRegressor on a sample of istdaten with
    raw + target-encoded + interaction features, then compute permutation
    importance on the held-out test set.

    Returns
    -------
    pd.DataFrame [feature, importance] sorted descending.
    """
    import numpy as np
    import pandas as pd
    import pyspark.sql.functions as F
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print(f"Sampling {sample_fraction:.0%} of istdaten…")
    df = (
        spark.read.parquet(istdaten_parquet_path)
        .withColumn('delay',
            F.col('arr_actual').cast('timestamp').cast('long') -
            F.col('arr_time').cast('timestamp').cast('long'))
        .filter(F.col('delay').between(-600, 3600))
        .withColumn('hour',         F.hour('arr_time').cast('int'))
        .withColumn('dow',          F.dayofweek('operating_day').cast('int'))
        .withColumn('month',        F.month('operating_day').cast('int'))
        .withColumn('is_weekend',   F.when(F.dayofweek('operating_day').isin([1, 7]), 1).otherwise(0))
        .withColumn('is_rush_hour', F.when(
            F.hour('arr_time').between(7, 9) | F.hour('arr_time').between(17, 19), 1).otherwise(0))
        .sample(fraction=sample_fraction, seed=random_state)
        .select('line_text', 'bpuic',
                'hour', 'dow', 'month', 'is_weekend', 'is_rush_hour',
                'delay')
        .dropna()
        .toPandas()
    )
    df['line_enc'] = LabelEncoder().fit_transform(df['line_text'])
    print(f"  Rows: {len(df):,}")

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    def encode(train, test, group_cols, target='delay', smoothing=20):
        g_mean = train[target].mean()
        stats = train.groupby(group_cols)[target].agg(['mean', 'count']).reset_index()
        stats['enc'] = (stats['mean'] * stats['count'] + g_mean * smoothing) / (stats['count'] + smoothing)
        name = 'enc_' + '_'.join(group_cols)
        keep = group_cols + ['enc']
        tr = train.merge(stats[keep], on=group_cols, how='left').rename(columns={'enc': name})
        te = test .merge(stats[keep], on=group_cols, how='left').rename(columns={'enc': name})
        te[name] = te[name].fillna(g_mean)
        return tr, te, name

    for gcols in [['bpuic'], ['line_text'],
                  ['bpuic', 'hour'], ['line_text', 'hour'], ['dow', 'hour']]:
        train, test, _ = encode(train, test, gcols)

    features = [
        'enc_bpuic_hour', 'enc_line_text_hour', 'enc_dow_hour',
        'enc_bpuic', 'enc_line_text',
        'hour', 'dow', 'month', 'is_weekend', 'is_rush_hour',
        'line_enc', 'bpuic',
    ]

    print("Training HistGradientBoostingRegressor…")
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.08, max_depth=6, random_state=random_state,
    )
    model.fit(train[features], train['delay'])
    print(f"  R² (test): {model.score(test[features], test['delay']):.3f}")

    print("Computing permutation importance…")
    perm = permutation_importance(
        model, test[features], test['delay'],
        n_repeats=n_perm_repeats, random_state=random_state, n_jobs=-1,
    )
    return (
        pd.DataFrame({'feature': features, 'importance': perm.importances_mean})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
# -