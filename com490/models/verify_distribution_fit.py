"""
Verification: held-out AIC for the parametric-distribution comparison
in section 6 of final_walkthrough.py.

The notebook fits four families (Gumbel, Skew-Normal, Standard Normal,
Shifted Gamma) to the per-bin mean arrival delay. Ranking them by
in-sample log-likelihood is biased: every distribution gets a free LL
boost from being evaluated on the same data it was fitted to, and the
boost is larger for higher-parameter families.

This script fixes that by

  1. splitting the istdaten parquet chronologically by `operating_day`
     (earliest 80% of unique dates -> train, latest 20% -> test);
  2. on each half, aggregating per-(line, bpuic, dow, hour) bin and
     computing the bin-mean arrival delay in seconds, keeping only bins
     with at least `min_n_per_bin` observations in BOTH halves
     (apples-to-apples comparison);
  3. fitting each parametric family on the train bin means;
  4. evaluating AIC = 2k - 2·LL on the HELD-OUT test bin means only.

Usage
-----
In the notebook, after spark and ISTDATEN_PARQUET are defined:

    from models.verify_distribution_fit import (
        build_train_test_bin_means, fit_and_evaluate, plot_results
    )

    train_means, test_means = build_train_test_bin_means(
        spark, ISTDATEN_PARQUET, train_ratio=0.8, min_n_per_bin=50,
    )
    df = fit_and_evaluate(train_means, test_means)
    print(df.to_string(index=False))
    plot_results(df, train_means, test_means,
                 out_path='figs/distribution_fit_holdout.png')

Standalone run uses synthetic data so the logic can be verified without
HDFS access:

    python -m com490.models.verify_distribution_fit
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────
# (1) Spark-side: chronological 80/20 date split + matched bin means.
#     This is the *only* part that needs HDFS access. Run it inside the
#     notebook where `spark` and `ISTDATEN_PARQUET` are reachable.
# ─────────────────────────────────────────────────────────────────────────
def build_train_test_bin_means(spark,
                               istdaten_parquet_path: str,
                               train_ratio: float = 0.8,
                               min_n_per_bin: int = 50,
                               delay_clip_min: Tuple[int, int] = (-10, 60),
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronologically split istdaten by `operating_day`, then for every
    (line, bpuic, dow, hour) bin compute the mean arrival delay in
    SECONDS on each half. Only bins present in BOTH halves with at
    least `min_n_per_bin` observations are returned.

    Returns
    -------
    train_means_s, test_means_s : np.ndarray
        Bin-mean arrival delays in seconds, aligned 1:1 by bin key.
    """
    from pyspark.sql import functions as F

    lo, hi = delay_clip_min
    df = (
        spark.read.parquet(istdaten_parquet_path)
        .filter(F.col('arr_actual').isNotNull() & F.col('arr_time').isNotNull())
        .withColumn(
            'arr_delay_min',
            F.round(
                (F.col('arr_actual').cast('timestamp').cast('long')
                 - F.col('arr_time').cast('timestamp').cast('long')) / 60.0
            ).cast('int')
        )
        .filter(F.col('arr_delay_min').between(lo, hi))
        .withColumn('dow',  F.dayofweek('operating_day').cast('int'))
        .withColumn('hour', F.hour('arr_time').cast('int'))
    )

    # Chronological 80/20 split of UNIQUE operating_days.
    dates = sorted(r['operating_day']
                   for r in df.select('operating_day').distinct().collect())
    if len(dates) < 5:
        raise ValueError(
            f'Need at least 5 distinct operating_days for a meaningful '
            f'date split; got {len(dates)}.')
    cutoff_idx = int(round(train_ratio * len(dates)))
    cutoff_idx = max(1, min(cutoff_idx, len(dates) - 1))
    cutoff_date = dates[cutoff_idx]
    print(f'[date split] {len(dates)} distinct operating_days   '
          f'train: {dates[0]} .. {dates[cutoff_idx-1]}   '
          f'test: {cutoff_date} .. {dates[-1]}')

    grp = ['line_text', 'bpuic', 'dow', 'hour']
    train_agg = (
        df.filter(F.col('operating_day') < F.lit(cutoff_date))
          .groupBy(*grp)
          .agg(F.count('arr_delay_min').alias('n_tr'),
               F.mean('arr_delay_min').alias('mean_tr_min'))
    )
    test_agg = (
        df.filter(F.col('operating_day') >= F.lit(cutoff_date))
          .groupBy(*grp)
          .agg(F.count('arr_delay_min').alias('n_te'),
               F.mean('arr_delay_min').alias('mean_te_min'))
    )
    matched = (
        train_agg.join(test_agg, grp, 'inner')
                 .filter((F.col('n_tr') >= min_n_per_bin) &
                         (F.col('n_te') >= min_n_per_bin))
                 .toPandas()
    )
    print(f'[date split] matched bins (n >= {min_n_per_bin} in both halves): '
          f'{len(matched):,}')

    train_means_s = matched['mean_tr_min'].to_numpy() * 60.0
    test_means_s  = matched['mean_te_min'].to_numpy() * 60.0
    return train_means_s, test_means_s


# ─────────────────────────────────────────────────────────────────────────
# (2) Fit on train, score on train AND test. AIC / BIC penalise the
#     parameter count so families with different k are comparable.
# ─────────────────────────────────────────────────────────────────────────
def _families(train_data: np.ndarray, test_data: np.ndarray):
    """
    Returns a list of (name, scipy_dist, fit_kwargs, k) tuples.
    `k` is the number of *free* parameters (the ones MLE estimates).
    """
    # Gamma is shifted with `floc` fixed below the joint minimum, so the
    # location parameter is NOT fit -> free params = (a, scale) = 2.
    shift = float(np.floor(min(train_data.min(), test_data.min())) - 1.0)
    return [
        ('Standard Normal',                   stats.norm,     {},               2),
        ('Gumbel (Right)',                    stats.gumbel_r, {},               2),
        ('Skew-Normal',                       stats.skewnorm, {},               3),
        (f'Shifted Gamma (loc={shift:.0f}s)', stats.gamma,    {'floc': shift},  2),
    ]


def fit_and_evaluate(train_data: np.ndarray,
                     test_data: np.ndarray) -> pd.DataFrame:
    """
    Fit each family on `train_data`, then compute LL/AIC/BIC on the
    held-out `test_data` only. Returns one row per family.
    """
    n_te = len(test_data)
    rows = []
    for name, dist, fit_kwargs, k in _families(train_data, test_data):
        params = dist.fit(train_data, **fit_kwargs)
        ll = float(np.sum(dist.logpdf(test_data, *params)))
        rows.append({
            'family': name,
            'k':      k,
            'LL':     ll,
            'AIC':    2 * k - 2 * ll,
            'BIC':    k * np.log(n_te) - 2 * ll,
            'params': params,
        })
    df = pd.DataFrame(rows)
    # Lower AIC/BIC is better; higher LL is better.
    df['rank_AIC'] = df['AIC'].rank(ascending=True).astype(int)
    df['rank_BIC'] = df['BIC'].rank(ascending=True).astype(int)
    return df


def print_summary(df: pd.DataFrame) -> None:
    cols = ['family', 'k', 'LL', 'AIC', 'BIC', 'rank_AIC']
    pretty = df[cols].copy()
    for c in ['LL', 'AIC', 'BIC']:
        pretty[c] = pretty[c].map(lambda v: f'{v:>10.1f}')
    print(pretty.to_string(index=False))
    best = df.loc[df['AIC'].idxmin(), 'family']
    print(f"\nBest family by held-out AIC: {best}")


# ─────────────────────────────────────────────────────────────────────────
# (3) Plot.
# ─────────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame,
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 close: bool = False):
    """
    Save (and optionally close) the AIC-comparison figure. When called
    from a Jupyter cell, leave the figure open so plt.show() can render
    it inline; from the standalone __main__ block we close it to free
    matplotlib state.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    # (a) Held-out AIC bars per family.
    ax = axes[0]
    x = np.arange(len(df))
    bars = ax.bar(x, df['AIC'], color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(' (', '\n(') for f in df['family']],
                       fontsize=9)
    ax.set_ylabel('AIC = 2k − 2·LL    (lower is better)')
    ax.set_title('Held-out AIC by family   (date-split 80/20)')
    ax.grid(True, axis='y', alpha=0.3)

    # Mark winner.
    best = int(df['AIC'].idxmin())
    bars[best].set_color('seagreen')
    ymax = ax.get_ylim()[1]
    ax.annotate('best', (best, df['AIC'].iloc[best]),
                xytext=(best, df['AIC'].iloc[best] - 0.04 * ymax),
                ha='center', fontsize=9, color='seagreen', fontweight='bold')

    # (b) Train-fitted PDFs overlaid on test histogram.
    ax = axes[1]
    lo = min(train_data.min(), test_data.min())
    hi = max(train_data.max(), test_data.max())
    x  = np.linspace(lo - 30, hi + 30, 600)
    ax.hist(test_data, bins=60, density=True, alpha=0.35, color='slategray',
            edgecolor='white', label=f'Test bin-means  (n={len(test_data):,})')

    colors  = ['#1976D2', '#388E3C', '#F57C00', '#D32F2F']
    styles  = ['-', '--', ':', '-.']
    for (_, r), col, ls in zip(df.iterrows(), colors, styles):
        dist = {
            'Standard Normal':  stats.norm,
            'Gumbel (Right)':   stats.gumbel_r,
            'Skew-Normal':      stats.skewnorm,
        }.get(r['family'], stats.gamma)
        pdf = dist.pdf(x, *r['params'])
        ax.plot(x, pdf, color=col, lw=1.8, linestyle=ls,
                label=f"{r['family']}  AIC={r['AIC']:.0f}")
    ax.set_xlabel('bin-mean arrival delay (seconds)')
    ax.set_ylabel('density')
    ax.set_title('Train-fitted PDFs vs. held-out test bin-means')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Held-out AIC for the delay-distribution family comparison',
                 fontsize=12, y=1.04)
    if close:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# (4) Synthetic validation. No istdaten / Spark needed — generates
#     train/test bin means from a known shifted-gamma "ground truth"
#     plus a small distribution-shift between halves (mimicking what
#     a chronological split would expose). Used to prove the evaluator
#     is correctly implemented.
# ─────────────────────────────────────────────────────────────────────────
def _synthetic_bin_means(rng: np.random.Generator,
                         n_train: int = 4_000,
                         n_test: int = 1_000,
                         drift_s: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    # Ground truth: shifted gamma in SECONDS.
    a, loc, scale = 2.8, -120.0, 110.0
    train = stats.gamma.rvs(a, loc=loc,           scale=scale,
                            size=n_train, random_state=rng)
    # Test set gets a small mean shift to simulate seasonal drift —
    # this is what makes the held-out evaluation meaningful.
    test  = stats.gamma.rvs(a, loc=loc + drift_s, scale=scale,
                            size=n_test, random_state=rng)
    return train, test


def main():
    print('=== Synthetic-data validation (no Spark required) ===\n')
    rng = np.random.default_rng(20260527)
    train_means, test_means = _synthetic_bin_means(rng)
    print(f'Synthetic bin means — train: n={len(train_means):,}   '
          f'test: n={len(test_means):,}\n')

    df = fit_and_evaluate(train_means, test_means)
    print_summary(df)

    out_dir  = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'figs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'distribution_fit_holdout.png')
    plot_results(df, train_means, test_means, out_path, close=True)
    print(f'\nSaved: {out_path}')
    print('\nWhen you run this on real istdaten in the notebook, replace')
    print('_synthetic_bin_means(...) with build_train_test_bin_means(spark, ...).')


if __name__ == '__main__':
    main()
