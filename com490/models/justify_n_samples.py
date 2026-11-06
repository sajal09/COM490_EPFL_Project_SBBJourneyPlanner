# +
"""
Why n_samples = 10_000?

Monte Carlo with n draws estimates a transfer probability p with standard
error  sigma(n) = sqrt(p (1-p) / n).  We assume the user-facing confidence knob Q
takes values in gaps of 2%.  We pick
the smallest n such that the 95% MC envelope (1.96 * sigma) is below the
smallest of those gaps for every realistic p, so MC noise can never flip
a route across a confidence threshold.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np


# ── Defaults — exported so callers can introspect / override ─────────────
N_GRID_DEFAULT     = tuple(np.logspace(1, 6, 60).astype(int))
P_GRID_DEFAULT     = tuple(np.linspace(0.05, 0.95, 19))
DEFAULT_N          = 10_000
DEFAULT_OUT_PATH   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'figs', 'justify_n_samples.png',
)


def envelope(n, p):
    """Half-width of the 95% MC confidence interval for a Bernoulli mean."""
    return 1.96 * np.sqrt(p * (1.0 - p) / n)


def justify_n_samples(
        n_grid: Sequence[int]   = N_GRID_DEFAULT,
        p_grid: Sequence[float] = P_GRID_DEFAULT,
        default_n: int          = DEFAULT_N,
        out_path: str | None    = DEFAULT_OUT_PATH,
        seed: int               = 42,
        verbose: bool           = True,
        close: bool             = False,
) -> Tuple["matplotlib.figure.Figure", dict]:
    """
    Build the n_samples-justification figure and summary.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object. Open by default so the caller (notebook) can
        render it inline via ``plt.show()`` or ``display(fig)``. Pass
        ``close=True`` to release it.
    summary : dict
        Numeric summary — minimum gap between Q thresholds, smallest n
        whose 95% MC envelope fits inside that gap, the chosen default
        n, the worst-case envelope at the default, and the empirical
        95% CI from a 1000-trial Bernoulli simulation at p = 0.90.
    """
    import matplotlib.pyplot as plt

    n_grid = np.asarray(n_grid, dtype=int)
    p_grid = np.asarray(p_grid, dtype=float)

    min_gap = 0.02

    # Worst-case envelope across p for each n.
    worst = np.array([envelope(n, p_grid).max() for n in n_grid])

    # Smallest n where worst-case envelope fits inside min_gap.
    below = worst < min_gap
    n_min = int(n_grid[np.argmax(below)]) if below.any() else None

    # Empirical check at the chosen default.
    rng    = np.random.default_rng(seed)
    p_true = 0.90
    draws  = rng.binomial(default_n, p_true, size=1000) / default_n
    p_lo, p_hi = np.quantile(draws, [0.025, 0.975])
    worst_at_default = float(envelope(default_n, p_grid).max())

    summary = {
        'min_gap_between_q':        min_gap,
        'n_min_for_gap':            n_min,
        'default_n':                int(default_n),
        'worst_envelope_at_default': worst_at_default,
        'empirical_ci_at_default':  (float(p_lo), float(p_hi)),
        'empirical_ci_width':       float(p_hi - p_lo),
    }

    if verbose:
        print(f"smallest gap between Q thresholds : {min_gap:.2f}")
        print(f"smallest n with 95% env < gap     : "
              f"{n_min:,}" if n_min is not None
              else "smallest n with 95% env < gap     : (none on this grid)")
        print(f"chosen default n_samples          : {default_n:,}")
        print(f"  worst-case 95% envelope @ default : {worst_at_default:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(n_grid, worst, lw=2, color='steelblue',
              label=f'worst-case 95% MC envelope across '
                    f'p ∈ [{p_grid.min():.2f}, {p_grid.max():.2f}]')
    ax.axhline(min_gap, color='darkorange', lw=1.5, ls='--',
               label=f'smallest gap between Q thresholds = {min_gap:.2f}')
    ax.axvline(default_n, color='crimson', lw=1.5,
               label=f'default n_samples = {default_n:,}')
    ax.fill_between(n_grid, worst, min_gap, where=(worst > min_gap),
                    color='darkorange', alpha=0.15,
                    label='noise can flip route across Q threshold')

    ax.set_xlabel('n_samples')
    ax.set_ylabel('1.96 · √(p(1−p)/n)')
    ax.set_title(f'Justifying n_samples = {default_n:,}:\n'
                 'smallest n where MC noise is below the Q-threshold spacing')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

    if close:
        plt.close(fig)

    return fig, summary


if __name__ == '__main__':
    justify_n_samples(close=True)
