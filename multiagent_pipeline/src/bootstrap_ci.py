"""Bootstrap confidence intervals for the comparative metrics.

The headline numbers of the project — *97.2 % label agreement*, *Pearson r =
0.9847* on the ensemble score — are point estimates over 567 routes. The
brief asks for a comparative analysis, so a non-trivial reviewer will want
to know how much those numbers move when we resample.

This module implements a single-purpose bootstrap:

    >>> ci = bootstrap_agreement(cl, ma, n_iter=1000, sample_frac=0.80)
    >>> print(ci)
    BootstrapResult(point=0.9719, mean=0.9716, lo=0.9577, hi=0.9824,
                    n_iter=1000, sample_frac=0.80)

`cl` and `ma` are the two pipelines' DataFrames already aligned on `ROTTA`
(use `bootstrap_agreement.align(cl, ma)` if not). Each iteration draws
`int(n_routes * sample_frac)` routes WITHOUT replacement and recomputes the
chosen metric. Confidence intervals are the 2.5 % and 97.5 % percentiles of
the bootstrap distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate + bootstrap distribution summary."""
    point: float           # value computed on the full sample
    mean: float            # mean across bootstrap iterations
    lo: float              # 2.5 % percentile
    hi: float              # 97.5 % percentile
    std: float             # std of the bootstrap distribution
    n_iter: int
    sample_frac: float

    def as_dict(self) -> dict:
        return {
            "point":       self.point,
            "mean":        self.mean,
            "lo_95":       self.lo,
            "hi_95":       self.hi,
            "std":         self.std,
            "n_iter":      self.n_iter,
            "sample_frac": self.sample_frac,
            "ci_width":    self.hi - self.lo,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def align_on_rotta(cl: pd.DataFrame, ma: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-joins the two pipelines on ROTTA so every metric is computed
    on the same routes. Returns the two slices with a shared sorted index."""
    if "ROTTA" not in cl.columns and cl.index.name != "ROTTA":
        raise ValueError("cl must have a 'ROTTA' column or index")
    if "ROTTA" not in ma.columns and ma.index.name != "ROTTA":
        raise ValueError("ma must have a 'ROTTA' column or index")
    cl = cl.set_index("ROTTA") if "ROTTA" in cl.columns else cl
    ma = ma.set_index("ROTTA") if "ROTTA" in ma.columns else ma
    common = cl.index.intersection(ma.index).sort_values()
    return cl.loc[common], ma.loc[common]


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap(
    metric_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    cl: pd.DataFrame,
    ma: pd.DataFrame,
    *,
    n_iter: int,
    sample_frac: float,
    seed: int,
) -> BootstrapResult:
    """Generic bootstrap loop used by every public helper below."""
    rng = np.random.default_rng(seed)
    n = len(cl)
    k = max(2, int(n * sample_frac))
    point = float(metric_fn(cl, ma))

    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.choice(n, size=k, replace=False)
        samples[i] = float(metric_fn(cl.iloc[idx], ma.iloc[idx]))

    lo, hi = _percentile_ci(samples)
    return BootstrapResult(
        point=point,
        mean=float(samples.mean()),
        lo=lo, hi=hi,
        std=float(samples.std(ddof=1)),
        n_iter=n_iter,
        sample_frac=sample_frac,
    )


# ── Public metrics ──────────────────────────────────────────────────────────

def bootstrap_agreement(
    cl: pd.DataFrame,
    ma: pd.DataFrame,
    *,
    label_col_cl: str = "anomaly_label",
    label_col_ma: str = "risk_label",
    n_iter: int = 1000,
    sample_frac: float = 0.80,
    seed: int = 42,
) -> BootstrapResult:
    """Row-by-row label agreement between the classical and multi-agent
    pipelines, expressed as a fraction in [0, 1]."""
    cl, ma = align_on_rotta(cl, ma)

    def _metric(c, m):
        return (c[label_col_cl].values == m[label_col_ma].values).mean()

    return _bootstrap(_metric, cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed)


def bootstrap_pearson(
    cl: pd.DataFrame,
    ma: pd.DataFrame,
    *,
    score_col_cl: str = "anomaly_score",
    score_col_ma: str = "ensemble_score",
    n_iter: int = 1000,
    sample_frac: float = 0.80,
    seed: int = 42,
) -> BootstrapResult:
    """Pearson r between the two pipelines' final scalar scores."""
    cl, ma = align_on_rotta(cl, ma)

    def _metric(c, m):
        return pearsonr(c[score_col_cl].values, m[score_col_ma].values)[0]

    return _bootstrap(_metric, cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed)


def bootstrap_spearman(
    cl: pd.DataFrame,
    ma: pd.DataFrame,
    *,
    score_col_cl: str = "anomaly_score",
    score_col_ma: str = "ensemble_score",
    n_iter: int = 1000,
    sample_frac: float = 0.80,
    seed: int = 42,
) -> BootstrapResult:
    """Spearman ρ between the two pipelines' final scalar scores."""
    cl, ma = align_on_rotta(cl, ma)

    def _metric(c, m):
        return spearmanr(c[score_col_cl].values, m[score_col_ma].values)[0]

    return _bootstrap(_metric, cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed)


# ── Convenience: all three at once ─────────────────────────────────────────

def bootstrap_all(
    cl: pd.DataFrame,
    ma: pd.DataFrame,
    *,
    n_iter: int = 1000,
    sample_frac: float = 0.80,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """Returns a dict {agreement, pearson, spearman} so the caller can render
    the three CIs in a single table."""
    return {
        "agreement": bootstrap_agreement(cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed),
        "pearson":   bootstrap_pearson(  cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed),
        "spearman":  bootstrap_spearman( cl, ma, n_iter=n_iter, sample_frac=sample_frac, seed=seed),
    }


if __name__ == "__main__":
    # Smoke test using the on-disk artefacts produced by the two pipelines.
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    cl = pd.read_csv(here / "data" / "processed" / "anomaly_results.csv")
    ma = pd.read_csv(here / "data" / "processed" / "anomaly_results_live.csv")

    res = bootstrap_all(cl, ma, n_iter=1000, sample_frac=0.80, seed=42)
    print("Bootstrap (1000 iterations, 80 % subsample, seed=42)")
    print("=" * 60)
    for name, r in res.items():
        print(f"  {name:10s}  point={r.point:.4f}  mean={r.mean:.4f}  "
              f"95 % CI = [{r.lo:.4f}, {r.hi:.4f}]  ±{(r.hi-r.lo)/2:.4f}")
