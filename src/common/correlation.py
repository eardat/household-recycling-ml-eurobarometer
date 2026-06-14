"""Association / correlation utilities for exploratory data analysis.

Provides:
  * Cramer's V (bias-corrected) for nominal-nominal association.
  * Spearman rank correlation for ordinal / continuous variables.
  * Helpers to rebuild original multi-category variables from one-hot dummies.
  * Heatmap plotting consistent with the project figure style.

The survey features are predominantly categorical / ordinal, so a plain Pearson
correlation is not appropriate. Cramer's V summarises the strength of
association between categorical variables, while Spearman captures the
monotonic relationship between ordered variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series (Bergsma, 2013).

    Returns a value in [0, 1]; 0 means no association, 1 means perfect
    association. The correction removes the upward bias that affects the raw
    statistic when tables have many categories or limited samples.
    """
    confusion = pd.crosstab(x, y)
    obs = confusion.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0:
        return np.nan
    row_totals = obs.sum(axis=1, keepdims=True)
    col_totals = obs.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    # Guard against empty expected cells (absent categories).
    mask = expected > 0
    chi2 = (((obs - expected) ** 2)[mask] / expected[mask]).sum()
    phi2 = chi2 / n
    r, k = obs.shape
    phi2corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def cramers_v_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Symmetric matrix of bias-corrected Cramer's V for the given columns."""
    mat = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            b = columns[j]
            value = cramers_v(df[a], df[b])
            mat.loc[a, b] = value
            mat.loc[b, a] = value
    return mat


def spearman_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Spearman rank correlation matrix (computed without SciPy).

    Spearman's rho equals the Pearson correlation of the average ranks, so we
    rank each column (ties averaged) and take the Pearson matrix. Values lie in
    [-1, 1].
    """
    return df[columns].rank(method="average").corr(method="pearson")


def reconstruct_from_dummies(df: pd.DataFrame, prefix: str, new_name: str) -> pd.Series:
    """Rebuild a single categorical column from its one-hot dummy columns.

    Each row is assigned the label of the dummy column equal to 1. Labels are
    the dummy column names with the prefix stripped.
    """
    dummies = [c for c in df.columns if c.startswith(prefix)]
    if not dummies:
        raise ValueError(f"No dummy columns found for prefix {prefix!r}")
    labels = df[dummies].idxmax(axis=1).str.replace(prefix, "", regex=False)
    return pd.Series(labels.to_numpy(), index=df.index, name=new_name)


def plot_cramers_v(matrix: pd.DataFrame, title: str, path) -> None:
    """Lower-triangle Cramer's V heatmap (sequential Blues palette, range 0-1)."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * n + 2), max(5.5, 0.85 * n + 1.5)))
    sns.heatmap(
        matrix, mask=mask, annot=True, fmt=".2f", cmap="Blues",
        vmin=0, vmax=1, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Cramer's V"}, ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spearman(matrix: pd.DataFrame, title: str, path) -> None:
    """Lower-triangle Spearman heatmap (diverging palette centred at 0)."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * n + 2), max(5.5, 0.85 * n + 1.5)))
    sns.heatmap(
        matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Spearman rho"}, ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
