"""Figures for the unit commitment and water value parts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"actual": "#222222", "merit": "#1f77b4", "restricted": "#ff7f0e", "relaxed": "#2ca02c"}


def uc_duration(res: pd.DataFrame, path: Path) -> None:
    years = sorted(res["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(4.2 * len(years), 3.6), sharey=True)
    for ax, y in zip(np.atleast_1d(axes), years):
        g = res[res["year"] == y]
        x = np.linspace(0, 100, len(g))
        for col, lab, c in (("price", "actual", "actual"), ("model_price", "merit order", "merit"),
                            ("uc_price_restricted", "UC restricted", "restricted"),
                            ("uc_price_relaxed", "UC relaxed", "relaxed")):
            ax.plot(x, np.sort(g[col].to_numpy())[::-1], label=lab, color=COLORS[c], lw=1.3)
        ax.set_title(str(y))
        ax.set_xlabel("share of hours (%)")
        ax.set_ylim(-60, 300)
        ax.axhline(0, color="grey", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("day-ahead price (EUR/MWh)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Price duration curves: observed, merit order and unit commitment")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def uc_week(res: pd.DataFrame, path: Path, start: str = "2024-04-01", days: int = 7) -> None:
    t0 = pd.Timestamp(start, tz="Europe/Berlin")
    loc = res["ts_utc"].dt.tz_convert("Europe/Berlin")
    g = res[(loc >= t0) & (loc < t0 + pd.Timedelta(days=days))]
    x = g["ts_utc"].dt.tz_convert("Europe/Berlin")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for col, lab, c in (("price", "actual", "actual"), ("model_price", "merit order", "merit"),
                        ("uc_price_restricted", "UC restricted", "restricted"),
                        ("uc_price_relaxed", "UC relaxed", "relaxed")):
        ax.plot(x, g[col], label=lab, color=COLORS[c], lw=1.3)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Hourly prices, week from {start}")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def water_values(wv: np.ndarray, levels: np.ndarray, capacity: float, path: Path,
                 observed: pd.DataFrame | None = None, floor: np.ndarray | None = None) -> None:
    """Water value map (multiple of the annual mean price) by week and filling; below the floor masked."""
    wv = wv.copy()
    if floor is not None:
        wv[levels[None, :] < floor[:, None]] = np.nan
    fig, ax = plt.subplots(figsize=(9, 4.6))
    weeks = np.arange(1, wv.shape[0] + 1)
    fill = 100 * levels / capacity
    cs = ax.contourf(weeks, fill, wv.T, levels=np.linspace(np.nanpercentile(wv, 1), np.nanpercentile(wv, 99), 15),
                     cmap="viridis", extend="both")
    fig.colorbar(cs, ax=ax, label="water value (x annual mean price)")
    if floor is not None:
        ax.plot(weeks, 100 * floor / capacity, color="black", lw=1.2, ls="--", label="operating floor (1995-2014 minimum)")
        ax.legend(loc="upper right", fontsize=8)
    if observed is not None:
        for y, g in observed.groupby("year"):
            ax.plot(g["week"], 100 * g["fill_twh"] / capacity, color="white", lw=0.8, alpha=0.7)
    ax.set_xlabel("week of year")
    ax.set_ylabel("reservoir filling (%)")
    ax.set_title("Water values of the aggregate Norwegian reservoir (lines: observed fillings)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def hydro_backtest(sim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = sim["date"]
    axes[0].plot(x, sim["fill_obs"], color=COLORS["actual"], label="observed")
    axes[0].plot(x, sim["fill_sdp"], color=COLORS["relaxed"], label="water value policy")
    axes[0].plot(x, sim["fill_pf"], color=COLORS["merit"], ls="--", label="perfect foresight")
    axes[0].set_ylabel("reservoir (TWh)")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, sim["price"], color=COLORS["actual"], label="observed price")
    axes[1].plot(x, sim["wv_obs"], color=COLORS["restricted"], label="model water value at observed filling")
    axes[1].set_ylabel("EUR/MWh")
    axes[1].legend(fontsize=8)
    fig.suptitle("Out-of-sample backtest (each year with a policy trained on the other years)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
