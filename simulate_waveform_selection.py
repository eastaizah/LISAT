"""
simulate_waveform_selection.py
------------------------------
Waveform selection frequency distribution by scenario and SNR regime.
Reproduces Figure 4 of the LiSAT paper.

Three SNR regimes:
    Low    : SNR < 5 dB    (average −2.5 dB)
    Medium : 5 ≤ SNR < 15 dB (average 10 dB)
    High   : SNR ≥ 15 dB   (average 22.5 dB)

Four scenarios:
    V2X, THz, IoE, UAV

For each (scenario, SNR regime) cell, the trained A3C agent's action
distribution is estimated over N_MC=2000 trials.

Output:
    results/waveform_selection.npz
    figures/fig4_waveform_selection.png
"""

from __future__ import annotations

import os
import sys
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

from channel_models import (
    VehicularB_V2X,
    TDL_C_THz,
    TDL_A_IoE,
    GaussMarkov_UAV,
    DoublyDispersiveChannel,
)
from lisat_model import A3CAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

# Prefix added by torch.compile() to every key in a compiled model's state_dict.
_COMPILE_PREFIX = "_orig_mod."

N_MC = 2000   # policy roll-outs per (scenario, SNR regime) cell

SNR_REGIMES = {
    "low":    {"range": (-5.0, 5.0),  "avg": -2.5,  "label": "Low\n(<5 dB)"},
    "medium": {"range": (5.0, 15.0),  "avg": 10.0,  "label": "Medium\n(5–15 dB)"},
    "high":   {"range": (15.0, 30.0), "avg": 22.5,  "label": "High\n(>15 dB)"},
}

SCENARIOS = ["V2X", "THz", "IoE", "UAV"]
WAVEFORMS = {0: "OTFS", 1: "OCDM", 2: "AFDM"}
WAVEFORM_COLORS = {0: "#2171b5", 1: "#2ca02c", 2: "#d62728"}  # blue, green, red

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------

def _make_channel(scenario: str) -> DoublyDispersiveChannel:
    if scenario == "V2X":
        return VehicularB_V2X()
    elif scenario == "THz":
        return TDL_C_THz()
    elif scenario == "IoE":
        return TDL_A_IoE()
    elif scenario == "UAV":
        return GaussMarkov_UAV()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _load_agent(scenario: str) -> A3CAgent:
    """Load pretrained agent if available; otherwise use random/initialised agent."""
    agent = A3CAgent()
    weight_path = os.path.join(
        RESULTS_DIR, f"lisat_agent_{scenario.lower()}.pt"
    )
    if os.path.exists(weight_path):
        raw_sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        # Weights may have been saved from a torch.compile()-wrapped model, which
        # prefixes every key with "_orig_mod.".  Strip that prefix so the state
        # dict matches a plain A3CAgent instance.
        state_dict = {
            (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
            for k, v in raw_sd.items()
        }
        agent.load_state_dict(state_dict)
        print(f"  [load] Loaded pretrained weights for {scenario}")
    else:
        print(f"  [warn] No pretrained weights for {scenario}; using initialised policy")
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Run policy to collect action distribution
# ---------------------------------------------------------------------------

def collect_action_distribution(
    agent: A3CAgent,
    channel: DoublyDispersiveChannel,
    snr_avg_db: float,
    snr_range_db: Tuple[float, float],
    n_mc: int = N_MC,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample the agent's policy n_mc times in the given SNR regime and
    return the empirical action distribution (3-vector summing to 1).

    Parameters
    ----------
    agent        : A3CAgent, trained policy
    channel      : channel model for context
    snr_avg_db   : representative SNR for this regime
    snr_range_db : (min, max) SNR range
    n_mc         : number of Monte-Carlo samples
    seed         : reproducibility seed

    Returns
    -------
    freq : np.ndarray [3], normalised selection frequencies
    """
    rng = np.random.RandomState(seed)
    action_counts = np.zeros(3, dtype=np.float64)

    tau_hat = channel.max_delay_samples / channel.sample_rate_hz
    nu_hat = channel.max_doppler_hz

    hidden = agent.init_hidden(batch_size=1)
    eta_s_prev = 0.8
    P_t_prev = 0.5
    wf_prev = 0

    with torch.no_grad():
        for mc in range(n_mc):
            # Sample SNR within regime
            snr_db = float(rng.uniform(*snr_range_db))

            # Add measurement noise to channel estimates (MERD realism)
            tau_noisy = tau_hat * (1 + 0.05 * rng.randn())
            nu_noisy = nu_hat * (1 + 0.05 * rng.randn())

            state = torch.tensor(
                [[
                    tau_noisy * 1e6,
                    nu_noisy / 1000.0,
                    snr_db / 30.0,
                    eta_s_prev,
                    P_t_prev,
                    float(wf_prev) / 2.0,
                ]],
                dtype=torch.float32,
            )

            action, log_prob, value, hidden = agent.select_action(
                state, hidden, deterministic=False
            )
            action_counts[action] += 1

            # Simulate feedback (lightweight)
            eta_s_prev = float(np.clip(
                0.85 + 0.1 * snr_db / 30.0 + 0.02 * rng.randn(), 0.0, 1.0
            ))
            P_t_prev = float(np.clip(0.5 + 0.05 * rng.randn(), 0.1, 1.0))
            wf_prev = action

            # Reset hidden state every 50 steps to avoid long-range correlation
            if mc % 50 == 49:
                hidden = agent.init_hidden(batch_size=1)

    freq = action_counts / action_counts.sum()
    return freq


# ---------------------------------------------------------------------------
# Main data collection
# ---------------------------------------------------------------------------

def run_waveform_selection_experiment() -> Dict[str, np.ndarray]:
    """
    Collect waveform selection frequency for all (scenario, SNR regime) pairs.

    Returns
    -------
    selection_data : dict  scenario → np.ndarray [3_regimes, 3_waveforms]
    """
    selection_data: Dict[str, np.ndarray] = {}
    regime_names = list(SNR_REGIMES.keys())

    total = len(SCENARIOS) * len(SNR_REGIMES)
    idx = 0

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario}")
        channel = _make_channel(scenario)
        agent = _load_agent(scenario)

        freqs = np.zeros((len(SNR_REGIMES), 3))

        for r_idx, (regime_name, regime_cfg) in enumerate(SNR_REGIMES.items()):
            idx += 1
            label = regime_cfg["label"].replace("\n", " ")
            print(f"  [{idx}/{total}] SNR regime: {label} (avg {regime_cfg['avg']:.1f} dB)...")

            t0 = time.time()
            freq = collect_action_distribution(
                agent=agent,
                channel=channel,
                snr_avg_db=regime_cfg["avg"],
                snr_range_db=regime_cfg["range"],
                n_mc=N_MC,
                seed=hash(f"{scenario}_{regime_name}") % 10000,
            )
            elapsed = time.time() - t0

            freqs[r_idx] = freq
            print(f"    OTFS={freq[0]:.3f}  OCDM={freq[1]:.3f}  AFDM={freq[2]:.3f}  "
                  f"({elapsed:.1f}s)")

        selection_data[scenario] = freqs

    return selection_data


# ---------------------------------------------------------------------------
# Plotting: grouped stacked bar chart
# ---------------------------------------------------------------------------

def plot_waveform_selection(
    selection_data: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    Plot grouped stacked bar chart of waveform selection frequencies.

    Layout: one group per scenario, three bars per group (SNR regimes),
    each bar stacked with OTFS/OCDM/AFDM fractions.
    """
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    regime_labels = [cfg["label"] for cfg in SNR_REGIMES.values()]
    n_regimes = len(regime_labels)
    n_scenarios = len(SCENARIOS)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(14, 5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]

    bar_width = 0.6
    x = np.arange(n_regimes)

    for ax_idx, (scenario, ax) in enumerate(zip(SCENARIOS, axes)):
        freqs = selection_data[scenario]  # [n_regimes, 3]

        bottom = np.zeros(n_regimes)
        for wf_idx in range(3):
            vals = freqs[:, wf_idx]
            color = WAVEFORM_COLORS[wf_idx]
            label = WAVEFORMS[wf_idx] if ax_idx == 0 else None
            bars = ax.bar(
                x, vals, bar_width,
                bottom=bottom,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )
            # Annotate bars with percentage if large enough
            for bar_i, (bar, val) in enumerate(zip(bars, vals)):
                if val > 0.08:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[bar_i] + val / 2,
                        f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(regime_labels, fontsize=9)
        ax.set_title(scenario, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        if ax_idx == 0:
            ax.set_ylabel("Selection Frequency")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_xlabel("SNR Regime")

    # Shared legend
    handles = [
        mpatches.Patch(color=WAVEFORM_COLORS[i], label=WAVEFORMS[i])
        for i in range(3)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=11,
        title="Waveform",
    )

    fig.suptitle(
        "Figure 4: LiSAT Waveform Selection Frequency by Scenario and SNR Regime",
        y=1.06, fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LiSAT Simulation: Waveform Selection (Figure 4)")
    print(f"MC samples per cell: {N_MC}")
    print("=" * 60)

    t_start = time.time()
    selection_data = run_waveform_selection_experiment()
    elapsed = time.time() - t_start
    print(f"\nTotal experiment time: {elapsed:.1f}s")

    # Save results
    npz_path = os.path.join(RESULTS_DIR, "waveform_selection.npz")
    save_dict: Dict[str, np.ndarray] = {}
    regime_labels = [cfg["label"] for cfg in SNR_REGIMES.values()]
    save_dict["regime_labels"] = np.array(regime_labels)
    save_dict["waveform_labels"] = np.array(list(WAVEFORMS.values()))
    for scenario, freqs in selection_data.items():
        save_dict[scenario] = freqs
    np.savez(npz_path, **save_dict)
    print(f"Saved results: {npz_path}")

    # Summary table
    print("\nSelection frequency summary:")
    print(f"{'Scenario':<8} | {'Regime':<22} | {'OTFS':>6} {'OCDM':>6} {'AFDM':>6}")
    print("-" * 58)
    for scenario, freqs in selection_data.items():
        for r_idx, regime in enumerate(regime_labels):
            f = freqs[r_idx]
            print(f"{scenario:<8} | {regime.replace(chr(10), ' '):<22} | "
                  f"{f[0]:6.3f} {f[1]:6.3f} {f[2]:6.3f}")

    # Plot
    fig_path = os.path.join(FIGURES_DIR, "fig4_waveform_selection.png")
    plot_waveform_selection(selection_data, fig_path)

    print("\nSimulation complete.")
    print(f"Results  : {npz_path}")
    print(f"Figure   : {fig_path}")


if __name__ == "__main__":
    main()
