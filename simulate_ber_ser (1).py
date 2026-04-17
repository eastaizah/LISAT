"""
simulate_ber_ser.py
-------------------
BER/SER vs. SNR simulation for all waveform methods in the V2X scenario.
Reproduces Figure 2 of the LiSAT paper.

Methods compared:
    - OFDM         (baseline, no waveform adaptation)
    - OTFS         (fixed)
    - OCDM         (fixed)
    - AFDM         (fixed)
    - CSI-Threshold (heuristic: selects OTFS if τ_max > threshold, else OFDM)
    - LiSAT        (A3C-based adaptive waveform selection)

SNR range : -5 dB to 30 dB in steps of 2.5 dB
MC runs   : 500 realisations per SNR point (reduced for speed; set N_MC=1000
            for publication-quality results)

Output:
    results/ber_ser_v2x.npz
    figures/fig2_ber_ser.png
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from the simulations directory
sys.path.insert(0, os.path.dirname(__file__))

from channel_models import VehicularB_V2X, DoublyDispersiveChannel
from waveforms import (
    OFDMModulator, OFDMDemodulator,
    OTFSModulator, OTFSDemodulator,
    OCDMModulator, OCDMDemodulator,
    AFDMModulator, AFDMDemodulator,
    qam_mapper, qam_demapper, compute_papr,
)
from lisat_model import LiSATSystem, A3CAgent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

# Prefix added by torch.compile() to every key in a compiled model's state_dict.
_COMPILE_PREFIX = "_orig_mod."

N_SUBCARRIERS = 64
N_CP = 20                # CP ≥ max delay spread at the effective sample rate
QAM_ORDER = 16
BITS_PER_SYM = int(math.log2(QAM_ORDER))
N_SYMBOLS = 14           # OFDM-like: 14 symbols per frame (LTE-alike)
N_MC = 500               # Monte-Carlo realisations per SNR point
SNR_RANGE_DB = np.arange(-5.0, 30.5, 2.5)

OTFS_M = N_SUBCARRIERS   # delay bins
OTFS_N = N_SYMBOLS       # Doppler bins

# Effective sample rate for 15 kHz SCS with N=64 subcarriers
SAMPLE_RATE_HZ = N_SUBCARRIERS * 15e3  # 0.96 MHz

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: compute semantic efficiency
# ---------------------------------------------------------------------------

def compute_semantic_efficiency(decoded: torch.Tensor, original: torch.Tensor) -> float:
    """
    Semantic efficiency η_s = 1 − MSE_normalised.

    MSE is normalised by the average signal power to produce a value in [0, 1].

    Parameters
    ----------
    decoded  : torch.Tensor, received/decoded symbols
    original : torch.Tensor, transmitted symbols

    Returns
    -------
    eta_s : float in [0, 1]
    """
    sig_power = (original.abs() ** 2).mean().item()
    mse = F.mse_loss(decoded.real.float(), original.real.float()).item()
    eta_s = 1.0 - mse / (sig_power + 1e-9)
    return float(np.clip(eta_s, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 16-QAM BER / SER analytical functions
# ---------------------------------------------------------------------------

def _ber_16qam(sinr_lin: float) -> float:
    """Approximate closed-form BER for 16-QAM in AWGN."""
    if sinr_lin <= 0:
        return 0.5
    x = math.sqrt(3.0 * sinr_lin / (2.0 * (QAM_ORDER - 1)))
    return max(float((3.0 / 8.0) * math.erfc(x)), 1e-7)


def _ser_16qam(sinr_lin: float) -> float:
    """Approximate closed-form SER for 16-QAM in AWGN."""
    if sinr_lin <= 0:
        return 1.0
    x = math.sqrt(3.0 * sinr_lin / (2.0 * (QAM_ORDER - 1)))
    pe = 0.75 * math.erfc(x)
    return max(float(1.0 - (1.0 - pe) ** 2), 1e-7)


# ---------------------------------------------------------------------------
# Semi-analytical BER/SER per waveform with random channel realisations
# ---------------------------------------------------------------------------

def _generate_channel_gains(channel: DoublyDispersiveChannel, batch: int = 1):
    """Generate random Rayleigh path gains for one channel realisation."""
    L = channel.num_paths
    powers = channel.path_powers_lin.cpu()
    amp = torch.sqrt(powers)
    gains = (
        torch.randn(batch, L) + 1j * torch.randn(batch, L)
    ).to(torch.complex64) * amp.unsqueeze(0) / math.sqrt(2.0)
    return gains


def _freq_response(channel: DoublyDispersiveChannel, gains: torch.Tensor, N: int):
    """
    Compute the N-point frequency-domain channel H[k] for given path gains.
    H[k] = sum_l gains[l] * exp(-j 2 pi tau_l k / N)
    """
    L = channel.num_paths
    delays = channel.delays_samples.cpu().float()
    k = torch.arange(N, dtype=torch.float32)
    H = torch.zeros(gains.shape[0], N, dtype=torch.complex64)
    for l_idx in range(L):
        tau = delays[l_idx].item()
        phase = -2.0 * math.pi * tau * k / N
        H += gains[:, l_idx:l_idx + 1] * torch.exp(1j * phase).unsqueeze(0)
    return H


def compute_ber_ser_for_waveform(
    waveform_name: str,
    channel: DoublyDispersiveChannel,
    snr_range: np.ndarray,
    n_mc: int = N_MC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-analytical BER/SER computation using effective-SINR models.

    For each Monte-Carlo trial a random channel realisation is drawn and the
    per-symbol SINR is computed analytically for the chosen waveform.  BER/SER
    are then obtained from the 16-QAM closed-form expressions.

    Waveform models
    ---------------
    OFDM  per-subcarrier ZF with ICI floor from Doppler spread.
    OTFS  full delay-Doppler diversity via MMSE in DD domain.
    OCDM  chirp spreading frequency diversity, partial Doppler protection.
    AFDM  affine frequency-division diversity, near-optimal DD exploitation.
    """
    ber_arr = np.zeros(len(snr_range))
    ser_arr = np.zeros(len(snr_range))

    delta_f = channel.sample_rate_hz / N_SUBCARRIERS
    eps_max = channel.max_doppler_hz / delta_f
    sigma2_ici = (math.pi * eps_max) ** 2 / 3.0  # ICI variance for OFDM

    for snr_idx, snr_db in enumerate(snr_range):
        snr_lin = 10.0 ** (snr_db / 10.0)
        ber_sum = 0.0
        ser_sum = 0.0

        for _mc in range(n_mc):
            gains = _generate_channel_gains(channel, batch=1)

            if waveform_name == "OFDM":
                # Per-subcarrier ZF equalization with ICI residual noise
                H = _freq_response(channel, gains, N_SUBCARRIERS)
                H_sq = (H.abs() ** 2).squeeze(0).numpy()
                ber_mc = 0.0
                ser_mc = 0.0
                for k_idx in range(N_SUBCARRIERS):
                    sinr = H_sq[k_idx] * snr_lin / (
                        1.0 + H_sq[k_idx] * snr_lin * sigma2_ici
                    )
                    ber_mc += _ber_16qam(sinr)
                    ser_mc += _ser_16qam(sinr)
                ber_sum += ber_mc / N_SUBCARRIERS
                ser_sum += ser_mc / N_SUBCARRIERS

            elif waveform_name == "OTFS":
                # OTFS: full DD diversity — MMSE exploits all resolvable paths
                g_total = (gains.abs() ** 2).sum().item()
                sinr = g_total * snr_lin
                ber_sum += _ber_16qam(sinr)
                ser_sum += _ser_16qam(sinr)

            elif waveform_name == "OCDM":
                # OCDM: chirp spreading provides frequency diversity
                H = _freq_response(channel, gains, N_SUBCARRIERS)
                H_sq = (H.abs() ** 2).squeeze(0).numpy()
                # Effective SINR from chirp averaging (harmonic mean)
                inv_sum = np.mean(1.0 / (H_sq * snr_lin + 1.0))
                sinr_chirp = max((1.0 / inv_sum) - 1.0, 0.0)
                # Partial Doppler penalty (smaller than OFDM)
                sinr_chirp = sinr_chirp / (1.0 + sinr_chirp * sigma2_ici * 0.1)
                ber_sum += _ber_16qam(sinr_chirp)
                ser_sum += _ser_16qam(sinr_chirp)

            elif waveform_name == "AFDM":
                # AFDM: DAFT provides near-optimal DD diversity
                g_total = (gains.abs() ** 2).sum().item()
                # Small DAFT-mismatch penalty relative to optimal OTFS
                sinr = g_total * snr_lin * 0.85
                ber_sum += _ber_16qam(sinr)
                ser_sum += _ser_16qam(sinr)

            else:
                raise ValueError(f"Unknown waveform: {waveform_name}")

        ber_arr[snr_idx] = ber_sum / n_mc
        ser_arr[snr_idx] = ser_sum / n_mc

    return ber_arr, ser_arr


# ---------------------------------------------------------------------------
# CSI-Threshold baseline
# ---------------------------------------------------------------------------

def compute_ber_ser_csi_threshold(
    channel: DoublyDispersiveChannel,
    snr_range: np.ndarray,
    tau_threshold_ns: float = 5000.0,
    n_mc: int = N_MC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSI-threshold heuristic: select OTFS if τ_max > threshold, else OFDM.

    Approximates a simple rule-based adaptive modulation scheme as baseline.
    """
    tau_max_s = channel.max_delay_samples / channel.sample_rate_hz
    tau_max_ns = tau_max_s * 1e9

    if tau_max_ns > tau_threshold_ns:
        # High delay spread → use OTFS
        return compute_ber_ser_for_waveform("OTFS", channel, snr_range, n_mc)
    else:
        return compute_ber_ser_for_waveform("OFDM", channel, snr_range, n_mc)


# ---------------------------------------------------------------------------
# LiSAT adaptive baseline (trained policy run)
# ---------------------------------------------------------------------------

def compute_ber_ser_lisat(
    channel: DoublyDispersiveChannel,
    snr_range: np.ndarray,
    n_mc: int = N_MC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LiSAT adaptive waveform selection via a pretrained A3C agent.

    For each SNR/MC trial, the agent observes the channel state and selects
    the optimal waveform, then evaluates BER/SER analytically for that choice.
    """
    agent = A3CAgent()
    agent.eval()

    # Load pretrained weights if available, else use random policy
    weight_path = os.path.join(os.path.dirname(__file__), "results", "lisat_agent_v2x.pt")
    if os.path.exists(weight_path):
        print("  [LiSAT] Loading pretrained agent weights...")
        raw_sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        state_dict = {
            (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
            for k, v in raw_sd.items()
        }
        agent.load_state_dict(state_dict)
    else:
        print("  [LiSAT] No pretrained weights found; using random policy.")

    wf_names = {0: "OTFS", 1: "OCDM", 2: "AFDM"}

    delta_f = channel.sample_rate_hz / N_SUBCARRIERS
    eps_max = channel.max_doppler_hz / delta_f
    sigma2_ici = (math.pi * eps_max) ** 2 / 3.0

    ber_arr = np.zeros(len(snr_range))
    ser_arr = np.zeros(len(snr_range))

    tau_hat = channel.max_delay_samples / channel.sample_rate_hz
    nu_hat = channel.max_doppler_hz

    for snr_idx, snr_db in enumerate(snr_range):
        snr_lin = 10.0 ** (snr_db / 10.0)
        ber_sum = 0.0
        ser_sum = 0.0

        hidden = agent.init_hidden(batch_size=1)
        eta_s_prev = 0.8
        P_t_prev = 0.5
        waveform_prev = 2  # start with AFDM

        for _mc in range(n_mc):
            state = torch.tensor(
                [[
                    tau_hat * 1e6,
                    nu_hat / 1000.0,
                    snr_db / 30.0,
                    eta_s_prev,
                    P_t_prev,
                    float(waveform_prev) / 2.0,
                ]],
                dtype=torch.float32,
            )

            with torch.no_grad():
                action, _, _, hidden = agent.select_action(state, hidden, deterministic=True)

            gains = _generate_channel_gains(channel, batch=1)
            wf = wf_names[action]

            # Compute BER/SER for the selected waveform
            if wf == "OTFS":
                g_total = (gains.abs() ** 2).sum().item()
                sinr = g_total * snr_lin
            elif wf == "OCDM":
                H = _freq_response(channel, gains, N_SUBCARRIERS)
                H_sq = (H.abs() ** 2).squeeze(0).numpy()
                inv_sum = np.mean(1.0 / (H_sq * snr_lin + 1.0))
                sinr = max((1.0 / inv_sum) - 1.0, 0.0)
                sinr = sinr / (1.0 + sinr * sigma2_ici * 0.1)
            else:  # AFDM
                g_total = (gains.abs() ** 2).sum().item()
                sinr = g_total * snr_lin * 0.85

            ber_mc = _ber_16qam(sinr)
            ser_mc = _ser_16qam(sinr)
            ber_sum += ber_mc
            ser_sum += ser_mc

            eta_s_prev = float(np.clip(1.0 - ber_mc * 4.0, 0.0, 1.0))
            waveform_prev = action

        ber_arr[snr_idx] = ber_sum / n_mc
        ser_arr[snr_idx] = ser_sum / n_mc

    return ber_arr, ser_arr


# ---------------------------------------------------------------------------
# Genie-aided oracle (omniscient waveform selection)
# ---------------------------------------------------------------------------

def compute_ber_ser_oracle(
    channel: DoublyDispersiveChannel,
    snr_range: np.ndarray,
    n_mc: int = N_MC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genie-aided oracle: for each trial, compute BER for all waveforms and
    pick the one with the lowest BER.  Represents the best possible adaptive
    waveform selection with perfect CSI.
    """
    delta_f = channel.sample_rate_hz / N_SUBCARRIERS
    eps_max = channel.max_doppler_hz / delta_f
    sigma2_ici = (math.pi * eps_max) ** 2 / 3.0

    ber_arr = np.zeros(len(snr_range))
    ser_arr = np.zeros(len(snr_range))

    for snr_idx, snr_db in enumerate(snr_range):
        snr_lin = 10.0 ** (snr_db / 10.0)
        ber_sum = 0.0
        ser_sum = 0.0

        for _mc in range(n_mc):
            gains = _generate_channel_gains(channel, batch=1)
            g_total = (gains.abs() ** 2).sum().item()
            H = _freq_response(channel, gains, N_SUBCARRIERS)
            H_sq = (H.abs() ** 2).squeeze(0).numpy()

            # OTFS: full DD diversity
            sinr_otfs = g_total * snr_lin
            # OCDM: chirp diversity
            inv_sum = np.mean(1.0 / (H_sq * snr_lin + 1.0))
            sinr_ocdm = max((1.0 / inv_sum) - 1.0, 0.0)
            sinr_ocdm = sinr_ocdm / (1.0 + sinr_ocdm * sigma2_ici * 0.1)
            # AFDM: DD diversity with mismatch penalty
            sinr_afdm = g_total * snr_lin * 0.85

            bers = [
                _ber_16qam(sinr_otfs),
                _ber_16qam(sinr_ocdm),
                _ber_16qam(sinr_afdm),
            ]
            sers = [
                _ser_16qam(sinr_otfs),
                _ser_16qam(sinr_ocdm),
                _ser_16qam(sinr_afdm),
            ]
            best_idx = int(np.argmin(bers))
            ber_sum += bers[best_idx]
            ser_sum += sers[best_idx]

        ber_arr[snr_idx] = ber_sum / n_mc
        ser_arr[snr_idx] = ser_sum / n_mc

    return ber_arr, ser_arr


# ---------------------------------------------------------------------------
# Plotting (IEEE-style)
# ---------------------------------------------------------------------------

def plot_ber_ser(
    results: Dict[str, Dict[str, np.ndarray]],
    snr_range: np.ndarray,
    save_path: str,
) -> None:
    """Plot BER/SER curves with IEEE-style formatting."""
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "figure.dpi": 100,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Styles matching article Fig. 2 description
    styles = {
        "OFDM":          {"color": "black",   "ls": "--", "marker": "o",  "ms": 4, "label": "OFDM"},
        "OTFS":          {"color": "blue",    "ls": "--", "marker": "s",  "ms": 4, "label": "OTFS (fixed)"},
        "OCDM":          {"color": "green",   "ls": "--", "marker": "^",  "ms": 4, "label": "OCDM (fixed)"},
        "AFDM":          {"color": "red",     "ls": "--", "marker": "D",  "ms": 4, "label": "AFDM (fixed)"},
        "CSI-Threshold": {"color": "magenta", "ls": ":",  "marker": "x",  "ms": 5, "label": "CSI-threshold heuristic"},
        "LiSAT":         {"color": "red",     "ls": "-",  "marker": "*",  "ms": 6, "label": "LiSAT (proposed)", "lw": 2.5},
        "Oracle":        {"color": "gray",    "ls": "-",  "marker": None, "ms": 0, "label": "Genie-aided oracle", "lw": 1.0},
    }

    for method, sty in styles.items():
        if method not in results:
            continue
        ber = np.clip(results[method]["ber"], 1e-4, 1.0)
        ser = np.clip(results[method]["ser"], 1e-4, 1.0)
        kw = dict(color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
                  markersize=sty["ms"], markevery=3, label=sty["label"],
                  linewidth=sty.get("lw", 1.5))
        axes[0].semilogy(snr_range, ber, **kw)
        axes[1].semilogy(snr_range, ser, **kw)

    for ax, title, ylabel in zip(
        axes,
        ["Bit Error Rate (BER)", "Symbol Error Rate (SER)"],
        ["BER", "SER"],
    ):
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_xlim(snr_range[0], snr_range[-1])
        ax.set_ylim(1e-4, 1.0)

    fig.suptitle("Figure 2: BER/SER vs. SNR — V2X Vehicular-B Channel", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LiSAT Simulation: BER/SER vs. SNR (V2X, Figure 2)")
    print("=" * 60)

    channel = VehicularB_V2X(sample_rate_hz=SAMPLE_RATE_HZ)
    results: Dict[str, Dict[str, np.ndarray]] = {}

    waveforms_fixed = ["OFDM", "OTFS", "OCDM", "AFDM"]

    for i, wf in enumerate(waveforms_fixed):
        print(f"\n[{i+1}/7] Computing BER/SER for {wf}...")
        ber, ser = compute_ber_ser_for_waveform(wf, channel, SNR_RANGE_DB, N_MC)
        results[wf] = {"ber": ber, "ser": ser}
        print(f"  BER @ 10 dB: {ber[np.argmin(np.abs(SNR_RANGE_DB - 10.0))]:.4e}")

    print(f"\n[5/7] Computing BER/SER for CSI-Threshold...")
    ber_csi, ser_csi = compute_ber_ser_csi_threshold(channel, SNR_RANGE_DB, n_mc=N_MC)
    results["CSI-Threshold"] = {"ber": ber_csi, "ser": ser_csi}

    print(f"\n[6/7] Computing BER/SER for LiSAT...")
    ber_lisat, ser_lisat = compute_ber_ser_lisat(channel, SNR_RANGE_DB, n_mc=N_MC)
    results["LiSAT"] = {"ber": ber_lisat, "ser": ser_lisat}

    print(f"\n[7/7] Computing BER/SER for Genie-aided Oracle...")
    ber_oracle, ser_oracle = compute_ber_ser_oracle(channel, SNR_RANGE_DB, n_mc=N_MC)
    results["Oracle"] = {"ber": ber_oracle, "ser": ser_oracle}

    # Save numerical results
    npz_path = os.path.join(RESULTS_DIR, "ber_ser_v2x.npz")
    save_dict = {"snr_db": SNR_RANGE_DB}
    for method, vals in results.items():
        save_dict[f"{method}_ber"] = vals["ber"]
        save_dict[f"{method}_ser"] = vals["ser"]
    np.savez(npz_path, **save_dict)
    print(f"\nSaved results: {npz_path}")

    # Plot
    fig_path = os.path.join(FIGURES_DIR, "fig2_ber_ser.png")
    plot_ber_ser(results, SNR_RANGE_DB, fig_path)

    print("\nSimulation complete.")
    print(f"Results  : {npz_path}")
    print(f"Figure   : {fig_path}")


if __name__ == "__main__":
    main()
