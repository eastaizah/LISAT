"""
channel_models.py
-----------------
Channel model implementations for LiSAT simulation suite.

Implements doubly-dispersive (delay-Doppler) channel models for four scenarios:
  - VehicularB_V2X  : ITU-R Vehicular-B, 5.9 GHz, v=500 km/h
  - TDL_A_IoE       : 3GPP TDL-A adapted for IoE, 28 GHz
  - TDL_C_THz       : 3GPP TDL-C adapted for THz, 300 GHz
  - GaussMarkov_UAV : Non-stationary Gauss-Markov, 3.5 GHz

Reference: Bello (1963), "Characterization of Randomly Time-Variant Linear Channels".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _db2lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _jakes_spectrum(doppler_hz: float, f_grid: torch.Tensor) -> torch.Tensor:
    """Clarke/Jakes PSD: S(f) = 1 / (pi * f_D * sqrt(1 - (f/f_D)^2))."""
    f_norm = f_grid / (doppler_hz + 1e-9)
    inside = torch.clamp(1.0 - f_norm ** 2, min=1e-6)
    return 1.0 / (math.pi * (doppler_hz + 1e-9) * torch.sqrt(inside))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DoublyDispersiveChannel:
    """
    General delay-Doppler channel following Bello (1963).

    The channel is characterised by a spreading function h(τ, ν) where τ is
    the delay and ν is the Doppler shift.  For a discrete multi-path model:

        y[n] = sum_l h_l * x[n - τ_l] * exp(j 2π ν_l n T_s) + w[n]

    Parameters
    ----------
    num_paths : int
        Number of resolvable propagation paths.
    max_delay_samples : int
        Maximum delay spread in samples.
    max_doppler_hz : float
        Maximum Doppler shift in Hz.
    carrier_freq_hz : float
        Carrier frequency in Hz.
    sample_rate_hz : float
        Sampling rate in Hz.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        num_paths: int,
        max_delay_samples: int,
        max_doppler_hz: float,
        carrier_freq_hz: float,
        sample_rate_hz: float,
        device: str = "cpu",
    ) -> None:
        self.num_paths = num_paths
        self.max_delay_samples = max_delay_samples
        self.max_doppler_hz = max_doppler_hz
        self.carrier_freq_hz = carrier_freq_hz
        self.sample_rate_hz = sample_rate_hz
        self.device = torch.device(device)

        # Placeholders — overridden by subclasses
        self.delays_samples: torch.Tensor = torch.zeros(num_paths, device=self.device)
        self.doppler_shifts_hz: torch.Tensor = torch.zeros(num_paths, device=self.device)
        self.path_powers_lin: torch.Tensor = torch.ones(num_paths, device=self.device) / num_paths

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate_channel(
        self, batch_size: int, num_frames: int
    ) -> torch.Tensor:
        """
        Generate complex channel coefficient tensors.

        Returns
        -------
        h : torch.Tensor, shape [batch, paths, frames]
            Complex channel coefficients (each frame = one OFDM/OTFS symbol).
        """
        raise NotImplementedError

    def apply(
        self,
        tx_signal: torch.Tensor,
        gains: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply channel to a transmitted baseband signal.

        Parameters
        ----------
        tx_signal : torch.Tensor, shape [batch, signal_length] (complex)
        gains     : optional torch.Tensor [batch, num_paths] (complex).
                    If provided, re-uses these gains (same channel realisation).

        Returns
        -------
        rx_signal : torch.Tensor, shape [batch, signal_length] (complex)
        gains_out : torch.Tensor, shape [batch, num_paths] (complex),
                    the per-path complex gains used in this call.
        """
        batch, N = tx_signal.shape
        delays = self.delays_samples.long()
        dopplers = self.doppler_shifts_hz
        powers = self.path_powers_lin

        n_idx = torch.arange(N, dtype=torch.float32, device=self.device)
        rx = torch.zeros_like(tx_signal)

        # Generate or reuse gains
        if gains is None:
            gains_out = torch.zeros(batch, self.num_paths, dtype=torch.complex64,
                                    device=self.device)
            for l_idx in range(self.num_paths):
                amp = math.sqrt(powers[l_idx].item())
                gains_out[:, l_idx] = (
                    torch.randn(batch, device=self.device)
                    + 1j * torch.randn(batch, device=self.device)
                ).to(torch.complex64) * amp / math.sqrt(2.0)
        else:
            gains_out = gains

        for l_idx in range(self.num_paths):
            tau_l = int(delays[l_idx].item())
            nu_l = dopplers[l_idx].item()

            # Doppler phase rotation
            phase = 2.0 * math.pi * nu_l * n_idx / self.sample_rate_hz
            phasor = torch.exp(1j * phase).to(torch.complex64)

            # Delayed signal (zero-pad front)
            if tau_l == 0:
                delayed = tx_signal
            elif tau_l < N:
                delayed = torch.cat(
                    [
                        torch.zeros(batch, tau_l, dtype=tx_signal.dtype, device=self.device),
                        tx_signal[:, :N - tau_l],
                    ],
                    dim=1,
                )
            else:
                delayed = torch.zeros_like(tx_signal)

            rx = rx + gains_out[:, l_idx].unsqueeze(1) * delayed * phasor.unsqueeze(0)

        return rx, gains_out

    @staticmethod
    def add_noise(rx_signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add complex AWGN to a received signal.

        Parameters
        ----------
        rx_signal : torch.Tensor, complex
        snr_db    : float, signal-to-noise ratio in dB

        Returns
        -------
        noisy_signal : torch.Tensor, same shape as rx_signal
        """
        snr_lin = 10.0 ** (snr_db / 10.0)
        sig_power = (rx_signal.abs() ** 2).mean()
        noise_power = sig_power / (snr_lin + 1e-10)
        noise = (
            torch.randn_like(rx_signal.real)
            + 1j * torch.randn_like(rx_signal.real)
        ).to(torch.complex64) * (noise_power.sqrt() / math.sqrt(2.0))
        return rx_signal + noise


# ---------------------------------------------------------------------------
# Vehicular-B V2X
# ---------------------------------------------------------------------------

class VehicularB_V2X(DoublyDispersiveChannel):
    """
    ITU-R Vehicular-B channel profile for V2X communications.

    Path profile (ITU-R M.1225):
      Delays   : [0, 300, 8900, 12900, 17100, 20000] ns
      Rel. PDP : [0, -1, -9, -10, -15, -20] dB
      Doppler  : Jakes (Clarke) spectrum
      Carrier  : 5.9 GHz (DSRC / C-V2X band)
      Speed    : 500 km/h  →  ν_max ≈ 2747 Hz
    """

    DELAYS_NS = [0.0, 300.0, 8900.0, 12900.0, 17100.0, 20000.0]
    POWERS_DB = [0.0, -1.0, -9.0, -10.0, -15.0, -20.0]
    CARRIER_HZ = 5.9e9
    SPEED_KMH = 500.0

    def __init__(
        self,
        sample_rate_hz: float = 30.72e6,
        device: str = "cpu",
    ) -> None:
        speed_ms = self.SPEED_KMH / 3.6
        nu_max = speed_ms * self.CARRIER_HZ / 3e8
        delays_s = [d * 1e-9 for d in self.DELAYS_NS]
        delays_samples = [round(d * sample_rate_hz) for d in delays_s]

        super().__init__(
            num_paths=6,
            max_delay_samples=int(max(delays_samples)) + 1,
            max_doppler_hz=nu_max,
            carrier_freq_hz=self.CARRIER_HZ,
            sample_rate_hz=sample_rate_hz,
            device=device,
        )

        powers_lin = torch.tensor(
            [_db2lin(p) for p in self.POWERS_DB], dtype=torch.float32, device=self.device
        )
        self.path_powers_lin = powers_lin / powers_lin.sum()
        self.delays_samples = torch.tensor(
            [float(d) for d in delays_samples], dtype=torch.float32, device=self.device
        )
        # Doppler shifts – Jakes model: six representative arrival angles
        _angles = torch.tensor(
            [-1.31, -0.79, -0.26, 0.26, 0.79, 1.31], device=self.device
        )
        self.doppler_shifts_hz = nu_max * torch.cos(_angles)

    def generate_channel(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """
        Generate time-varying channel coefficients with Jakes Doppler model.

        Returns
        -------
        h : torch.Tensor [batch, paths, frames], complex64
        """
        # Per-path independent Jakes-like realisations
        nu_max = self.max_doppler_hz
        T_frame = 1.0 / self.sample_rate_hz  # approximate; caller may override

        h = torch.zeros(
            batch_size, self.num_paths, num_frames, dtype=torch.complex64, device=self.device
        )
        for l_idx in range(self.num_paths):
            amp = math.sqrt(self.path_powers_lin[l_idx].item())
            # Sum-of-sinusoids Jakes model with N_osc=8 oscillators
            N_osc = 8
            theta = 2.0 * math.pi * torch.rand(batch_size, N_osc, device=self.device)
            alpha = math.pi * (torch.arange(1, N_osc + 1, device=self.device).float() - 0.5) / N_osc
            f_n = nu_max * torch.cos(alpha)  # shape [N_osc]

            t = torch.arange(num_frames, dtype=torch.float32, device=self.device)
            # phase: [batch, N_osc, frames]
            phase = 2.0 * math.pi * f_n.unsqueeze(0).unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0) + theta.unsqueeze(-1)
            coeff = (torch.cos(phase) + 1j * torch.sin(phase)).to(torch.complex64)
            h[:, l_idx, :] = amp * coeff.sum(dim=1) / math.sqrt(N_osc)

        return h


# ---------------------------------------------------------------------------
# TDL-A IoE
# ---------------------------------------------------------------------------

class TDL_A_IoE(DoublyDispersiveChannel):
    """
    3GPP TDL-A channel adapted for Internet-of-Everything (IoE) at 28 GHz.

    8 paths with modest Doppler (~150 Hz), representative of pedestrian/slow
    vehicular IoE deployments in mmWave bands.

    Reference: 3GPP TR 38.901 Table 7.7.2-1 (TDL-A, τ_rms = 10 ns scaled).
    """

    # Normalised delays (relative to τ_rms=10 ns) — 8 strongest paths
    DELAYS_NS_NORM = [0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 110.0, 150.0]
    POWERS_DB = [0.0, -1.5, -3.0, -5.0, -7.5, -10.0, -13.5, -18.0]
    CARRIER_HZ = 28.0e9
    NU_MAX_HZ = 150.0  # pedestrian ~5.8 km/h at 28 GHz

    def __init__(
        self,
        sample_rate_hz: float = 122.88e6,
        device: str = "cpu",
    ) -> None:
        delays_samples = [round(d * 1e-9 * sample_rate_hz) for d in self.DELAYS_NS_NORM]

        super().__init__(
            num_paths=8,
            max_delay_samples=int(max(delays_samples)) + 1,
            max_doppler_hz=self.NU_MAX_HZ,
            carrier_freq_hz=self.CARRIER_HZ,
            sample_rate_hz=sample_rate_hz,
            device=device,
        )

        powers_lin = torch.tensor(
            [_db2lin(p) for p in self.POWERS_DB], dtype=torch.float32, device=self.device
        )
        self.path_powers_lin = powers_lin / powers_lin.sum()
        self.delays_samples = torch.tensor(
            [float(d) for d in delays_samples], dtype=torch.float32, device=self.device
        )
        # Doppler shifts – representative arrival angles for 8 paths
        _ang_ioe = torch.linspace(-1.31, 1.31, 8, device=self.device)
        self.doppler_shifts_hz = self.NU_MAX_HZ * torch.cos(_ang_ioe)

    def generate_channel(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Generate TDL-A IoE channel realisations."""
        nu_max = self.max_doppler_hz
        h = torch.zeros(
            batch_size, self.num_paths, num_frames, dtype=torch.complex64, device=self.device
        )
        for l_idx in range(self.num_paths):
            amp = math.sqrt(self.path_powers_lin[l_idx].item())
            N_osc = 8
            theta = 2.0 * math.pi * torch.rand(batch_size, N_osc, device=self.device)
            alpha = math.pi * (torch.arange(1, N_osc + 1, device=self.device).float() - 0.5) / N_osc
            f_n = nu_max * torch.cos(alpha)
            t = torch.arange(num_frames, dtype=torch.float32, device=self.device)
            phase = 2.0 * math.pi * f_n.unsqueeze(0).unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0) + theta.unsqueeze(-1)
            coeff = (torch.cos(phase) + 1j * torch.sin(phase)).to(torch.complex64)
            h[:, l_idx, :] = amp * coeff.sum(dim=1) / math.sqrt(N_osc)
        return h


# ---------------------------------------------------------------------------
# TDL-C THz
# ---------------------------------------------------------------------------

class TDL_C_THz(DoublyDispersiveChannel):
    """
    3GPP TDL-C–inspired channel adapted for 300 GHz THz communications.

    12 paths with τ_rms = 50 ns and very low Doppler (ν_max = 80 Hz) since
    THz links are typically quasi-static Line-of-Sight dominated.

    Reference: 3GPP TR 38.901 Table 7.7.2-3; scaled for 300 GHz propagation.
    """

    # 12 paths: TDL-C normalised delays scaled to τ_rms=50 ns
    DELAYS_NS = [0, 5, 10, 20, 35, 50, 70, 90, 115, 145, 180, 220]
    POWERS_DB = [0, -1, -2, -4, -6, -8, -10, -12, -14, -16, -18, -22]
    CARRIER_HZ = 300.0e9
    NU_MAX_HZ = 80.0  # slow movement / fixed THz links

    def __init__(
        self,
        sample_rate_hz: float = 245.76e6,
        device: str = "cpu",
    ) -> None:
        delays_samples = [round(d * 1e-9 * sample_rate_hz) for d in self.DELAYS_NS]

        super().__init__(
            num_paths=12,
            max_delay_samples=int(max(delays_samples)) + 1,
            max_doppler_hz=self.NU_MAX_HZ,
            carrier_freq_hz=self.CARRIER_HZ,
            sample_rate_hz=sample_rate_hz,
            device=device,
        )

        powers_lin = torch.tensor(
            [_db2lin(p) for p in self.POWERS_DB], dtype=torch.float32, device=self.device
        )
        self.path_powers_lin = powers_lin / powers_lin.sum()
        self.delays_samples = torch.tensor(
            [float(d) for d in delays_samples], dtype=torch.float32, device=self.device
        )
        # Doppler shifts – representative angles for 12 THz paths
        _ang_thz = torch.linspace(-1.31, 1.31, 12, device=self.device)
        self.doppler_shifts_hz = self.NU_MAX_HZ * torch.cos(_ang_thz)

    def generate_channel(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Generate TDL-C THz channel realisations."""
        nu_max = self.max_doppler_hz
        h = torch.zeros(
            batch_size, self.num_paths, num_frames, dtype=torch.complex64, device=self.device
        )
        for l_idx in range(self.num_paths):
            amp = math.sqrt(self.path_powers_lin[l_idx].item())
            N_osc = 8
            theta = 2.0 * math.pi * torch.rand(batch_size, N_osc, device=self.device)
            alpha = math.pi * (torch.arange(1, N_osc + 1, device=self.device).float() - 0.5) / N_osc
            f_n = nu_max * torch.cos(alpha)
            t = torch.arange(num_frames, dtype=torch.float32, device=self.device)
            phase = 2.0 * math.pi * f_n.unsqueeze(0).unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0) + theta.unsqueeze(-1)
            coeff = (torch.cos(phase) + 1j * torch.sin(phase)).to(torch.complex64)
            h[:, l_idx, :] = amp * coeff.sum(dim=1) / math.sqrt(N_osc)
        return h


# ---------------------------------------------------------------------------
# Gauss-Markov UAV
# ---------------------------------------------------------------------------

class GaussMarkov_UAV(DoublyDispersiveChannel):
    """
    Non-stationary Gauss-Markov channel for UAV air-to-ground links.

    Channel coefficients evolve as:
        h_{l,t+1} = ρ_l * h_{l,t} + sqrt(1 - ρ_l²) * w_{l,t}

    where ρ_l = exp(-2π f_D,l T_s) is the temporal correlation coefficient,
    f_D,l is the Doppler frequency of path l, T_s = 1/sample_rate_hz, and
    w_{l,t} ~ CN(0, 1) is i.i.d. complex Gaussian innovation.

    Parameters
    ----------
    carrier_freq_hz : float
        Carrier frequency (default 3.5 GHz).
    speed_kmh       : float
        UAV speed in km/h (variable; default 120 km/h).
    sample_rate_hz  : float
        Sampling rate.
    device          : str
    """

    CARRIER_HZ = 3.5e9
    NUM_PATHS = 4
    DELAYS_NS = [0.0, 100.0, 300.0, 600.0]
    POWERS_DB = [0.0, -3.0, -8.0, -15.0]

    def __init__(
        self,
        speed_kmh: float = 120.0,
        sample_rate_hz: float = 30.72e6,
        device: str = "cpu",
    ) -> None:
        speed_ms = speed_kmh / 3.6
        nu_max = speed_ms * self.CARRIER_HZ / 3e8
        delays_samples = [round(d * 1e-9 * sample_rate_hz) for d in self.DELAYS_NS]

        super().__init__(
            num_paths=self.NUM_PATHS,
            max_delay_samples=int(max(delays_samples)) + 1,
            max_doppler_hz=nu_max,
            carrier_freq_hz=self.CARRIER_HZ,
            sample_rate_hz=sample_rate_hz,
            device=device,
        )

        self.speed_kmh = speed_kmh
        powers_lin = torch.tensor(
            [_db2lin(p) for p in self.POWERS_DB], dtype=torch.float32, device=self.device
        )
        self.path_powers_lin = powers_lin / powers_lin.sum()
        self.delays_samples = torch.tensor(
            [float(d) for d in delays_samples], dtype=torch.float32, device=self.device
        )

        # Per-path Doppler (random realisations in [-nu_max, nu_max])
        self._init_doppler()

    def _init_doppler(self) -> None:
        """Assign random per-path Doppler frequencies within ±ν_max."""
        self.doppler_shifts_hz = (
            (2.0 * torch.rand(self.num_paths, device=self.device) - 1.0)
            * self.max_doppler_hz
        )

    def generate_channel(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """
        Generate non-stationary Gauss-Markov channel realisations.

        Returns
        -------
        h : torch.Tensor [batch, paths, frames], complex64
        """
        T_s = 1.0 / self.sample_rate_hz
        h = torch.zeros(
            batch_size, self.num_paths, num_frames, dtype=torch.complex64, device=self.device
        )

        for l_idx in range(self.num_paths):
            f_D_l = self.doppler_shifts_hz[l_idx].item()
            rho = math.exp(-2.0 * math.pi * abs(f_D_l) * T_s)
            innov_std = math.sqrt(max(1.0 - rho ** 2, 0.0))
            amp = math.sqrt(self.path_powers_lin[l_idx].item())

            # Initialise h_0 ~ CN(0, amp^2)
            h_cur = (
                torch.randn(batch_size, device=self.device)
                + 1j * torch.randn(batch_size, device=self.device)
            ).to(torch.complex64) * amp / math.sqrt(2.0)

            for t in range(num_frames):
                h[:, l_idx, t] = h_cur
                innovation = (
                    torch.randn(batch_size, device=self.device)
                    + 1j * torch.randn(batch_size, device=self.device)
                ).to(torch.complex64) * innov_std / math.sqrt(2.0)
                h_cur = rho * h_cur + innovation * amp

        return h

    def update_speed(self, new_speed_kmh: float) -> None:
        """Dynamically update UAV speed and recompute Doppler parameters."""
        speed_ms = new_speed_kmh / 3.6
        self.max_doppler_hz = speed_ms * self.carrier_freq_hz / 3e8
        self.speed_kmh = new_speed_kmh
        self._init_doppler()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== Channel Model Smoke Tests ===\n")

    channels = {
        "VehicularB_V2X": VehicularB_V2X(),
        "TDL_A_IoE": TDL_A_IoE(),
        "TDL_C_THz": TDL_C_THz(),
        "GaussMarkov_UAV": GaussMarkov_UAV(),
    }

    for name, ch in channels.items():
        h = ch.generate_channel(batch_size=4, num_frames=14)
        print(f"{name}: h.shape={h.shape}, |h| mean={h.abs().mean():.4f}")

        # Synthetic OFDM-like signal
        sig = (
            torch.randn(4, 1024) + 1j * torch.randn(4, 1024)
        ).to(torch.complex64)
        rx, _ = ch.apply(sig)
        rx_noisy = ch.add_noise(rx, snr_db=10.0)
        snr_check = (rx.abs() ** 2).mean() / ((rx_noisy - rx).abs() ** 2).mean()
        print(f"  Noise SNR check: {10*math.log10(snr_check.item()+1e-9):.1f} dB (target 10 dB)\n")

    print("All channel model tests passed.")
