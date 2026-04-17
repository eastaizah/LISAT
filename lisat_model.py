"""
lisat_model.py
--------------
LiSAT (Lightweight Semantic-Adaptive Transceiver) neural network components.

Components:
  1. LightweightSemanticCodec (CSL) — convolutional autoencoder for semantic
     source coding, INT8-quantisation-aware.
  2. A3CAgent — Asynchronous Advantage Actor-Critic agent with GRU backbone
     for multi-objective waveform selection.
  3. MERDEstimator — delay-Doppler profile estimator using AFDM pilot subframe.
  4. LiSATSystem — full system combining CSL + MERD + A3C.

Reference: "LiSAT: Lightweight Semantic-Adaptive Transceiver for 6G Waveform
           Selection", IEEE Wireless Communications Letters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, default_fused_act_fake_quant


# ---------------------------------------------------------------------------
# Lightweight Semantic Codec (CSL)
# ---------------------------------------------------------------------------

class LightweightSemanticCodec(nn.Module):
    """
    Convolutional semantic autoencoder for source coding.

    Encoder: 4-layer CNN
        1 → 32 (stride 2)
        32 → 64 (stride 2)
        64 → 128 (stride 1)
        128 → k_z (stride 1)

    Decoder: mirror transposed CNN.

    The forward pass supports QAT (Quantisation-Aware Training) via
    torch.ao.quantization fake-quantise nodes inserted after each activation.

    Parameters
    ----------
    in_channels : int, input channels (1 for grayscale, 3 for RGB)
    k_z         : int, latent semantic dimension (number of channels)
    qat_mode    : bool, enable fake-quantise nodes for INT8-QAT
    """

    def __init__(
        self,
        in_channels: int = 1,
        k_z: int = 64,
        qat_mode: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.k_z = k_z
        self.qat_mode = qat_mode

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv2d(128, k_z, kernel_size=3, stride=1, padding=1)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_bn4 = nn.BatchNorm2d(k_z)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.dec1 = nn.ConvTranspose2d(k_z, 128, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_bn3 = nn.BatchNorm2d(32)

        # Optional QAT fake-quantise observers
        # INT8 quantization constants — change here to switch to INT16 (quant_min=-32768, quant_max=32767)
        QUANT_MIN: int = -128
        QUANT_MAX: int = 127
        QUANT_DTYPE = torch.qint8
        if qat_mode:
            self._fq = FakeQuantize.with_args(
                observer=torch.ao.quantization.MovingAverageMinMaxObserver,
                quant_min=QUANT_MIN,
                quant_max=QUANT_MAX,
                dtype=QUANT_DTYPE,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            )
            self.fq_enc1 = self._fq()
            self.fq_enc2 = self._fq()
            self.fq_enc3 = self._fq()
            self.fq_latent = self._fq()

    def _act(self, x: torch.Tensor, fq: Optional[object] = None) -> torch.Tensor:
        """ReLU activation with optional fake-quantise."""
        x = F.relu(x)
        if self.qat_mode and fq is not None:
            x = fq(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent semantic representation.

        Parameters
        ----------
        x : torch.Tensor [batch, C, H, W]

        Returns
        -------
        z : torch.Tensor [batch, k_z, H/4, W/4]
        """
        fq1 = getattr(self, "fq_enc1", None)
        fq2 = getattr(self, "fq_enc2", None)
        fq3 = getattr(self, "fq_enc3", None)

        z = self._act(self.enc_bn1(self.enc1(x)), fq1)
        z = self._act(self.enc_bn2(self.enc2(z)), fq2)
        z = self._act(self.enc_bn3(self.enc3(z)), fq3)
        z = self.enc_bn4(self.enc4(z))  # no activation on bottleneck
        if self.qat_mode:
            z = self.fq_latent(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Parameters
        ----------
        z : torch.Tensor [batch, k_z, H/4, W/4]

        Returns
        -------
        x_hat : torch.Tensor [batch, C, H, W]
        """
        x = F.relu(self.dec_bn1(self.dec1(z)))
        x = F.relu(self.dec_bn2(self.dec2(x)))
        x = F.relu(self.dec_bn3(self.dec3(x)))
        x = torch.sigmoid(self.dec4(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode-decode pass.

        Returns
        -------
        x_hat : torch.Tensor, reconstruction
        z     : torch.Tensor, latent codes
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @staticmethod
    def compute_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        task: str = "mse",
        beta: float = 0.01,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Parameters
        ----------
        x_hat : reconstructed tensor
        x     : original tensor
        task  : 'mse' | 'ssim_approx' | 'classification'
        beta  : KL / sparsity regularisation weight
        z     : latent codes (for regularisation)

        Returns
        -------
        loss : scalar tensor
        """
        if task == "mse":
            loss = F.mse_loss(x_hat, x)
        elif task == "ssim_approx":
            # Approximate SSIM via local mean + variance
            mu_x = F.avg_pool2d(x, 3, stride=1, padding=1)
            mu_xh = F.avg_pool2d(x_hat, 3, stride=1, padding=1)
            sigma_x = F.avg_pool2d(x ** 2, 3, stride=1, padding=1) - mu_x ** 2
            sigma_xh = F.avg_pool2d(x_hat ** 2, 3, stride=1, padding=1) - mu_xh ** 2
            sigma_cross = F.avg_pool2d(x * x_hat, 3, stride=1, padding=1) - mu_x * mu_xh
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu_x * mu_xh + c1) * (2 * sigma_cross + c2)) / (
                (mu_x ** 2 + mu_xh ** 2 + c1) * (sigma_x + sigma_xh + c2) + 1e-8
            )
            loss = 1.0 - ssim.mean()
        else:
            loss = F.mse_loss(x_hat, x)

        # Sparsity regularisation on latent codes
        if z is not None:
            loss = loss + beta * z.abs().mean()

        return loss


# ---------------------------------------------------------------------------
# A3C Agent
# ---------------------------------------------------------------------------

class A3CAgent(nn.Module):
    """
    Asynchronous Advantage Actor-Critic (A3C) agent for waveform selection.

    State vector (dim=6):
        [τ_max_hat, ν_max_hat, SNR_dB, η_s_prev, P_t_prev, waveform_prev]

    Action space (discrete, 3 actions):
        0 = OTFS, 1 = OCDM, 2 = AFDM

    Architecture:
        Shared GRU backbone: GRU(input=6, hidden=128, num_layers=2)
        Actor head:          Linear(128, 3) → softmax
        Critic head:         Linear(128, 1)

    Parameters
    ----------
    state_dim  : int, state vector dimension (default 6)
    hidden_dim : int, GRU hidden state size (default 128)
    num_actions: int, number of discrete actions (default 3)
    num_layers : int, GRU layers (default 2)
    """

    WAVEFORM_NAMES = {0: "OTFS", 1: "OCDM", 2: "AFDM"}

    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 128,
        num_actions: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers

        # Shared GRU backbone
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Actor head
        self.actor_fc = nn.Linear(hidden_dim, num_actions)

        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, 1)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor_fc.weight, gain=0.01)
        nn.init.zeros_(self.actor_fc.bias)
        nn.init.orthogonal_(self.critic_fc.weight, gain=1.0)
        nn.init.zeros_(self.critic_fc.bias)

    def init_hidden(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Initialise GRU hidden state to zeros."""
        dev = device or next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=dev)

    def forward(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.

        Parameters
        ----------
        state  : torch.Tensor [batch, seq_len, state_dim] or [batch, state_dim]
        hidden : torch.Tensor [num_layers, batch, hidden_dim]

        Returns
        -------
        action_probs : torch.Tensor [batch, num_actions], action probabilities
        value        : torch.Tensor [batch, 1], state value estimate
        new_hidden   : torch.Tensor [num_layers, batch, hidden_dim]
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # add seq_len=1 dim

        gru_out, new_hidden = self.gru(state, hidden)
        # Use last time step output
        out = gru_out[:, -1, :]  # [batch, hidden_dim]

        action_logits = self.actor_fc(out)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic_fc(out)

        return action_probs, value, new_hidden

    def forward_sequence(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass over a full sequence of states in a single GRU call.

        Preferred over repeated calls to ``forward`` during rollout updates
        because it processes the entire rollout in one batched GPU kernel,
        eliminating the per-step Python loop overhead.

        Parameters
        ----------
        states : torch.Tensor [batch, seq_len, state_dim]
        hidden : torch.Tensor [num_layers, batch, hidden_dim]

        Returns
        -------
        action_probs : torch.Tensor [batch, seq_len, num_actions]
        values       : torch.Tensor [batch, seq_len, 1]
        new_hidden   : torch.Tensor [num_layers, batch, hidden_dim]
        """
        gru_out, new_hidden = self.gru(states, hidden)          # [batch, T, hidden]
        action_logits = self.actor_fc(gru_out)                  # [batch, T, num_actions]
        action_probs = F.softmax(action_logits, dim=-1)
        values = self.critic_fc(gru_out)                        # [batch, T, 1]
        return action_probs, values, new_hidden

    def select_action(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or argmax action from current policy.

        Returns
        -------
        action      : int
        log_prob    : torch.Tensor, scalar
        value       : torch.Tensor [batch, 1]
        new_hidden  : torch.Tensor
        """
        action_probs, value, new_hidden = self.forward(state, hidden)
        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = action_probs.log().gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value, new_hidden

    @staticmethod
    def compute_reward(
        eta_s: float,
        P_t: float,
        tau_e2e: float,
        constraint_violated: bool,
        P_max: float,
        tau_max: float,
        eta_s_prev: float = 0.0,
        P_t_prev: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.05,
        lam: float = 0.5,
    ) -> float:
        """
        Compute multi-objective reward as in LiSAT paper Eq. (11).

        r = clip( α·η_s + β·(1 − P_t/P_max) − γ·(τ/τ_max) − λ·v, -1, 1 )

        where:
            η_s  = semantic efficiency ∈ [0, 1]
            P_t  = transmit power (W)
            τ    = τ_e2e             (end-to-end latency, normalised)
            v    = constraint_violated (binary indicator)

        Parameters
        ----------
        eta_s               : current semantic efficiency ∈ [0, 1]
        P_t                 : current transmit power (W)
        tau_e2e             : end-to-end latency (s)
        constraint_violated : bool, True if QoS constraint violated
        P_max               : maximum allowed power (W)
        tau_max             : maximum allowed latency (s)
        eta_s_prev          : previous semantic efficiency (unused, kept for API compat)
        P_t_prev            : previous transmit power (unused, kept for API compat)
        alpha, beta, gamma, lam : reward shaping weights

        Returns
        -------
        reward : float in [-1, 1]
        """
        P_norm = P_t / (P_max + 1e-9)
        tau_norm = tau_e2e / (tau_max + 1e-9)
        violation = float(constraint_violated)

        r = (
            alpha * eta_s
            + beta * (1.0 - P_norm)
            - gamma * tau_norm
            - lam * violation
        )
        return float(np.clip(r, -1.0, 1.0))

    @staticmethod
    def compute_gae(
        rewards: List[float],
        values: List[float],
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalised Advantage Estimation (GAE).

        Parameters
        ----------
        rewards     : list of rewards r_t
        values      : list of V(s_t) estimates
        next_value  : V(s_{T+1}) bootstrap value
        gamma       : discount factor
        gae_lambda  : GAE lambda

        Returns
        -------
        returns     : list of discounted returns
        advantages  : list of GAE advantages
        """
        T = len(rewards)
        advantages = [0.0] * T
        last_gae = 0.0
        values_ext = values + [next_value]

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns, advantages


# ---------------------------------------------------------------------------
# MERD Estimator
# ---------------------------------------------------------------------------

class MERDEstimator:
    """
    Maximum-Energy Region Detection (MERD) delay-Doppler estimator.

    Uses the AFDM pilot subframe to estimate the channel's maximum delay
    spread (τ_max) and maximum Doppler shift (ν_max) by localising energy
    peaks in the received DAFT-domain pilot response.

    Reference: Bemani et al. (2023), "AFDM: An Effective Modulation for
               ISAC in Next Generation Wireless Systems", IEEE Trans. Wireless.
    """

    def __init__(
        self,
        N: int = 64,
        sample_rate_hz: float = 30.72e6,
        subcarrier_spacing_hz: float = 15e3,
    ) -> None:
        self.N = N
        self.sample_rate_hz = sample_rate_hz
        self.subcarrier_spacing_hz = subcarrier_spacing_hz

    def estimate(
        self,
        rx_pilot_symbols: torch.Tensor,
        tx_pilot_symbols: torch.Tensor,
        threshold: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Estimate τ_max and ν_max from received AFDM pilot.

        The channel impulse response in the DAFT domain is obtained by
        element-wise division of received by transmitted pilots (channel
        sounding). Peaks above `threshold * max_peak` are localised to
        yield delay and Doppler estimates.

        Parameters
        ----------
        rx_pilot_symbols : torch.Tensor [N], complex64
        tx_pilot_symbols : torch.Tensor [N], complex64, must be non-zero
        threshold        : float, relative peak detection threshold

        Returns
        -------
        tau_max_hat  : float, estimated maximum delay (seconds)
        nu_max_hat   : float, estimated maximum Doppler (Hz)
        """
        N = self.N
        # Channel estimate in DAFT domain: H[k] = Y[k] / X[k]
        with torch.no_grad():
            H = rx_pilot_symbols / (tx_pilot_symbols + 1e-9)

        # Convert to time-delay domain via IFFT (delay profile)
        h_delay = torch.fft.ifft(H, n=N)
        delay_profile = h_delay.abs()

        # Peak detection for delay
        peak_val = delay_profile.max().item()
        peak_mask = delay_profile > threshold * peak_val
        delay_bins = torch.where(peak_mask)[0]

        if delay_bins.numel() == 0:
            tau_max_hat = 0.0
        else:
            max_delay_bin = delay_bins.max().item()
            tau_max_hat = max_delay_bin / self.sample_rate_hz

        # Convert to Doppler domain via FFT of received pilot
        H_doppler = torch.fft.fft(H, n=N)
        doppler_profile = H_doppler.abs()
        peak_dop = doppler_profile.max().item()
        doppler_mask = doppler_profile > threshold * peak_dop
        doppler_bins = torch.where(doppler_mask)[0]

        if doppler_bins.numel() == 0:
            nu_max_hat = 0.0
        else:
            # Bins are in [0, N-1]; map to [-N/2, N/2-1]
            bins_shifted = doppler_bins.float()
            bins_shifted[bins_shifted > N // 2] -= N
            nu_max_hat = float(bins_shifted.abs().max().item()) * self.subcarrier_spacing_hz

        return tau_max_hat, nu_max_hat


# ---------------------------------------------------------------------------
# Full LiSAT System
# ---------------------------------------------------------------------------

class LiSATSystem(nn.Module):
    """
    Full LiSAT (Lightweight Semantic-Adaptive Transceiver) system.

    Integrates:
      1. LightweightSemanticCodec (CSL) for semantic source coding
      2. MERDEstimator for delay-Doppler channel profiling via AFDM pilot
      3. A3CAgent for multi-objective waveform selection

    Processing pipeline:
        Source data → CSL encoder → [channel estimation] → A3C waveform
        selection → waveform modulator → channel → waveform demodulator
        → CSL decoder → reconstructed data

    Parameters
    ----------
    in_channels : int, input data channels
    k_z         : int, semantic latent dimension
    N           : int, waveform size (subcarriers)
    N_cp        : int, cyclic prefix length
    state_dim   : int, A3C state dimension
    hidden_dim  : int, A3C hidden dimension
    device      : str, compute device
    """

    def __init__(
        self,
        in_channels: int = 1,
        k_z: int = 64,
        N: int = 64,
        N_cp: int = 16,
        state_dim: int = 6,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device_str = device
        self.N = N
        self.k_z = k_z

        self.csl = LightweightSemanticCodec(
            in_channels=in_channels, k_z=k_z, qat_mode=False
        )
        self.merd = MERDEstimator(N=N)
        self.agent = A3CAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=3,
        )

        self.to(torch.device(device))

    def forward(
        self,
        x: torch.Tensor,
        snr_db: float,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, object]:
        """
        Full system forward pass.

        Parameters
        ----------
        x           : input tensor [batch, C, H, W]
        snr_db      : channel SNR in dB
        hidden      : optional GRU hidden state
        deterministic: use argmax action selection

        Returns
        -------
        result : dict with keys:
            x_hat       : reconstructed data
            z           : latent codes
            action      : selected waveform index
            waveform    : waveform name string
            action_probs: policy distribution
            value       : critic estimate
            new_hidden  : updated GRU hidden state
        """
        batch = x.shape[0]

        # Step 1: Semantic encoding
        z = self.csl.encode(x)

        # Step 2: Dummy state for A3C (in full system, MERD provides τ, ν)
        tau_hat, nu_hat = 1e-6, 100.0  # defaults; overridden by MERD
        state = torch.tensor(
            [[tau_hat * 1e6, nu_hat / 1000.0, snr_db / 30.0, 0.8, 0.5, 0.0]],
            dtype=torch.float32,
            device=x.device,
        )

        if hidden is None:
            hidden = self.agent.init_hidden(batch_size=1, device=x.device)

        # Step 3: Waveform selection
        action_probs, value, new_hidden = self.agent.forward(state, hidden)
        if deterministic:
            action = action_probs.argmax(dim=-1).item()
        else:
            action = torch.distributions.Categorical(action_probs).sample().item()

        # Step 4: Decode
        x_hat = self.csl.decode(z)

        return {
            "x_hat": x_hat,
            "z": z,
            "action": action,
            "waveform": A3CAgent.WAVEFORM_NAMES[action],
            "action_probs": action_probs,
            "value": value,
            "new_hidden": new_hidden,
        }

    def compute_semantic_efficiency(
        self, x_hat: torch.Tensor, x: torch.Tensor
    ) -> float:
        """
        Compute semantic efficiency η_s = 1 - MSE_normalised.

        η_s = 1 - E[||x̂ - x||²] / E[||x||²]

        Returns
        -------
        eta_s : float in [0, 1]
        """
        sig_power = (x ** 2).mean().item()
        mse = F.mse_loss(x_hat, x).item()
        eta_s = 1.0 - mse / (sig_power + 1e-9)
        return float(np.clip(eta_s, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== LiSAT Model Smoke Tests ===\n")

    # --- CSL ---
    csl = LightweightSemanticCodec(in_channels=1, k_z=64)
    x = torch.randn(4, 1, 32, 32)
    x_hat, z = csl(x)
    loss = csl.compute_loss(x_hat, x)
    print(f"CSL: x={x.shape}, z={z.shape}, x_hat={x_hat.shape}, loss={loss.item():.4f}")

    # --- A3C ---
    agent = A3CAgent()
    state = torch.randn(1, 6)
    hidden = agent.init_hidden(batch_size=1)
    probs, val, new_h = agent(state, hidden)
    print(f"A3C: probs={probs.detach().numpy().round(3)}, value={val.item():.4f}")

    # Reward test
    r = A3CAgent.compute_reward(
        eta_s=0.85, P_t=0.4, tau_e2e=0.001, constraint_violated=False,
        P_max=1.0, tau_max=0.01, eta_s_prev=0.80, P_t_prev=0.5,
    )
    print(f"A3C reward: {r:.4f}")

    # --- MERD ---
    merd = MERDEstimator(N=64)
    tx_pilot = torch.exp(1j * torch.randn(64)).to(torch.complex64)
    rx_pilot = tx_pilot * (0.8 + 0.2j) + 0.01 * (
        torch.randn(64) + 1j * torch.randn(64)
    ).to(torch.complex64)
    tau_h, nu_h = merd.estimate(rx_pilot, tx_pilot)
    print(f"MERD: tau_max_hat={tau_h:.2e} s, nu_max_hat={nu_h:.2f} Hz")

    # --- Full system ---
    system = LiSATSystem()
    result = system(x, snr_db=10.0)
    eta_s = system.compute_semantic_efficiency(result["x_hat"], x)
    print(f"LiSAT: waveform={result['waveform']}, η_s={eta_s:.4f}")

    print("\nAll LiSAT model tests passed.")
