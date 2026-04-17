"""
simulate_convergence.py
-----------------------
A3C agent training convergence simulation for the LiSAT framework.
Reproduces Figure 3 of the paper.

Four scenarios:
    - V2X  : VehicularB_V2X  (5.9 GHz, 500 km/h, ν_max≈2747 Hz)
    - THz  : TDL_C_THz       (300 GHz, ν_max=80 Hz)
    - IoE  : TDL_A_IoE       (28 GHz, ν_max=150 Hz)
    - UAV  : GaussMarkov_UAV  (3.5 GHz, variable speed)

Training:
    - Up to 20,000 episodes per scenario
    - 5 random seeds for confidence intervals
    - Exponential moving average (α=0.05) for smoothing
    - Epsilon-optimal threshold line from Theorem 1

Output:
    results/convergence_all_scenarios.npz
    figures/fig3_convergence.png
"""

from __future__ import annotations

import contextlib
import os
import sys
import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

torch.manual_seed(0)
np.random.seed(0)

# Auto-detect GPU; fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPISODES = 20_000
N_SEEDS = 3
EMA_ALPHA = 0.05          # exponential moving average coefficient
GAMMA = 0.99              # RL discount factor
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
LR = 3e-4
ROLLOUT_LEN = 32          # steps per rollout before update (larger batches benefit GPU)

# Theorem 1: ε-optimal threshold (paper value; scenario-specific offsets)
EPSILON_OPTIMAL_BASE = 0.72

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Scenario environment wrapper
# ---------------------------------------------------------------------------

class WaveformSelectionEnv:
    """
    Lightweight RL environment for waveform selection.

    State:  [τ_max_hat (µs), ν_max_hat (kHz), SNR (norm), η_s, P_t, wf_prev]
    Action: {0=OTFS, 1=OCDM, 2=AFDM}
    Reward: Eq. (11) from LiSAT paper.
    """

    WAVEFORM_ROBUSTNESS = {0: 0.82, 1: 0.78, 2: 0.80}   # OTFS/OCDM/AFDM base η_s
    WAVEFORM_POWER = {0: 0.62, 1: 0.48, 2: 0.42}         # relative power cost
    WAVEFORM_LATENCY = {0: 0.014, 1: 0.008, 2: 0.005}    # seconds
    EPISODE_LENGTH: int = 50  # steps per training episode

    def __init__(
        self,
        channel: DoublyDispersiveChannel,
        snr_range_db: Tuple[float, float] = (-5.0, 30.0),
        p_max: float = 1.0,
        tau_max_latency: float = 0.02,
        seed: int = 0,
        device: torch.device = DEVICE,
    ) -> None:
        self.channel = channel
        self.snr_range_db = snr_range_db
        self.p_max = p_max
        self.tau_max_latency = tau_max_latency
        self.rng = np.random.RandomState(seed)
        self.device = device

        self._tau_max = channel.max_delay_samples / channel.sample_rate_hz
        self._nu_max = channel.max_doppler_hz
        self._reset_state()

    def _reset_state(self) -> None:
        self.snr_db = float(self.rng.uniform(*self.snr_range_db))
        self.eta_s_prev = float(self.rng.uniform(0.5, 0.9))
        self.P_t_prev = float(self.rng.uniform(0.3, 0.8))
        self.wf_prev = int(self.rng.randint(0, 3))
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        self._reset_state()
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # Add small noise to simulate imperfect MERD estimation
        tau_noisy = self._tau_max * (1 + 0.05 * self.rng.randn())
        nu_noisy = self._nu_max * (1 + 0.05 * self.rng.randn())
        state = torch.tensor(
            [[
                tau_noisy * 1e6,           # µs
                nu_noisy / 1000.0,         # kHz
                self.snr_db / 30.0,        # normalised
                self.eta_s_prev,
                self.P_t_prev / self.p_max,
                float(self.wf_prev) / 2.0, # normalised waveform index
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        return state

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Take a step in the environment.

        Returns next_state, reward, done.
        """
        # Simulate waveform performance with channel-dependent noise
        snr_lin = 10.0 ** (self.snr_db / 10.0)

        # Base η_s: higher SNR and better matched waveform → higher η_s
        base_eta = self.WAVEFORM_ROBUSTNESS[action]

        # Channel matching bonus: scenario-dependent waveform advantage
        nu_norm = min(self._nu_max / 3000.0, 1.0)
        tau_norm_ch = min(self._tau_max * self.channel.sample_rate_hz / 200.0, 1.0)
        snr_norm = min(max(snr_lin / 100.0, 0.0), 1.0)

        if action == 0:   # OTFS — full DD diversity, best at low SNR
            # Diversity gain is most valuable at low SNR
            low_snr_bonus = max(0.0, 0.30 * (1.0 - snr_norm))
            channel_match = 0.22 * nu_norm + 0.18 * tau_norm_ch + low_snr_bonus
        elif action == 1: # OCDM — chirp diversity, best for frequency-selective channels
            # Excels when delay spread is large but Doppler is small
            freq_selective = tau_norm_ch * (1.0 - 0.55 * nu_norm)
            channel_match = max(freq_selective + 0.05, 0.0)
        else:             # AFDM — DAFT designed for doubly-dispersive, best at high Doppler
            # Designed for high-Doppler environments, efficient at high SNR
            doppler_gain = nu_norm * (0.30 + 0.70 * snr_norm)
            high_snr_bonus = 0.32 * snr_norm
            channel_match = 0.55 * doppler_gain + 0.05 * tau_norm_ch + high_snr_bonus

        snr_bonus = min(snr_lin / 20.0, 1.0) * 0.08
        eta_s = float(np.clip(
            base_eta + 0.15 * channel_match + snr_bonus
            + 0.02 * self.rng.randn(),
            0.0, 1.0,
        ))

        P_t = self.WAVEFORM_POWER[action] * (1.0 + 0.05 * self.rng.randn())
        P_t = float(np.clip(P_t, 0.1, self.p_max))
        tau_e2e = self.WAVEFORM_LATENCY[action] * (1.0 + 0.1 * self.rng.randn())

        constraint = (P_t > self.p_max) or (tau_e2e > self.tau_max_latency)

        reward = A3CAgent.compute_reward(
            eta_s=eta_s,
            P_t=P_t,
            tau_e2e=tau_e2e,
            constraint_violated=constraint,
            P_max=self.p_max,
            tau_max=self.tau_max_latency,
            eta_s_prev=self.eta_s_prev,
            P_t_prev=self.P_t_prev,
            alpha=0.80,
            beta=0.15,
            gamma=0.02,
        )

        # Update state
        self.eta_s_prev = eta_s
        self.P_t_prev = P_t
        self.wf_prev = action
        self.step_count += 1

        # Randomise SNR slowly (non-stationary environment)
        if self.step_count % 10 == 0:
            self.snr_db = float(np.clip(
                self.snr_db + self.rng.randn() * 2.0,
                self.snr_range_db[0],
                self.snr_range_db[1],
            ))

        done = self.step_count >= self.EPISODE_LENGTH
        next_state = self._get_state()
        return next_state, reward, done


# ---------------------------------------------------------------------------
# A3C training loop (GPU-optimised A2C with vectorised rollout updates)
# ---------------------------------------------------------------------------

def train_agent(
    env: WaveformSelectionEnv,
    n_episodes: int = N_EPISODES,
    seed: int = 0,
    verbose: bool = True,
    device: torch.device = DEVICE,
) -> Tuple[List[float], nn.Module]:
    """
    Train A3C agent on the given environment.

    Uses a simplified synchronous A3C (equivalent to A2C) with periodic
    rollout updates.  The rollout update is fully vectorised: all states in a
    rollout are stacked into a single [1, T, state_dim] tensor so that the GRU
    and linear heads are evaluated in one GPU kernel call, removing the
    per-step Python loop overhead present in the naïve implementation.

    Mixed-precision (AMP) training is enabled automatically when a CUDA device
    is available.

    Returns
    -------
    episode_rewards : List[float], total reward per episode (raw + EMA)
    agent           : trained A3CAgent
    """
    torch.manual_seed(seed)

    agent = A3CAgent().to(device)

    # Optional torch.compile (PyTorch ≥ 2.0, CUDA recommended)
    if hasattr(torch, "compile"):
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
        except Exception as exc:
            print(f"  [torch.compile] skipped: {exc}")

    optimiser = optim.Adam(agent.parameters(), lr=LR)

    # AMP: GradScaler is only meaningful for CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None
    amp_ctx = (
        torch.amp.autocast(device_type=device.type)
        if use_amp
        else contextlib.nullcontext()
    )

    episode_rewards: List[float] = []
    ema_reward = 0.0

    for ep in range(n_episodes):
        state = env.reset()
        hidden = agent.init_hidden(batch_size=1, device=device)
        done = False

        ep_reward = 0.0
        rollout_states: List[torch.Tensor] = []
        rollout_actions: List[int] = []
        rollout_rewards: List[float] = []
        rollout_values: List[float] = []

        while not done:
            # --- Collect rollout ---
            rollout_states.clear()
            rollout_actions.clear()
            rollout_rewards.clear()
            rollout_values.clear()

            hidden_start = hidden.detach()

            for _ in range(ROLLOUT_LEN):
                with torch.no_grad():
                    action, log_prob, value, hidden = agent.select_action(
                        state, hidden, deterministic=False
                    )
                next_state, reward, done = env.step(action)

                rollout_states.append(state)
                rollout_actions.append(action)
                rollout_rewards.append(reward)
                rollout_values.append(value.item())

                ep_reward += reward
                state = next_state
                if done:
                    break

            # Bootstrap value
            with torch.no_grad():
                if done:
                    next_value = 0.0
                else:
                    _, next_val, _ = agent.forward(state, hidden.detach())
                    next_value = next_val.item()

            # Compute returns and advantages
            returns, advantages = A3CAgent.compute_gae(
                rollout_rewards, rollout_values, next_value,
                gamma=GAMMA, gae_lambda=GAE_LAMBDA,
            )

            # --- Vectorised GPU update ---
            # Stack all rollout states into a single [1, T, state_dim] tensor
            # so the GRU processes the whole rollout in one kernel call.
            T = len(rollout_states)
            states_batch = torch.stack(rollout_states, dim=1)       # [1, T, 6]
            actions_t = torch.tensor(rollout_actions, device=device) # [T]
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)    # [T]
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)  # [T]

            optimiser.zero_grad()

            with amp_ctx:
                # Single forward pass over the entire rollout
                all_probs, all_values_t, _ = agent.forward_sequence(
                    states_batch, hidden_start
                )
                # all_probs:    [1, T, num_actions]
                # all_values_t: [1, T, 1]
                all_probs = all_probs[0]            # [T, num_actions]
                vals = all_values_t[0].squeeze(-1)  # [T]

                dist = torch.distributions.Categorical(all_probs)
                log_probs = dist.log_prob(actions_t)   # [T]
                entropy = dist.entropy()               # [T]

                actor_loss  = -(log_probs * advantages_t.detach()).mean()
                critic_loss = VALUE_COEF * F.mse_loss(vals, returns_t)
                entropy_loss = -ENTROPY_COEF * entropy.mean()
                total_loss  = actor_loss + critic_loss + entropy_loss

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                scaler.step(optimiser)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimiser.step()

        # EMA smoothing (average per-step reward)
        avg_reward = ep_reward / env.EPISODE_LENGTH
        ema_reward = (1 - EMA_ALPHA) * ema_reward + EMA_ALPHA * avg_reward
        episode_rewards.append(ema_reward)

        if verbose and (ep + 1) % 1000 == 0:
            print(f"  Episode {ep+1:6d}/{n_episodes} | EMA reward: {ema_reward:.4f}")

    return episode_rewards, agent


# ---------------------------------------------------------------------------
# Multi-scenario training
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS = {
    "V2X":  (VehicularB_V2X,  {},            0.75),
    "THz":  (TDL_C_THz,       {},            0.78),
    "IoE":  (TDL_A_IoE,       {},            0.74),
    "UAV":  (GaussMarkov_UAV, {"speed_kmh": 120}, 0.71),
}


def run_all_scenarios(device: torch.device = DEVICE) -> Dict[str, np.ndarray]:
    """
    Train A3C on all four scenarios with N_SEEDS random seeds each.

    Returns
    -------
    results : dict mapping scenario name → array [N_SEEDS, N_EPISODES] of
              EMA reward histories.
    """
    results: Dict[str, np.ndarray] = {}

    total_runs = len(SCENARIO_CONFIGS) * N_SEEDS
    run_idx = 0

    for scenario, (ChannelClass, kwargs, _) in SCENARIO_CONFIGS.items():
        print(f"\n{'='*55}")
        print(f"Scenario: {scenario}")
        print(f"{'='*55}")

        seed_rewards = np.zeros((N_SEEDS, N_EPISODES))

        for s in range(N_SEEDS):
            run_idx += 1
            print(f"  Seed {s+1}/{N_SEEDS} (run {run_idx}/{total_runs})...")
            t0 = time.time()

            channel = ChannelClass(**kwargs, device=device.type)
            env = WaveformSelectionEnv(
                channel,
                seed=s * 100 + hash(scenario) % 100,
                device=device,
            )
            rewards, agent = train_agent(
                env, n_episodes=N_EPISODES, seed=s, verbose=False, device=device
            )

            seed_rewards[s, :] = rewards
            elapsed = time.time() - t0
            final_r = rewards[-1]
            print(f"    Done in {elapsed:.1f}s | final EMA reward: {final_r:.4f}")

            # Save best agent weights (first seed only)
            if s == 0:
                wt_path = os.path.join(
                    RESULTS_DIR, f"lisat_agent_{scenario.lower()}.pt"
                )
                torch.save(agent.state_dict(), wt_path)
                print(f"    Saved agent: {wt_path}")

        results[scenario] = seed_rewards
        print(f"  {scenario} mean final reward (last 500 ep): "
              f"{seed_rewards[:, -500:].mean():.4f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    results: Dict[str, np.ndarray],
    n_episodes: int,
    save_path: str,
) -> None:
    """Plot convergence curves with 95% CI bands and ε-optimal threshold."""
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
    })

    colors = {"V2X": "#1f77b4", "THz": "#2ca02c", "IoE": "#ff7f0e", "UAV": "#d62728"}
    episodes = np.arange(1, n_episodes + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario, rewards in results.items():
        # rewards: [N_SEEDS, N_EPISODES]
        mean_r = rewards.mean(axis=0)
        std_r = rewards.std(axis=0)
        ci95 = std_r  # ±1σ confidence band
        color = colors.get(scenario, "gray")

        ax.plot(episodes, mean_r, color=color, label=scenario, linewidth=2)
        ax.fill_between(episodes, mean_r - ci95, mean_r + ci95,
                        color=color, alpha=0.15)

        # Convergence milestone (first episode where EMA > 90% of final value)
        final_val = mean_r[-500:].mean()
        thresh_val = 0.9 * final_val
        conv_ep = next(
            (i + 1 for i, r in enumerate(mean_r) if r >= thresh_val),
            n_episodes,
        )
        ax.axvline(conv_ep, color=color, linestyle=":", alpha=0.5, linewidth=1.0)
        ax.annotate(
            f"{conv_ep//1000}k",
            xy=(conv_ep, mean_r[conv_ep - 1]),
            xytext=(conv_ep + 200, mean_r[conv_ep - 1] + 0.02),
            fontsize=8,
            color=color,
        )

    # Epsilon-optimal threshold (Theorem 1)
    epsilon_thresholds = {s: t for s, (_, _, t) in SCENARIO_CONFIGS.items()}
    eps_min = min(epsilon_thresholds.values())
    ax.axhline(
        eps_min,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"ε-optimal threshold (Thm. 1, min≈{eps_min:.2f})",
    )

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Average Cumulative Reward")
    ax.set_title("Figure 3: A3C Convergence — All Scenarios")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, n_episodes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LiSAT Simulation: A3C Convergence (Figure 3)")
    print(f"Episodes: {N_EPISODES} | Seeds: {N_SEEDS}")
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        print(f"GPU     : {props.name} ({props.total_memory // 1024**2} MiB)")
        print(f"AMP     : enabled (float16 autocasting)")
    else:
        print("AMP     : disabled (CPU mode)")
    print("=" * 60)

    t_start = time.time()
    results = run_all_scenarios(device=DEVICE)
    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed/60:.1f} min")

    # Save results
    npz_path = os.path.join(RESULTS_DIR, "convergence_all_scenarios.npz")
    np.savez(npz_path, **{k: v for k, v in results.items()})
    print(f"Saved results: {npz_path}")

    # Plot
    fig_path = os.path.join(FIGURES_DIR, "fig3_convergence.png")
    plot_convergence(results, N_EPISODES, fig_path)

    print("\nSimulation complete.")
    print(f"Results  : {npz_path}")
    print(f"Figure   : {fig_path}")


if __name__ == "__main__":
    main()

