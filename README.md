# LiSAT Simulation Suite

Python/PyTorch simulation scripts for the paper:

> **"LiSAT: Lightweight Semantic-Adaptive Transceiver for 6G Waveform Selection"**
> IEEE Wireless Communications Letters

---

## Repository Structure

```
simulations/
├── channel_models.py           # Channel model implementations
├── waveforms.py                # Waveform modulator/demodulator implementations
├── lisat_model.py              # LiSAT CSL encoder/decoder and A3C agent
├── simulate_ber_ser.py         # BER/SER vs. SNR simulation (Figure 2)
├── simulate_convergence.py     # A3C convergence curves (Figure 3)
├── simulate_waveform_selection.py  # Waveform selection frequency (Figure 4)
├── results/                    # Saved .npz numerical results (auto-created)
└── figures/                    # Saved .png plots (auto-created)
```

---

## Dependencies

| Package    | Version  | Notes                                      |
|------------|----------|--------------------------------------------|
| Python     | ≥ 3.9    |                                            |
| PyTorch    | ≥ 2.0    | CPU sufficient; CUDA optional              |
| NumPy      | ≥ 1.23   |                                            |
| Matplotlib | ≥ 3.6    | For figure generation                      |
| SciPy      | ≥ 1.9    | Optional (used for advanced DSP utilities) |

### Installation

```bash
pip install torch>=2.0 numpy>=1.23 matplotlib>=3.6 scipy>=1.9
```

---

## Scripts

### 1. `channel_models.py` — Channel Models

Implements five doubly-dispersive channel models:

| Class              | Scenario    | Carrier   | ν_max      |
|--------------------|-------------|-----------|------------|
| `VehicularB_V2X`   | V2X DSRC    | 5.9 GHz   | ~2747 Hz   |
| `TDL_A_IoE`        | IoE mmWave  | 28 GHz    | 150 Hz     |
| `TDL_C_THz`        | THz static  | 300 GHz   | 80 Hz      |
| `GaussMarkov_UAV`  | UAV A2G     | 3.5 GHz   | variable   |
| `DoublyDispersiveChannel` | General | configurable | configurable |

**Smoke test:**
```bash
python channel_models.py
```

**Key API:**
```python
from channel_models import VehicularB_V2X

ch = VehicularB_V2X()
h = ch.generate_channel(batch_size=4, num_frames=14)  # [4, 6, 14]
rx = ch.apply(tx_signal)
rx_noisy = ch.add_noise(rx, snr_db=10.0)
```

---

### 2. `waveforms.py` — Waveform Implementations

Implements four waveform modulator/demodulator pairs:

| Waveform | Class                       | Transform              |
|----------|-----------------------------|------------------------|
| OFDM     | `OFDMModulator/Demodulator` | IFFT + CP              |
| OTFS     | `OTFSModulator/Demodulator` | ISFFT + Heisenberg     |
| OCDM     | `OCDMModulator/Demodulator` | DFnT (pre-chirp+IFFT)  |
| AFDM     | `AFDMModulator/Demodulator` | IDAFT (affine Fourier) |

Also provides:
- `qam_mapper(bits, order=16)` / `qam_demapper(symbols, order=16)` — Gray-coded QAM
- `compute_papr(signal)` — Peak-to-Average Power Ratio

**Smoke test:**
```bash
python waveforms.py
```

**Key API:**
```python
from waveforms import AFDMModulator, AFDMDemodulator, qam_mapper

mod = AFDMModulator(N=64, c1=0.25, c2=0.0, N_cp=16)
dem = AFDMDemodulator(N=64, c1=0.25, c2=0.0, N_cp=16)

signal = mod.modulate(symbols)   # [batch, N+N_cp]
rx_sym = dem.demodulate(signal)  # [batch, N]
```

**AFDM IDAFT (Eq. 1):**
```
x[n] = (1/√N) Σ_k X[k] · exp(j2π(c₁n²/N + kn/N + c₂k²/N))
```

---

### 3. `lisat_model.py` — LiSAT Components

#### `LightweightSemanticCodec` (CSL)
4-layer convolutional autoencoder for semantic source coding.
- INT8-quantisation-aware (QAT) mode via `torch.ao.quantization`
- Methods: `encode(x)`, `decode(z)`, `compute_loss(x_hat, x)`

#### `A3CAgent`
GRU-based actor-critic for waveform selection.
- State: `[τ̂_max (µs), ν̂_max (kHz), SNR (norm), η_s, P_t, wf_prev]`
- Actions: `{0=OTFS, 1=OCDM, 2=AFDM}`
- Reward: `clip(α·Δη_s + β·(-ΔP/P_max) − γ·τ/τ_max − λ·v, −1, 1)` — Eq. (11)

#### `MERDEstimator`
Estimates channel delay-Doppler profile from AFDM pilot subframe.

#### `LiSATSystem`
Full system combining CSL + MERD + A3C.

**Smoke test:**
```bash
python lisat_model.py
```

---

### 4. `simulate_ber_ser.py` — Figure 2: BER/SER vs. SNR

Reproduces Figure 2 comparing six methods in the V2X Vehicular-B scenario.

| Method         | Description                        |
|----------------|------------------------------------|
| OFDM           | Baseline, no adaptation            |
| OTFS           | Fixed OTFS                         |
| OCDM           | Fixed OCDM                         |
| AFDM           | Fixed AFDM                         |
| CSI-Threshold  | Rule-based: OTFS if τ_max > thresh |
| LiSAT          | A3C-adaptive waveform selection    |

**Run:**
```bash
python simulate_ber_ser.py
```

**Output:**
- `results/ber_ser_v2x.npz` — numerical BER/SER arrays
- `figures/fig2_ber_ser.png` — BER and SER curves (IEEE-style)

**Key hyperparameters** (edit at top of file):
```python
N_SUBCARRIERS = 64
QAM_ORDER = 16
N_MC = 500         # increase to 1000 for publication quality
SNR_RANGE_DB = np.arange(-5.0, 30.5, 2.5)
```

**Estimated runtime:** ~5–15 min on CPU (N_MC=500)

---

### 5. `simulate_convergence.py` — Figure 3: A3C Convergence

Trains the A3C agent on all four scenarios and plots cumulative reward vs. episodes.

**Run:**
```bash
python simulate_convergence.py
```

**Output:**
- `results/convergence_all_scenarios.npz` — reward histories [seeds × episodes]
- `results/lisat_agent_{scenario}.pt` — saved agent weights per scenario
- `figures/fig3_convergence.png` — convergence curves with 95% CI bands

**Key hyperparameters:**
```python
N_EPISODES = 20_000   # training episodes per scenario
N_SEEDS = 5           # random seeds for confidence intervals
EMA_ALPHA = 0.05      # exponential moving average smoothing
LR = 3e-4             # Adam learning rate
GAMMA = 0.99          # RL discount factor
```

**Estimated runtime:** ~10–40 min on CPU (all 4 scenarios × 5 seeds)

---

### 6. `simulate_waveform_selection.py` — Figure 4: Waveform Selection Frequency

Uses trained A3C agents to estimate waveform selection distributions across
scenarios and SNR regimes.

**Run (requires convergence simulation to have been run first):**
```bash
python simulate_waveform_selection.py
```

Or run standalone (uses initialised/random policy if no saved weights):
```bash
python simulate_waveform_selection.py
```

**Output:**
- `results/waveform_selection.npz` — selection frequency arrays
- `figures/fig4_waveform_selection.png` — grouped stacked bar chart

**SNR regimes:**
| Regime | Range        |
|--------|--------------|
| Low    | −5 to 5 dB   |
| Medium | 5 to 15 dB   |
| High   | 15 to 30 dB  |

**Estimated runtime:** ~1–3 min on CPU (N_MC=2000)

---

## Recommended Execution Order

```bash
# 1. Verify all components work
python channel_models.py
python waveforms.py
python lisat_model.py

# 2. Train agents (saves weights to results/)
python simulate_convergence.py

# 3. Run BER/SER simulation
python simulate_ber_ser.py

# 4. Run waveform selection (uses trained agents from step 2)
python simulate_waveform_selection.py
```

---

## Reproducibility

All scripts use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
```

Results may vary slightly across platforms due to floating-point non-determinism
in PyTorch operations. Set `torch.use_deterministic_algorithms(True)` for full
determinism (may be slower).

---

## Key Mathematical Definitions

### Reward Function (Eq. 11)
```
r = clip( α·Δη_s + β·(−ΔP/P_max) − γ·(τ/τ_max) − λ·v, −1, 1 )
```
Default: α=0.5, β=0.3, γ=0.1, λ=0.5

### Semantic Efficiency
```
η_s = 1 − MSE(x̂, x) / E[||x||²]
```

### AFDM IDAFT
```
x[n] = (1/√N) Σ_k X[k] · exp(j2π(c₁n²/N + kn/N + c₂k²/N))
```

### OCDM DFnT (pre-chirp + IFFT)
```
x_OCDM[n] = IFFT{ X[k] · exp(jπk²/N) }
```

### OTFS ISFFT
```
S[n,m] = (1/√NM) Σ_{k,l} X[k,l] · exp(j2π(kn/N − lm/M))
```

### Gauss-Markov channel evolution
```
h_{l,t+1} = ρ_l · h_{l,t} + √(1−ρ_l²) · w_{l,t},   ρ_l = exp(−2π f_{D,l} T_s)
```

---

## Citation

If you use these simulations in your research, please cite:

```
@article{lisat2024,
  title   = {LiSAT: Lightweight Semantic-Adaptive Transceiver for 6G
             Waveform Selection},
  journal = {IEEE Wireless Communications Letters},
  year    = {2024},
}
```
