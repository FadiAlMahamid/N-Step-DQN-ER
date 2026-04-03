# N-Step DQN with Experience Replay: Atari Breakout

An **N-Step DQN** implementation with **Experience Replay** applied to `BreakoutNoFrameskip-v4` from OpenAI Gymnasium. This project replaces the standard single-step TD target with **multi-step returns** that accumulate rewards over N steps before bootstrapping from the Q-network. Combined with **epsilon-greedy exploration**, a **Replay Buffer** to break temporal correlations, and a **Target Network** to stabilize Q-value targets.

## Project Overview

The agent learns to play Atari Breakout using a convolutional neural network to approximate Q-values. The implementation incorporates four key components:

1. **N-Step Returns** — Instead of bootstrapping from the Q-network after a single step, the agent accumulates discounted rewards over N steps before adding the bootstrapped Q-value. This propagates rewards back to earlier states faster and reduces the bias of the TD estimate at the cost of slightly higher variance.
2. **Epsilon-Greedy Exploration** — The agent explores by taking random actions with probability epsilon, which decays linearly from 1.0 to 0.01 over training. This ensures broad exploration early on and shifts to exploitation as the agent learns.
3. **Experience Replay Buffer** — A memory-efficient buffer that stores individual frames (uint8) and reconstructs stacked states on demand, breaking the correlation between consecutive samples while minimizing memory usage.
4. **Target Network** — A periodically-updated frozen copy of the policy network provides stable TD targets, solving the moving target problem.

### Why N-Step Returns Over Single-Step TD?

Single-step TD targets bootstrap heavily — the target is mostly determined by the Q-network's own (potentially inaccurate) estimate of the next state. N-step returns reduce this dependence by using actual observed rewards for N steps before bootstrapping:

```
Single-step: target = rₜ + γ · Q(sₜ₊₁)
N-step:      target = rₜ + γ · rₜ₊₁ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁ + γⁿ · Q(sₜ₊ₙ)
```

This means rewards propagate back to earlier states in fewer training updates. For example, with N=3, a reward signal reaches 3 states back in a single update instead of requiring 3 separate updates.

### Core Features

- **Modular Architecture:** Separated into distinct files for the environment, agent, network, replay buffer, utilities, and training logic.
- **Configurable N:** The number of steps is controlled via `config.yaml` — experiment with different values without editing code.
- **Episode-Boundary Handling:** When an episode ends mid-accumulation, remaining partial N-step transitions are flushed with shorter returns so no experience is wasted.
- **Step-Based Training Loop:** Training runs for a fixed number of agent steps (not episodes), matching the original DeepMind approach. All logging and checkpointing are step-based.
- **YAML Configuration:** All hyperparameters, paths, and execution modes are controlled via `config.yaml` — no need to edit code between experiments.
- **GPU/MPS Support:** Automatic device detection for CUDA, Apple Silicon (MPS), or CPU fallback.
- **Persistent Storage:**
  - `.pth` files store the trained model weights.
  - `.npz` files store the full training history (rewards, losses, step count).
- **Resume Training:** Interrupt training at any time (Ctrl+C) and resume from the last checkpoint without losing progress.
- **Visualization:** Automatic generation of reward and loss training curves at each checkpoint, plus a deployment mode with action distribution logging.

---

## Project Structure

```
N-Step-DQN-ER/
├── config.yaml            # All hyperparameters and settings
├── q_network.py           # Standard CNN Q-Network (DeepMind 2015 Nature architecture)
├── replay_buffer.py       # Memory-efficient Experience Replay Buffer
├── dqn_agent.py           # N-Step DQN agent (n-step returns + epsilon-greedy)
├── environment.py         # Gym environment setup with Atari preprocessing
├── utils.py               # Config loading, plotting, and deployment visualization
├── training_script.py     # Step-based training loop and main entry point
└── nstep_dqn_results/     # Generated outputs (model, history, plots)
```

---

## Key Components

| Component | Purpose |
|---|---|
| **N-Step Returns** | Accumulates discounted rewards over N steps before bootstrapping, reducing bias and propagating rewards faster |
| **N-Step Buffer** | A local deque of size N in the agent that accumulates transitions before computing the multi-step return and storing in the replay buffer |
| **Episode Flushing** | When an episode ends, remaining partial transitions are stored with shorter-than-N returns so no experience is wasted |
| **Epsilon-Greedy** | Linear decay from 1.0 to 0.01 over 1M steps for exploration; standard approach from DeepMind 2015 |
| **Experience Replay Buffer** | Stores individual frames (uint8) and reconstructs stacked states on demand, breaking temporal correlation while minimizing memory usage |
| **Target Network** | A frozen copy of the policy network, updated every 10,000 steps, provides stable TD targets |
| **Double DQN Target Calculation** | Policy network selects the best action; target network evaluates it — reduces overestimation bias |

### N-Step Return Mechanism

```
Single-step DQN:
    target = rₜ + γ · Q_target(sₜ₊₁, a')

N-Step DQN (N=3):
    Rₙ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂
    target = Rₙ + γ³ · Q_target(sₜ₊₃, a')

where:
  Rₙ          — the N-step discounted return (actual observed rewards)
  γⁿ          — discount factor raised to N (accounts for the N steps already covered)
  Q_target    — bootstrapped value from the target network at step t+n
```

As N increases, the target relies more on actual rewards and less on the Q-network's estimate. The trade-off: higher N means more variance (rewards are stochastic) but lower bias (less bootstrapping).

### Double DQN Target Calculation

```
Standard DQN:   target = Rₙ + γⁿ · maxₐ' Q_target(sₜ₊ₙ, a')
Double DQN:     target = Rₙ + γⁿ · Q_target(sₜ₊ₙ, argmaxₐ' Q_policy(sₜ₊ₙ, a'))
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Gymnasium with Atari support (`ale-py`)
- NumPy, Matplotlib, PyYAML

### Installation

```bash
pip install torch gymnasium ale-py numpy matplotlib pyyaml
```

### Running the Script

Control the execution mode via `config.yaml`:

```yaml
training:
  mode: "new"      # Train from scratch
  # mode: "resume" # Load saved model and continue training
  # mode: "deploy" # Watch the trained agent play
```

Then run:

```bash
python training_script.py
```

---

## Configuration

All parameters are managed in [`config.yaml`](config.yaml). The config is organized into logical sections:

### Training Loop

| Parameter | Default | Description |
|---|---|---|
| `num_steps` | 2,500,000 | Total agent steps to train for (each step = 4 frames) |
| `target_reward` | 40 | Average reward for early stopping (per-life, not per-game) |
| `print_every` | 10,000 | Steps between console progress logs |
| `checkpoint_every` | 100,000 | Steps between model/plot checkpoints (0 to disable) |
| `plot_window` | 500 | Moving average window for reward curves and early stopping |

### Agent Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | adam | Optimizer type (`adam` or `rmsprop`) |
| `learning_rate` | 0.00025 | Optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `loss_function` | huber | Loss type (`huber` or `mse`) |
| `grad_clip_norm` | 1.0 | Maximum gradient norm for clipping |
| `clip_rewards` | true | Clip rewards to [-1, 1] for stable gradients |

### N-Step Returns

| Parameter | Default | Description |
|---|---|---|
| `n` | 3 | Number of steps to accumulate before bootstrapping (1 = standard DQN) |

### Epsilon-Greedy Exploration

| Parameter | Default | Description |
|---|---|---|
| `start` | 1.0 | Initial exploration rate (100% random) |
| `min` | 0.01 | Minimum exploration rate after decay |
| `decay_steps` | 1,000,000 | Steps over which epsilon decays linearly |

### Experience Replay Buffer

| Parameter | Default | Description |
|---|---|---|
| `capacity` | 1,000,000 | Maximum transitions stored (FIFO eviction) |
| `batch_size` | 32 | Mini-batch size for training |
| `learning_starts` | 50,000 | Warmup transitions before learning begins |

### Target Network

| Parameter | Default | Description |
|---|---|---|
| `update_freq` | 10,000 | Steps between target network weight copies |

### Other

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Random seed for reproducibility |

---

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* Chapter 7: n-step Bootstrapping.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
- van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
- Hessel, M. et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI.
