# Getting started


A simple and modular reinforcement learning library implemented in JAX, inspired by OpenAI's Spinning Up.

## Installation

```bash
pip install spinningup-jax
```

## Quick Start

Here's a minimal example to train a VPG (Vanilla Policy Gradient) agent on the CartPole environment:

```python
from spinningup_jax import vpg
from gymnasium import make

# Create environment
env = make("CartPole-v1")

# Train the agent
vpg(
    env_fn=lambda: env,
    epochs=50,
    steps_per_epoch=4000,
)
```

## Core Features

- 🚀 **JAX-based Implementation**: Leverages JAX's automatic differentiation and JIT compilation for fast training
- 🎮 **Gymnasium Compatible**: Works with all standard Gymnasium environments
- 📊 **Built-in Logging**: Automatic experiment tracking and visualization
- 🔧 **Modular Design**: Easy to modify and extend algorithms

## Available Algorithms

- VPG (Vanilla Policy Gradient)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- TRPO (Trust Region Policy Optimization)
- CEM (Cross-Entropy Method)

## Basic Usage

1. **Create an Environment**
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
```

2. **Configure Training**
```python
from spinningup_jax import vpg

vpg_kwargs = {
    "hidden_sizes": [64, 64],
    "learning_rate": 3e-4,
    "gamma": 0.99,
}
```

3. **Train the Agent**
```python
vpg(
    env_fn=lambda: env,
    **vpg_kwargs
)
```

## Logging and Visualization

Training progress is automatically logged and can be visualized:

```python
from spinningup_jax.utils.plot import plot_results

plot_results("./data/experiment_name")
```

## Next Steps

- Check out the [Algorithms](algorithms/index.md) section for detailed documentation of each method
- See [Examples](examples/index.md) for more complex usage scenarios
- Visit [Configuration](config/index.md) to learn about customizing training parameters

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](contributing.md) for more information.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{spinningup_jax2024,
  author = {Sandesh Katakam},
  title = {SpinningUp RL JAX},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sandeshkatakam/SpinningUp-RL-JAX}
}
```