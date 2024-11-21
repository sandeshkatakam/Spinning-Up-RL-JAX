from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..env.base import BaseEnvLoader
from ..algorithms.base import BaseAlgorithm
from ..training.base import BaseLogger
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import yaml
import wandb

@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    project: str
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    mode: str = "online"  # online, offline, or disabled

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Experiment settings
    experiment_name: str
    output_dir: str
    
    # Training settings
    n_trials: int = 5
    n_timesteps: int = 1_000_000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    n_jobs: int = -1
    seed: int = 42
    
    # Logging settings
    wandb: Optional[WandBConfig] = None
    save_freq: int = 100000
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BenchmarkConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Handle WandB config if present
        if 'wandb' in config_dict:
            config_dict['wandb'] = WandBConfig(**config_dict['wandb'])
            
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(
        self,
        algo_name: str,
        env_id: str,
        metrics: Dict[str, List[float]],
        config: Dict[str, Any]
    ):
        self.algo_name = algo_name
        self.env_id = env_id
        self.metrics = metrics
        self.config = config
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    def save(self, path: str):
        """Save benchmark results to disk."""
        save_dict = {
            'algo_name': self.algo_name,
            'env_id': self.env_id,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': self.timestamp
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkResult':
        """Load benchmark results from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            data['algo_name'],
            data['env_id'],
            data['metrics'],
            data['config']
        )
# ... (previous imports) ...
import wandb
from .config import BenchmarkConfig, WandBConfig

class BenchmarkRunner:
    """Runs benchmarks for RL algorithms on specified environments."""
    
    def __init__(
        self,
        config: BenchmarkConfig
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(self.output_dir / 'config.yaml')
        
        # Initialize W&B if configured
        self.wandb_run = None
        if self.config.wandb and self.config.wandb.mode != "disabled":
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                group=self.config.wandb.group,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=self.config.to_dict(),
                mode=self.config.wandb.mode
            )
        
        # Store available algorithms and environments
        self.registered_algorithms = {}
        self.registered_environments = {}

    def _log_to_wandb(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: Optional[str] = None
    ):
        """Log metrics to Weights & Biases."""
        if self.wandb_run is not None:
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb.log(metrics, step=step)

    def _run_single_trial(
        self,
        algo_name: str,
        env_id: str,
        trial_seed: int,
        algo_config: Dict[str, Any],
        env_config: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Run a single benchmark trial."""
        # ... (previous implementation) ...
        
        # Add W&B logging
        if timesteps % self.config.eval_freq == 0:
            eval_metrics = {
                'eval/mean_return': np.mean(eval_returns),
                'eval/std_return': np.std(eval_returns),
                'eval/mean_length': np.mean(eval_lengths),
                'eval/std_length': np.std(eval_lengths)
            }
            self._log_to_wandb(
                eval_metrics,
                timesteps,
                prefix=f"{algo_name}/{env_id}/trial_{trial_seed}"
            )
        
        return metrics

    def run_benchmark(
        self,
        algo_names: Union[str, List[str]],
        env_ids: Union[str, List[str]],
        algo_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        env_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for specified algorithms and environments."""
        # ... (previous implementation) ...
        
        # Log final results to W&B
        if self.wandb_run is not None:
            for name, result in results.items():
                # Log learning curves
                for i, (step, mean, std) in enumerate(zip(
                    result.metrics['timesteps'],
                    result.metrics['mean_return'],
                    result.metrics['std_return']
                )):
                    self._log_to_wandb(
                        {
                            'final/mean_return': mean,
                            'final/std_return': std
                        },
                        step,
                        prefix=name
                    )
                
                # Log final metrics
                final_metrics = {
                    'final/mean_return': result.metrics['mean_return'][-1],
                    'final/std_return': result.metrics['std_return'][-1],
                }
                self._log_to_wandb(final_metrics, step=0, prefix=name)
                
                # Save learning curve plot to W&B
                fig = self.plot_results({name: result}, show=False)
                wandb.log({f"{name}/learning_curve": wandb.Image(fig)})
                plt.close(fig)
        
        return results

    def plot_results(
        self,
        results: Dict[str, BenchmarkResult],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot benchmark results."""
        fig = plt.figure(figsize=(10, 6))
        # ... (previous implementation) ...
        if show:
            plt.show()
        return fig

    def __del__(self):
        """Cleanup when the runner is destroyed."""
        if self.wandb_run is not None:
            self.wandb_run.finish()


################################################# Example usage ######################################################
from src.benchmarks.runner import BenchmarkRunner
from src.benchmarks.config import BenchmarkConfig
from src.algos import PPO, SAC
from src.env import GymEnvLoader

def run_example_benchmark():
    # Load configuration from YAML
    config = BenchmarkConfig.from_yaml('configs/benchmark_config.yaml')
    
    # Create benchmark runner
    runner = BenchmarkRunner(config)
    
    # Register algorithms and environments
    # ... (same as before) ...
    
    # Run benchmarks
    results = runner.run_benchmark(
        algo_names=["PPO", "SAC"],
        env_ids=["HalfCheetah-v4", "Hopper-v4"],
        algo_configs={
            "PPO": {"n_steps": 1024},
        }
    )
    
    # Results are automatically logged to W&B and saved locally