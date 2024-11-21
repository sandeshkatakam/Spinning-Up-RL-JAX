import jax
import jax.numpy as jnp

#TODO: Add Support for All Gymnasium Environments

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union, NamedTuple
import numpy as np

class EnvStep(NamedTuple):
    """Standardized environment step output."""
    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]

class EnvReset(NamedTuple):
    """Standardized environment reset output."""
    observation: np.ndarray
    info: Dict[str, Any]

class BaseEnvLoader(ABC):
    """Abstract base class for environment loaders."""
    
    @abstractmethod
    def reset(self) -> EnvReset:
        """Reset the environment."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> EnvStep:
        """Take a step in the environment."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Get observation space."""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Get action space."""
        pass
    
    @abstractmethod
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        pass
    
    @abstractmethod
    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation."""
        pass
    
    @abstractmethod
    def preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Preprocess action."""
        pass



############################### Environment Loader ###############################d
from typing import Optional, Tuple, Dict, Any, Union
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    RecordVideo,
    RecordEpisodeStatistics,
    ClipAction,
    NormalizeObservation,
    NormalizeReward,
)

class EnvLoader:
    """Handles environment loading and preprocessing for RL algorithms."""

    def __init__(
        self,
        env_id: str,
        seed: int = 0,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        clip_actions: bool = True,
        capture_video: bool = False,
        video_dir: Optional[str] = None,
        **env_kwargs
    ):
        """
        Initialize environment loader.

        Args:
            env_id: Gymnasium environment ID
            seed: Random seed
            normalize_obs: Whether to normalize observations
            normalize_reward: Whether to normalize rewards
            clip_actions: Whether to clip actions to valid range
            capture_video: Whether to record videos
            video_dir: Directory to save videos
            **env_kwargs: Additional arguments passed to env constructor
        """
        self.env_id = env_id
        self.seed = seed
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.clip_actions = clip_actions
        self.capture_video = capture_video
        self.video_dir = video_dir
        self.env_kwargs = env_kwargs
        
        self.env = self._make_env()
        self._check_env_compatibility()
        
    def _make_env(self) -> gym.Env:
        """Create and wrap the environment."""
        env = gym.make(self.env_id, **self.env_kwargs)
        
        # Add episode statistics recording
        env = RecordEpisodeStatistics(env)
        
        # Optional video recording
        if self.capture_video and self.video_dir:
            env = RecordVideo(
                env,
                video_folder=self.video_dir,
                episode_trigger=lambda x: x % 100 == 0  # Record every 100th episode
            )
        
        # Clip actions to valid range
        if self.clip_actions and isinstance(env.action_space, gym.spaces.Box):
            env = ClipAction(env)
            
        # Normalize observations
        if self.normalize_obs:
            env = NormalizeObservation(env)
            
        # Normalize rewards
        if self.normalize_reward:
            env = NormalizeReward(env)
        
        # Set seeds
        env.reset(seed=self.seed)
        
        return env
    
    def _check_env_compatibility(self):
        """Verify environment compatibility with standard RL algorithms."""
        # Check observation space
        assert isinstance(
            self.env.observation_space,
            (gym.spaces.Box, gym.spaces.Discrete)
        ), "Unsupported observation space"
        
        # Check action space
        assert isinstance(
            self.env.action_space,
            (gym.spaces.Box, gym.spaces.Discrete)
        ), "Unsupported action space"
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        """Get environment observation space."""
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Space:
        """Get environment action space."""
        return self.env.action_space
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'env_id': self.env_id,
            'obs_space': self.observation_space,
            'act_space': self.action_space,
            'obs_shape': self.observation_space.shape,
            'act_shape': self.action_space.shape if isinstance(self.action_space, gym.spaces.Box) else (1,),
            'is_discrete_action': isinstance(self.action_space, gym.spaces.Discrete),
            'action_dim': self.action_space.n if isinstance(self.action_space, gym.spaces.Discrete) 
                         else self.action_space.shape[0]
        }
    
    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation if needed."""
        if isinstance(self.observation_space, gym.spaces.Discrete):
            # One-hot encode discrete observations
            one_hot = np.zeros(self.observation_space.n)
            one_hot[obs] = 1
            return one_hot
        return obs
    
    def preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Preprocess action if needed."""
        if isinstance(self.action_space, gym.spaces.Box):
            # Clip continuous actions to valid range
            return np.clip(
                action,
                self.action_space.low,
                self.action_space.high
            )
        return action
