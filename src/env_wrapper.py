from typing import Optional, Dict, Any
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    RecordVideo,
    RecordEpisodeStatistics,
    ClipAction,
    NormalizeObservation,
    NormalizeReward,
)

from .base import BaseEnvLoader, EnvStep, EnvReset

class GymEnvLoader(BaseEnvLoader):
    """Environment loader for Gymnasium environments."""

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
        super().__init__()
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
        
        env = RecordEpisodeStatistics(env)
        
        if self.capture_video and self.video_dir:
            env = RecordVideo(
                env,
                video_folder=self.video_dir,
                episode_trigger=lambda x: x % 100 == 0
            )
        
        if self.clip_actions and isinstance(env.action_space, gym.spaces.Box):
            env = ClipAction(env)
            
        if self.normalize_obs:
            env = NormalizeObservation(env)
            
        if self.normalize_reward:
            env = NormalizeReward(env)
        
        env.reset(seed=self.seed)
        return env
    
    def reset(self) -> EnvReset:
        obs, info = self.env.reset()
        return EnvReset(observation=obs, info=info)
    
    def step(self, action: np.ndarray) -> EnvStep:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return EnvStep(
            observation=obs,
            reward=reward,
            done=terminated,
            truncated=truncated,
            info=info
        )
    
    def close(self):
        self.env.close()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    # ... rest of the implementation remains the same as before ..
from typing import Optional, Dict, Any
import numpy as np
import dm_env
from dm_env import specs
from .base import BaseEnvLoader, EnvStep, EnvReset

class DMEnvLoader(BaseEnvLoader):
    """Environment loader for DeepMind environments."""

    def __init__(
        self,
        env_name: str,
        seed: int = 0,
        **env_kwargs
    ):
        super().__init__()
        self.env_name = env_name
        self.seed = seed
        self.env_kwargs = env_kwargs
        
        # Import and create environment
        from dm_control import suite
        self.env = suite.load(env_name, **env_kwargs)
        self.env.reset(seed=seed)
    
    def reset(self) -> EnvReset:
        time_step = self.env.reset()
        return EnvReset(
            observation=self._process_dm_obs(time_step.observation),
            info={}
        )
    
    def step(self, action: np.ndarray) -> EnvStep:
        time_step = self.env.step(action)
        return EnvStep(
            observation=self._process_dm_obs(time_step.observation),
            reward=float(time_step.reward),
            done=time_step.last(),
            truncated=False,  # DM Control doesn't have truncation
            info={}
        )
    
    def _process_dm_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert DM Control dict observation to array."""
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs
    
    @property
    def observation_space(self):
        # Convert DM Control specs to standardized format
        obs_spec = self.env.observation_spec()
        if isinstance(obs_spec, dict):
            total_size = sum(np.prod(spec.shape) for spec in obs_spec.values())
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,)
            )
        return obs_spec
    
    @property
    def action_space(self):
        # Convert DM Control action spec to standardized format
        action_spec = self.env.action_spec()
        return gym.spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape
        )
    
    def close(self):
        self.env.close()
    
    # ... implement remaining abstract methods ...