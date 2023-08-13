import jax.numpy as jnp
import scipy.signal
import gym.spaces import Box, Discrete

import jax
import equinox as eqx
from jax.random.distributions.normal import Normal
from jax.random.distributions.categorical import Categorical


def combined_shape(length, shape = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes - 1)):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.layers
