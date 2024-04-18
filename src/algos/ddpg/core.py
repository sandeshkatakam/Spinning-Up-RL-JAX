import jax.numpy as jnp
import scipy.signal
import equinox as eqx
import jax

def combined_shape(length, shape = None):
    if shape is None:
        return (length,)
    return (length, shape) if jnp.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes) -1):
        act = activation if j < len(sizes) -2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(eqx.Module):
    
    def __init__(self,obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


class MLPQFunction(eqx.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim = -1))
        return jnp.squeeze(q, -1)


        
