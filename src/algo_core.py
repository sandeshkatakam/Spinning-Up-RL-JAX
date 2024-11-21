from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax.random import PRNGKey

class BaseModel(ABC):
    """
    Abstract base class for user-defined models in the RL library.
    """

    @abstractmethod
    def forward(self, params, inputs, rng: PRNGKey):
        """
        Forward pass of the model.

        Parameters:
            params: PyTree. Model parameters.
            inputs: Array. Input to the model (e.g., states or observations).
            rng: PRNGKey. Random key for stochastic operations.

        Returns:
            outputs: Array. Model output (e.g., action logits or value estimates).
        """
        pass

    @abstractmethod
    def init_params(self, rng: PRNGKey, input_shape: tuple):
        """
        Initialize model parameters.

        Parameters:
            rng: PRNGKey. Random key for initialization.
            input_shape: Tuple. Shape of the input data.

        Returns:
            params: PyTree. Initialized model parameters.
        """
        pass


# ... existing BaseModel class code ...

class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    """
    
    def __init__(self, model: BaseModel, input_shape: tuple, optimizer, rng: PRNGKey):
        """
        Initialize the RL algorithm.

        Parameters:
            model: Instance of a BaseModel subclass.
            input_shape: Shape of the input to the model.
            optimizer: Optimizer instance for parameter updates.
            rng: PRNGKey for random initialization.
        """
        self.model = model
        self.rng = rng
        self.params = model.init_params(rng, input_shape)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.params)

    @abstractmethod
    def select_action(self, state):
        """
        Select an action given the current state.

        Parameters:
            state: Current environment state/observation.

        Returns:
            action: Selected action.
        """
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """
        Perform a single training step.

        Parameters and returns should be defined by the specific algorithm.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the algorithm's parameters.

        Parameters and returns should be defined by the specific algorithm.
        """
        pass

# ... existing model examples ...

# Update PolicyGradient to inherit from BaseAlgorithm
class PolicyGradient(BaseAlgorithm):
    def __init__(self, model: BaseModel, input_shape, optimizer, rng):
        super().__init__(model, input_shape, optimizer, rng)

    def select_action(self, state):
        logits, self.rng = self.model.forward(self.params, state, self.rng)
        action = jax.random.categorical(self.rng, logits)
        return action

    def train_step(self, states, actions, rewards, loss_fn):
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(self.params, states, actions, rewards)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss
        
    def update(self, states, actions, rewards):
        """
        Update policy parameters using the collected experience.
        
        Parameters:
            states: Array of states from collected experience
            actions: Array of actions taken
            rewards: Array of rewards received
            
        Returns:
            loss: The training loss for this update
        """
        def loss_fn(params, states, actions, rewards):
            logits, _ = self.model.forward(params, states, self.rng)
            # Example loss computation - should be implemented based on specific algorithm
            return -jnp.mean(rewards)  # Placeholder loss
            
        return self.train_step(states, actions, rewards, loss_fn)
############################# User Defined Models Examples in Different neural network libraries ##########################
# In FLAX:
from flax import linen as nn

class FlaxModel(BaseModel):
    def __init__(self):
        self.model = nn.Dense(10)

    def forward(self, params, inputs, rng):
        return self.model.apply(params, inputs), rng

    def init_params(self, rng, input_shape):
        dummy_input = jnp.ones(input_shape)
        return self.model.init(rng, dummy_input)


# In Equinox:

import equinox as eqx

class EquinoxModel(BaseModel):
    def __init__(self):
        self.model = eqx.nn.Linear(4, 10)

    def forward(self, params, inputs, rng):
        return self.model(inputs), rng

    def init_params(self, rng, input_shape):
        # Equinox models already contain their parameters
        return self.model


# In Native JAX:
class JaxModel(BaseModel):
    def forward(self, params, inputs, rng):
        W, b = params
        return jnp.dot(inputs, W) + b, rng

    def init_params(self, rng, input_shape):
        input_dim = input_shape[-1]
        output_dim = 10  # Example
        W = jax.random.normal(rng, (input_dim, output_dim))
        b = jax.random.normal(rng, (output_dim,))
        return (W, b)

## Integrating the Model Class into the RL algorithm class:

class PolicyGradient:
    def __init__(self, model: BaseModel, input_shape, optimizer, rng):
        """
        Initialize the RL agent.

        Parameters:
            model: Instance of a BaseModel subclass.
            input_shape: Shape of the input to the model.
            optimizer: Optax optimizer.
            rng: PRNGKey for random initialization.
        """
        self.model = model
        self.rng = rng
        self.params = model.init_params(rng, input_shape)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.params)

    def select_action(self, state):
        logits, self.rng = self.model.forward(self.params, state, self.rng)
        action = jax.random.categorical(self.rng, logits)
        return action

    def train_step(self, states, actions, rewards, loss_fn):
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(self.params, states, actions, rewards)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

