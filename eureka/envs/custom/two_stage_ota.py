import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from flax import linen as nn
import numpy as np


class FlaxMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.silu(x)
        x = nn.Dense(512)(x)
        x = nn.silu(x)
        x = nn.Dense(256)(x)
        x = nn.silu(x)
        x = nn.Dense(128)(x)
        x = nn.silu(x)
        x = nn.Dense(9)(x)
        return x

@struct.dataclass
class EnvState:
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float
    x9: float
    x10: float
    x11: float
    x12: float
    x13: float
    x14: float
    x15: float
    time: int


@struct.dataclass
class EnvParams:
    x0_bounds: Tuple[float, float] = (100e-15, 10000e-15)
    x1_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x2_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x3_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x4_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x5_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x6_bounds: Tuple[float, float] = (100e-15, 2000e-15)
    x7_bounds: Tuple[float, float] = (1, 20)
    x8_bounds: Tuple[float, float] = (1, 20)
    x9_bounds: Tuple[float, float] = (1, 20)
    x10_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x11_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x12_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x13_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x14_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x15_bounds: Tuple[float, float] = (1e2, 1e5)
    out0_constraints: Tuple[float, int] = (30e6, 0)
    out1_constraints: Tuple[float, int] = (60, 0)
    out2_constraints: Tuple[float, int] = (100e-9, 1)
    out3_constraints: Tuple[float, int] = (80, 0)
    out4_constraints: Tuple[float, int] = (60, 0)
    out5_constraints: Tuple[float, int] = (80, 0)
    out6_constraints: Tuple[float, int] = (1.5, 0)
    out7_constraints: Tuple[float, int] = (30e-3, 1)
    out0_denormalize: Tuple[float, float] = (0.0, 645470000.0) # min, max, were mapped to [-1,1]
    out1_denormalize: Tuple[float, float] = (-180.0, 179.99)
    out2_denormalize: Tuple[float, float] = (1.2585e-11, 1e-06)
    out3_denormalize: Tuple[float, float] = (-68.19, 210.7)
    out4_denormalize: Tuple[float, float] = (-181.83, 99.497)
    out5_denormalize: Tuple[float, float] = (-4.0129, 150.14)
    out6_denormalize: Tuple[float, float] = (-1.3605, 1.8004)
    out7_denormalize: Tuple[float, float] = (8.661e-06, 2.882)
    out8_denormalize: Tuple[float, float] = (9.6713e-05, 0.0047157)
    num_states: int = 16
    num_spects: int = 9
    max_steps_in_episode: int = 20

WEIGHT_ROOT = '/home/kmcho/Analog_GPU_RL_SJPark_coop/Env/gymnax_Analog_RL/gymnax/environments/custom'

class TwoStageOTA(environment.Environment):
    """
    JAX Compatible version of TwoStageOTA
    """

    def __init__(self):
        super().__init__()
        self.weights = jnp.array([1, 0, 1, 1, 1, 1, 1, 1, 1])  # Weights for each measurement
        self.model = FlaxMLP()
        input_shape = (1, 16)
        key = jax.random.PRNGKey(0)

        # Initialize model parameters
        dummy_input = jax.random.normal(key, input_shape)
        self.model_params = self.model.init(key, dummy_input)

        # Load and update model parameters
        new_params = self.load_model_params()
        self.update_model_params(new_params)


        # Assume that params remain static for the jit-ed functions
        # JIT compile the step_env method
        self.step_env = jax.jit(self.step_env, static_argnames='params')

        # reward dictionary to accomodate eureka
        self.rew_dict = {}


    def load_model_params(self):
        new_params = {
            'Dense_0': {'kernel': np.load(f'{WEIGHT_ROOT}/layers.0.weight.npy').T, 'bias': np.load(f'{WEIGHT_ROOT}/layers.0.bias.npy')},
            'Dense_1': {'kernel': np.load(f'{WEIGHT_ROOT}/layers.2.weight.npy').T, 'bias': np.load(f'{WEIGHT_ROOT}/layers.2.bias.npy')},
            'Dense_2': {'kernel': np.load(f'{WEIGHT_ROOT}/layers.4.weight.npy').T, 'bias': np.load(f'{WEIGHT_ROOT}/layers.4.bias.npy')},
            'Dense_3': {'kernel': np.load(f'{WEIGHT_ROOT}/layers.6.weight.npy').T, 'bias': np.load(f'{WEIGHT_ROOT}/layers.6.bias.npy')},
            'Dense_4': {'kernel': np.load(f'{WEIGHT_ROOT}/layers.8.weight.npy').T, 'bias': np.load(f'{WEIGHT_ROOT}/layers.8.bias.npy')}
        }
        return new_params

    def update_model_params(self, new_params):
        # Updating the model parameters with loaded values
        for layer in ['Dense_0', 'Dense_1', 'Dense_2', 'Dense_3', 'Dense_4']:
            for param in ['kernel', 'bias']:
                self.model_params['params'][layer][param] = new_params[layer][param]


    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def deNorm_action(
        current_state: EnvState,
        params: EnvParams,
    ) -> EnvState:
        # Denoramlize the action based on the bounds within param
        x0 = (current_state.x0 + 1) * ((params.x0_bounds[1] - params.x0_bounds[0]) / 2) + params.x0_bounds[0]
        x1 = (current_state.x1 + 1) * ((params.x1_bounds[1] - params.x1_bounds[0]) / 2) + params.x1_bounds[0]
        x2 = (current_state.x2 + 1) * ((params.x2_bounds[1] - params.x2_bounds[0]) / 2) + params.x2_bounds[0]
        x3 = (current_state.x3 + 1) * ((params.x3_bounds[1] - params.x3_bounds[0]) / 2) + params.x3_bounds[0]
        x4 = (current_state.x4 + 1) * ((params.x4_bounds[1] - params.x4_bounds[0]) / 2) + params.x4_bounds[0]
        x5 = (current_state.x5 + 1) * ((params.x5_bounds[1] - params.x5_bounds[0]) / 2) + params.x5_bounds[0]
        x6 = (current_state.x6 + 1) * ((params.x6_bounds[1] - params.x6_bounds[0]) / 2) + params.x6_bounds[0]
        x7 = (current_state.x7 + 1) * ((params.x7_bounds[1] - params.x7_bounds[0]) / 2) + params.x7_bounds[0]
        x8 = (current_state.x8 + 1) * ((params.x8_bounds[1] - params.x8_bounds[0]) / 2) + params.x8_bounds[0]
        x9 = (current_state.x9 + 1) * ((params.x9_bounds[1] - params.x9_bounds[0]) / 2) + params.x9_bounds[0]
        x10 = (current_state.x10 + 1) * ((params.x10_bounds[1] - params.x10_bounds[0]) / 2) + params.x10_bounds[0]
        x11 = (current_state.x11 + 1) * ((params.x11_bounds[1] - params.x11_bounds[0]) / 2) + params.x11_bounds[0]
        x12 = (current_state.x12 + 1) * ((params.x12_bounds[1] - params.x12_bounds[0]) / 2) + params.x12_bounds[0]
        x13 = (current_state.x13 + 1) * ((params.x13_bounds[1] - params.x13_bounds[0]) / 2) + params.x13_bounds[0]
        x14 = (current_state.x14 + 1) * ((params.x14_bounds[1] - params.x14_bounds[0]) / 2) + params.x14_bounds[0]
        x15 = (current_state.x15 + 1) * ((params.x15_bounds[1] - params.x15_bounds[0]) / 2) + params.x15_bounds[0]
        return EnvState(
            x0=x0,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x5=x5,
            x6=x6,
            x7=x7,
            x8=x8,
            x9=x9,
            x10=x10,
            x11=x11,
            x12=x12,
            x13=x13,
            x14=x14,
            x15=x15,
            time=current_state.time,
        )

    def reward_compute_input(self, model_output: chex.Array, params: EnvParams) -> list:
        denormalize_params_min = jnp.array([
            params.out0_denormalize[0], params.out1_denormalize[0], params.out2_denormalize[0],
            params.out3_denormalize[0], params.out4_denormalize[0], params.out5_denormalize[0],
            params.out6_denormalize[0], params.out7_denormalize[0], params.out8_denormalize[0]
        ])
        denormalize_params_max = jnp.array([
            params.out0_denormalize[1], params.out1_denormalize[1], params.out2_denormalize[1],
            params.out3_denormalize[1], params.out4_denormalize[1], params.out5_denormalize[1],
            params.out6_denormalize[1], params.out7_denormalize[1], params.out8_denormalize[1]
        ])
        
        constraints = jnp.array([
            params.out0_constraints, params.out1_constraints, params.out2_constraints,
            params.out3_constraints, params.out4_constraints, params.out5_constraints,
            params.out6_constraints, params.out7_constraints
        ])
        out = denormalize_params_min + (((model_output + 1.0) * (denormalize_params_max - denormalize_params_min)) / 2.0)
        reward_input = [out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
            constraints[0], constraints[1], constraints[2], constraints[3], constraints[4],
            constraints[5], constraints[6], constraints[7],
            self.weights[0], self.weights[1], self.weights[2], self.weights[3], self.weights[4],
            self.weights[5], self.weights[6], self.weights[7], self.weights[8]]
        return reward_input
    
    def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:
        reward, self.rew_dict = compute_ota_reward(*(self.reward_compute_input(model_output, params)))
        return reward

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        # Denormalize the action
        #action = self.deNorm_action(action, params)
        jax_state = self.get_obs(state)

        output = self.model.apply(self.model_params, jax_state)
        reward = self.compute_reward(output, params)

        # Update state
        next_state = EnvState(
            x0=action[0],
            x1=action[1],
            x2=action[2],
            x3=action[3],
            x4=action[4],
            x5=action[5],
            x6=action[6],
            x7=action[7],
            x8=action[8],
            x9=action[9],
            x10=action[10],
            x11=action[11],
            x12=action[12],
            x13=action[13],
            x14=action[14],
            x15=action[15],
            time=state.time + 1,
        )
        done = self.is_terminal(next_state, params)

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            reward,
            done,
            {},
        )


    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        # Sample from a uniform distribution within [-1, 1)
        # Split the key for each random number generation
        keys = jax.random.split(key, 17)  # 16 for state variables + 1 to pass on
        
        # Initialize state with uniformly distributed values
        init_state = EnvState(
            x0=jax.random.uniform(keys[0], (), minval=-1, maxval=1),
            x1=jax.random.uniform(keys[1], (), minval=-1, maxval=1),
            x2=jax.random.uniform(keys[2], (), minval=-1, maxval=1),
            x3=jax.random.uniform(keys[3], (), minval=-1, maxval=1),
            x4=jax.random.uniform(keys[4], (), minval=-1, maxval=1),
            x5=jax.random.uniform(keys[5], (), minval=-1, maxval=1),
            x6=jax.random.uniform(keys[6], (), minval=-1, maxval=1),
            x7=jax.random.uniform(keys[7], (), minval=-1, maxval=1),
            x8=jax.random.uniform(keys[8], (), minval=-1, maxval=1),
            x9=jax.random.uniform(keys[9], (), minval=-1, maxval=1),
            x10=jax.random.uniform(keys[10], (), minval=-1, maxval=1),
            x11=jax.random.uniform(keys[11], (), minval=-1, maxval=1),
            x12=jax.random.uniform(keys[12], (), minval=-1, maxval=1),
            x13=jax.random.uniform(keys[13], (), minval=-1, maxval=1),
            x14=jax.random.uniform(keys[14], (), minval=-1, maxval=1),
            x15=jax.random.uniform(keys[15], (), minval=-1, maxval=1),
            time=0)
        
        return self.get_obs(init_state), init_state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x0, state.x1, state.x2, state.x3, state.x4, state.x5, state.x6, state.x7, state.x8, state.x9, state.x10, state.x11, state.x12, state.x13, state.x14, state.x15]).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoStageOTA-custom"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.default_params.num_states

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(params.num_states,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(params.num_states,), dtype=jnp.float32)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
            "x0": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x1": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x2": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x3": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x4": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x5": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x6": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x7": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x8": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x9": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x10": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x11": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x12": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x13": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x14": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x15": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "time": spaces.Discrete(self.default_params.max_steps_in_episode),
            }
        )

from typing import Tuple
import jax.numpy as jnp
def compute_ota_reward(
    out0: float, out1: float, out2: float, out3: float, out4: float, 
    out5: float, out6: float, out7: float, out8: float,
    out0_constraints: Tuple[float, int], out1_constraints: Tuple[float, int], 
    out2_constraints: Tuple[float, int], out3_constraints: Tuple[float, int], 
    out4_constraints: Tuple[float, int], out5_constraints: Tuple[float, int], 
    out6_constraints: Tuple[float, int], out7_constraints: Tuple[float, int],
    w0: int, w1: int, w2: int, w3: int, w4: int, w5: int, 
    w6: int, w7: int, w8: int
) -> float:

    out = jnp.array([out0, out1, out2, out3, out4, out5, out6, out7, out8])
    constraints = jnp.array([
        out0_constraints, out1_constraints, out2_constraints, out3_constraints, 
        out4_constraints, out5_constraints, out6_constraints, out7_constraints
    ])
    weights = jnp.array([w0, w1, w2, w3, w4, w5, w6, w7, w8])
    # Calculate scaled differences from constraints
    scaled_diffs = jnp.where(
        constraints[:, 1] == 1, 
        jnp.clip((out[:-1] - constraints[:, 0]) / constraints[:, 0], 0, 1),
        jnp.clip((constraints[:, 0] - out[:-1]) / constraints[:, 0], 0, 1)
    )
    # Append out[-1] to scaled_diffs
    scaled_diffs = jnp.append(scaled_diffs, out[-1])

    # Apply weights and sum
    weighted_diffs = weights * scaled_diffs
    FoM = jnp.sum(weighted_diffs)

    # Convert FoM to a reward
    reward = -1 * FoM
    return reward
