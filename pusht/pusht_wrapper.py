import os
import numpy as np
import gym

import sys
sys.path.append('/home/aurora/handful-of-trials-pytorch/dyanmics_model/new_model')
from pusht.pusht_env import PushTEnv
from pusht.utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            add_noise: int = 0,
            noise_std = 0.1,
        ):
        """
        Args
        -----
        with_velocity / with_target : see PushTEnv

        add_noise :
            0 → no noise (default)
            1 → additive Gaussian noise is applied to every action dimension
            2 → additive Gaussian noise is applied to every *state* dimension

        noise_std : float | Sequence[float]
            Standard deviation(s) of the Gaussian noise.  Can be
            • scalar → same σ for all dimensions, or
            • sequence / list / np.ndarray of length = action_dim (mode 1) or
              state_dim  (mode 2).
        """
        super().__init__(with_velocity=with_velocity, with_target=with_target)
        self.action_dim = self.action_space.shape[0]
        self.add_noise = add_noise  # 0: none, 1: action noise, 2: state noise

        # --------------------------------------------------------------
        # Noise standard deviation can be:
        #   • scalar        → same σ across all dims
        #   • sequence/list → per-dim σ (length must match action_dim or
        #                      state_dim depending on noise mode)
        # We keep the original user-supplied object for reference, actual
        # vector σ is materialised lazily in the step() call so that we can
        # raise useful dimension-mismatch errors only when the information
        # (action_dim / state_dim) is available.
        # --------------------------------------------------------------
        self._noise_std_raw = noise_std
        
        # Set state dimension based on velocity option
        if self.with_velocity:
            self.state_dim = 7  # [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        else:
            self.state_dim = 5  # [agent_x, agent_y, T_x, T_y, angle]
    
    def step(self, action):
        """
        Override step method to add Gaussian noise to state if enabled
        """
        """Execute one environment step with optional additive Gaussian noise.

        add_noise == 0  →  deterministic (original PushTEnv step)
        add_noise == 1  →  noise added to the *action* before stepping
        add_noise == 2  →  noise added to the *observed next state* afterwards

        When `noise_std` is a sequence its length must match the relevant
        dimension (action_dim or state_dim).  A scalar standard deviation is
        broadcast to all dimensions.
        """

        # ------------------------------------------------------------------
        def _vector_std(std_raw, dim_required):
            """Return per-dim σ vector as a np.ndarray of length *dim_required*."""
            if np.isscalar(std_raw):
                return float(std_raw)  # scalar ok → broadcast later
            std_arr = np.asarray(std_raw, dtype=float)
            assert std_arr.shape[0] == dim_required, (
                f"noise_std length {std_arr.shape[0]} does not match required "
                f"dimension {dim_required}")
            return std_arr

        # ----------------------------- no noise ----------------------------
        if self.add_noise == 0:
            return super().step(action)

        # ------------------ noise on ACTION --------------------------------
        if self.add_noise == 1:
            scale = _vector_std(self._noise_std_raw, self.action_dim)
            noise = np.random.normal(loc=0.0, scale=scale, size=self.action_dim)
            action = action + noise
            obs, reward, done, info = super().step(action)
            return obs, reward, done, info

        # ------------------ noise on STATE ---------------------------------
        if self.add_noise == 2:
            obs, reward, done, info = super().step(action)
            scale = _vector_std(self._noise_std_raw, self.state_dim)
            noise = np.random.normal(loc=0.0, scale=scale, size=self.state_dim)
            info['state'] = info['state'] + noise
            return obs, reward, done, info

        # Invalid add_noise option
        raise ValueError(f"Unknown add_noise mode {self.add_noise}; expected 0,1,2")

    def sample_random_init_goal_states(self, seed, random_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state(return_goal=False):
            if self.with_velocity:
                if return_goal:
                    return np.array(
                        [
                            50,
                            50,
                            256,
                            256,
                            np.pi / 4,
                            0,
                            0,  # agent velocities default 0
                        ]
                    )
                else:
                    return np.array(
                        [
                            rs.randint(50, 450),
                            rs.randint(50, 450),
                            rs.randint(100, 400),
                            rs.randint(100, 400),
                            rs.randn() * 2 * np.pi - np.pi,
                            0,
                            0,  # agent velocities default 0
                        ]
                    )
            else:
                if return_goal:
                    return np.array(
                        [
                            50,
                            50,
                            256,
                            256,
                            np.pi / 4,
                        ]
                    )
                else:
                    return np.array(
                        [
                            rs.randint(50, 450),
                            rs.randint(50, 450),
                            rs.randint(100, 400),
                            rs.randint(100, 400),
                            rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state(return_goal=True)
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        eef_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
        pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9 and eef_diff < 20
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'pos_diff': pos_diff,
            'angle_diff': angle_diff,
            'eef_diff': eef_diff,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
