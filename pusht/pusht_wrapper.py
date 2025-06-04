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
            add_noise=0, # 0: no noise, 1: add noise to action, 2: add noise to state
            noise_std=0.1,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target, 
        )
        self.action_dim = self.action_space.shape[0]
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Set state dimension based on velocity option
        if self.with_velocity:
            self.state_dim = 7  # [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        else:
            self.state_dim = 5  # [agent_x, agent_y, T_x, T_y, angle]
    
    def step(self, action):
        """
        Override step method to add Gaussian noise to state if enabled
        """
        if self.add_noise == 0:
            obs, reward, done, info = super().step(action)
        else:
            if self.add_noise == 1:
                noise = np.random.normal(0, self.noise_std, size=self.action_dim)
                action = action + noise
            obs, reward, done, info = super().step(action)
            if self.add_noise == 2:
                noise = np.random.normal(0, self.noise_std, size=self.state_dim)
                info['state'] = info['state'] + noise
        return obs, reward, done, info

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
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
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
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