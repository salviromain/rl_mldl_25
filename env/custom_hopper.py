"""
Goal-reaching Hopper environment with domain randomization.
"""

from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, include_goal_in_obs=True):
        self.include_goal_in_obs = include_goal_in_obs
        self.target_xy = np.array([5.0, 0.0])  # placeholder
        MujocoEnv.__init__(self, frame_skip=4)
        utils.EzPickle.__init__(self, domain, include_goal_in_obs)

        # Store default masses for later use in domain randomization
        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        
        if domain == 'source':
            self.sim.model.body_mass[1] *= 0.7
        elif domain == 'target':
            self.sim.model.body_mass[1] *= 1.2  # Example for target domain

    def sample_parameters(self):
        """Randomize the hopper's link masses between 0.5x to 1.5x."""
        return self.original_masses * self.np_random.uniform(0.5, 1.5, size=self.original_masses.shape)

    def set_random_parameters(self):
        new_masses = self.sample_parameters()
        self.set_parameters(new_masses)

    def get_parameters(self):
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, masses):
        self.sim.model.body_mass[1:] = masses

    def reset_model(self):
        # Sample a new goal for this episode
        self.target_xy = self.np_random.uniform(low=[5.0, -4.0], high=[10.0, 4.0])
        #print(self.target_xy)

        # Reset state
        qpos = self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        # Optional: apply domain randomization
        self.set_random_parameters()

        return self._get_obs()

    def _get_obs(self):
        base_obs = np.concatenate([
            self.sim.data.qpos.flat[1:],  # omit root x
            self.sim.data.qvel.flat
        ])
        if self.include_goal_in_obs:
            xpos, ypos = self.sim.data.qpos[0:2]
            rel_goal = self.target_xy - np.array([xpos, ypos])
            return np.concatenate([base_obs, rel_goal])
        return base_obs

    from mujoco_py import functions

    def step(self, a):
        pos_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[0:2]
    
        velocity_xy = (pos_after - pos_before) / self.dt
        forward_vel = velocity_xy[0]
    
        alive_bonus = 1.0
        reward = forward_vel + alive_bonus - 1e-3 * np.square(a).sum()
    
        height = self.sim.data.qpos[1]
    
        # Get torso rotation matrix (3x3)
        rot_mat = self.sim.data.body_xmat[self.model.body_name2id("torso")].reshape(3, 3)
    
        # Calculate roll and pitch from rotation matrix
        roll = np.arctan2(rot_mat[2,1], rot_mat[2,2])
        pitch = np.arctan2(-rot_mat[2,0], np.sqrt(rot_mat[2,1]**2 + rot_mat[2,2]**2))
    
        done = not (
            np.isfinite(self.state_vector()).all() and
            (np.abs(roll) < 0.5) and
            (np.abs(pitch) < 0.5) and
            (height > 0.7)
        )
    
        ob = self._get_obs()
        return ob, reward, done, {}



    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_mujoco_state(self, state):
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="env.custom_hopper:CustomHopper",  # Adjust the module path as needed
    max_episode_steps=500,
    kwargs={"domain": "source", "include_goal_in_obs": True}

)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="env.custom_hopper:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="env.custom_hopper:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "target"}
)
