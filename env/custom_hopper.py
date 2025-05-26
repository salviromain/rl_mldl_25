"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, target_xy=None, include_goal_in_obs=True):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.include_goal_in_obs = include_goal_in_obs
        self.target_xy = target_xy  # externally passed fixed target (x, y) or None

        if domain == 'source':
            self.sim.model.body_mass[1] *= 0.7

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        
        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        #         at the start of each training episode.
        
        raise NotImplementedError()

        return


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xpos, ypos, height, ang = self.sim.data.qpos[0:4]

        # Reward: encourage moving toward goal
        distance_to_goal = np.linalg.norm(np.array([xpos, ypos]) - self.target_xy)
        reward = -distance_to_goal
        reward += 1.0 if distance_to_goal < 0.5 else 0.0  # bonus for getting close
        reward -= 1e-3 * np.square(a).sum()  # small penalty for large actions

        s = self.state_vector()
        done = not (
            np.isfinite(s).all() and
            (np.abs(s[2:]) < 100).all() and
            (height > .7) and (abs(ang) < .2)
        )
        ob = self._get_obs()
        return ob, reward, done, {'goal': self.target_xy}



    def _get_obs(self):
        base_obs = np.concatenate([
            self.sim.data.qpos.flat[1:],  # exclude x pos
            self.sim.data.qvel.flat
        ])
        if self.include_goal_in_obs:
            xpos, ypos = self.sim.data.qpos[0], self.sim.data.qpos[1]
            rel_goal = self.target_xy - np.array([xpos, ypos])
            return np.concatenate([base_obs, rel_goal])
        return base_obs


    def reset_model(self):
        """Reset the environment to a random initial state and sample new target."""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        # Sample a new (x, y) goal if not provided
        if self.target_xy is None:
            self.target_xy = self.np_random.uniform(low=[1.5,-3.0], high=[10.0, 3.0])

        return self._get_obs())


   


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

