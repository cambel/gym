import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

class BaxterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hand = False
        mujoco_env.MujocoEnv.__init__(self, 'baxter.xml') 
        utils.EzPickle.__init__(self)
        

    def get_qpos(self):
        return self.data.qpos[:self.model.nu]

    def set_qpos(self, new_qpos):
        self.data.qpos[:] = np.concatenate((new_qpos, self.data.qpos[self.model.nu:]))


    def _get_obs(self):
        return np.concatenate([ self.get_qpos() ]) 

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = 0
        info = []
        return self._get_obs(), reward, False, info

    def reset_model(self):
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        return self._get_obs()

    def set_reward_func(self, func):
        self.reward_func = func

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.2
        self.viewer.cam.azimuth = -180
        self.viewer.cam.elevation = -25.5