import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

NXO_DOF = 9

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class NextageEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'nxo_basic.xml') #TODO: Frameskip, how does this affect?
        utils.EzPickle.__init__(self)
        self.init_qpos = np.concatenate(([0, -0.0104, 0, -1.745, 0.265, 0.164, 0.0558, 0, 0], self.init_qpos[NXO_DOF:]) )

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat
        # ,
        #                        data.qvel.flat,
        #                        data.site_xpos.flat
                               ]) # TODO: what else can I use?

    def step(self, a):
        # self.do_simulation(self.init_qpos, self.frame_skip)
        self._position_control(self._limit_actions(a))
        reward = 0
        info = []
        return self._get_obs(), reward, False, info

    def _position_control(self, a):
        i = 0
        while True:
            # compute simulation
            self.do_simulation(np.concatenate((a[:NXO_DOF], self.model.data.qpos[NXO_DOF:].flatten())), self.frame_skip)
            # render for the user
            self.render()
            # check if we achieve the desired position
            qpos = self.model.data.qpos.flatten()
            if self._compare_arrays(a[:NXO_DOF], qpos[:NXO_DOF]):
                break
            if i%500 == 0:
                print("not looking good :(")
                print("action", np.round(a[:NXO_DOF], 3))
                print("qpos", np.around(qpos[:NXO_DOF], 3))
            i+=1
            
    def _compare_arrays(self, a, b):
        a = np.round(a, 3)
        b = np.round(b, 3)
        c = abs(a.sum() - b.sum())
        return c <= 0.4
        # return np.array_equal(a, b)

    def _limit_actions(self, a):
        jnt_range = self.model.jnt_range
        for jnt in range(NXO_DOF):
            if a[jnt] < jnt_range[jnt][0]:
                a[jnt] = jnt_range[jnt][0]
            elif a[jnt] > jnt_range[jnt][1]:
                a[jnt] = jnt_range[jnt][1]
        return a


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
        self.viewer.cam.distance = 2.30
        self.viewer.cam.azimuth = -140
        self.viewer.cam.elevation = -32