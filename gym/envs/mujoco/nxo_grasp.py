import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

class NextageGraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # Parameters
        self.max_wait = 500
        self.hand = False
        self.debug = False
        self.auto_render = False 
        self.auto_render_rate = self.max_wait / 10

        mujoco_env.MujocoEnv.__init__(self, 'nxo_hand.xml')
        utils.EzPickle.__init__(self)
        
        self.init_qpos = np.concatenate(([0, -0.0304, -0.25, -1.745, 0.265, 0.164, 0.0558, 0.03, -0.03], self.init_qpos[self.model.nu:]))

    def get_qpos(self):
        if self.hand:
            hand = [round((abs(self.data.qpos[7]) + abs(self.data.qpos[8])) / 0.06)]
            return np.concatenate((self.data.qpos[:self.model.nu-2], hand))
        else:
            return self.data.qpos[:self.model.nu]

    def set_qpos(self, new_qpos):
        if self.hand:
            new_qpos = self._hand2joints(new_qpos)
        self.data.qpos[:] = np.concatenate((new_qpos, self.data.qpos[self.model.nu:]))


    def _get_obs(self):
        return np.concatenate([ self.get_qpos() ]) 


    def reset_model(self):
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        return self._get_obs()

    def set_reward_func(self, func):
        self.reward_func = func

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 1.95
        self.viewer.cam.azimuth = -180
        self.viewer.cam.elevation = -30.66
        self.viewer.cam.lookat[:] = [-0.00033394,  0.02207251,  0.05051445]

    ### Handle movement ###

    def step(self, a):
        """ Forward simulation for actions a. If hand active,
            consider last a[-1] as binary: 
            - 0, hand closed 
            - 1, hand open """

        if self.hand:
            a = self._hand2joints(a)

        elif a.shape != (self.model.nu,):
            print("invalid action:", a, self.hand)
            return None, None, None, None

        self._position_control(self._limit_actions(a))

        reward = 0
        info = []
        return self._get_obs(), reward, False, info

    def _hand2joints(self, a):
        hand = [0.03, -0.03] if a[7] == 1 else [0, 0]
        return np.concatenate((a[:self.model.nu-2], hand))

    def _position_control(self, a):
        """ Compute simulation until achieving given position
          or running out of steps """
        for i in range(self.max_wait):
            # compute simulation
            self.do_simulation(a, self.frame_skip)
            
            # render automatically
            if self.auto_render:
                if i%self.auto_render_rate == 0:
                    self.render()
            
            # check if we achieve the desired position
            qpos = self.data.qpos.copy() 
            if self._compare_arrays(self.norm_jnt(a), self.norm_jnt(qpos[:self.model.nu])):
                if self.debug:
                    print("steps:", i * self.frame_skip)
                break

            if i == self.max_wait-1 and self.debug:
                print("diff", np.round(np.subtract(self.norm_jnt(a), self.norm_jnt(qpos[:self.model.nu])), 4))
                print("diff sum", np.absolute(np.subtract(self.norm_jnt(a), self.norm_jnt(qpos[:self.model.nu]))).sum())
        
    def norm_jnt(self, a):
        """ Normalize joints array """
        arr = a.copy()
        jnt_range = self.model.jnt_range
        for jnt in range(self.model.nu):
            a = 10 /(jnt_range[jnt][0]-jnt_range[jnt][1])
            b = 10 - a * jnt_range[jnt][0]
            arr[jnt] = a * arr[jnt] + b
        return arr 


    def _compare_arrays(self, a, b):
        a = np.round(a, 3)
        b = np.round(b, 3)
        c = abs(a.sum() - b.sum())
        return c <= 0.3
        # return np.array_equal(a, b)

    def _limit_actions(self, a):
        """ Check that no action exceeds the given limits """
        jnt_range = self.model.jnt_range
        for jnt in range(self.model.nu):
            if a[jnt] < jnt_range[jnt][0]:
                a[jnt] = jnt_range[jnt][0]
            elif a[jnt] > jnt_range[jnt][1]:
                a[jnt] = jnt_range[jnt][1]
        return a