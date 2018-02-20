import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

NXO_DOF = 8

def mass_center(model):
    mass = model.body_mass
    xpos = data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class NextageGraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'nxo_hand.xml') #TODO: Frameskip, how does this affect?
        utils.EzPickle.__init__(self)
        self.init_qpos = np.concatenate(([0, -0.0204, -0.25, -1.745, 0.265, 0.164, 0.0558, 0.03, -0.03], self.init_qpos[NXO_DOF+1:]) )
        self.max_wait = 300
        self.debug = False

    def _get_obs(self):
        return np.concatenate([  self.data.qpos.flat
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

    # def _position_control(self, a):
    #   q = self.data.qpos
    #   print "qpos", q
    #   hand = [0.03, -0.03] if a[7] == 1 else [0, 0]
    #   new_qpos = np.concatenate((a[:NXO_DOF-1], hand, q[NXO_DOF+1:].flatten()))
    #   self.data.qpos = np.reshape(new_qpos, (12, 1))
    #   print "qpos2", q
    #   self.model.kinematics()
    #   self.model.step()


    def _position_control(self, a):
        i = 0
        while True:
            hand = [0.03, -0.03] if a[7] == 1 else [0, 0]
            # compute simulation
            self.do_simulation(np.concatenate((a[:NXO_DOF-1], hand)), self.frame_skip)
            self.last_action = np.concatenate((a[:NXO_DOF-1], hand))
            # render for the user
            self.render()
            # check if we achieve the desired position
            qpos = self.data.qpos.flatten() 
            if self._compare_arrays(a[:NXO_DOF], qpos[:NXO_DOF]) and abs(qpos[8]-qpos[7]) < 0.005:
                break
            if i%self.max_wait == 0 and self.debug:
                print("not looking good :(")
                print("action", np.round(a[:NXO_DOF+1], 3))
                print("qpos", np.around(qpos[:NXO_DOF+1], 3))
            
            # Don't wait anymore, continue...
            if i >= self.max_wait:
                break
            i+=1

    def _compare_arrays(self, a, b):
        a = np.round(a, 3)
        b = np.round(b, 3)
        c = abs(a.sum() - b.sum())
        return c <= 0.3
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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 1.95
        self.viewer.cam.azimuth = -180
        self.viewer.cam.elevation = -30.66
        self.viewer.cam.lookat[:] = [-0.00033394,  0.02207251,  0.05051445]

    def set_qpos(self, value):
        q = self.data.qpos.flatten().copy()
        for i in range(len(value)):
          q[i] = value[i]
        self.data.qpos = q