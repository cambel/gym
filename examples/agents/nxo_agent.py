import gym
import random
import numpy as np

# Define model
# env = gym.make('NextageOpen-v0')
env = gym.make('NextageGrasp-v0')

# Always reset first of all
env.reset()
env.render()

# Variables
model =  env.env.model
data = env.env.data
viewer = env.env.viewer
sim = env.env.sim

# Testing params
env.env.debug = True
env.env.hand = True
ROPEN = -0.03
LOPEN = 0.03
CLOSE = 0
env.env.frame_skip = 15
env.env.max_wait = 5000

def random_policy(steps=500):
    for _ in range(steps):
        action = env.action_space.sample()
        joint_list = model.data.qpos.copy().flatten()[:9]
        action[0] = -.05 if joint_list[0] > 0.5 else action[0]
        action[0] = .05 if joint_list[0] < -0.5 else action[0]
        joint_list += action / 50.0
        env.step(joint_list)
        env.render()

def display_image():
    data = env.env.get_resize_image(300, 300)
    from scipy.misc import toimage
    toimage(data).show()

def reset():
    env.reset()
    render(10)

def move_joint(joint, distance):      
    action = env.env.get_qpos()
    if joint == 7:
        action[7] = distance
    else:
        action[joint] += distance
    env.step(action)
    render(5)

def render(steps):
    for _ in range(steps):
        # move_joint(0,0)
        env.env.sim.step()
        env.render()

def body_pos(body, pos):
    ### Example modify body pos
    b = model.body_pos.copy()
    b[body] = pos
    model.body_pos = b
    model.kinematics()
    model.step()
    render(1)


# env.render()
# # action = [0, -0.0104, 0, -1.745, 0.265, 0.164, 0.0558, 0, 0]
# # action = [ 0., 0.063, -0.61,  -0.685,  1.83,  -1.489, -2.594,  0.03,  -0.03]
# action = [5, 0.06334643698466469, -0.6105326945985968, -0.6849219029920323, 1.8305557027770751, -1.4894137507168421, -2.593769688808892, LOPEN, ROPEN]
# env.step(action)
# env.render()

# print "target pos", env.env.data.qpos.flatten()[:9]
# print "target eefp", env.env.data.site_xpos.flatten()

# render(100)

# reset()

# action = [ 0.,     0.194, -0.815, -0.561,  0.265, -0.14,   0.056,  0.,    -0., 0 ,0]
# env.step(action)
# env.render()

### Example of changing programatically the qpos qvel
# q = env.env.data.qpos.copy()
# q[15] = .5

# env.env.data.qpos = q
# env.env.kinematics()
# env.env.step()

# random_policy(10)
move_joint(2, -.5)
render(100)
# for c in env.env.data.contact:
#     print (c.geom1, c.geom2)