import gym
import random
import numpy as np

# Define model
# env = gym.make('NextageOpen-v0')
env = gym.make('Baxter-v0')

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
env.env.frame_skip = 1
env.env.max_wait = 5000

def random_policy(steps=500):
    for _ in range(steps):
        action = env.action_space.sample()
        joint_list = data.qpos.copy().flatten()
        joint_list = action 
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