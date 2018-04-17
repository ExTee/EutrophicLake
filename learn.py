# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:26:38 2018

@author: Robert
"""

import gym
from lake import LakeLoadEnv
env = LakeLoadEnv()
env.reset()
for i_episode in range(1):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()