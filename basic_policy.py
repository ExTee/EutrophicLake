# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:26:38 2018

@author: Robert
"""

import gym
from lake import LakeLoadEnv
from matplotlib import pyplot as plt
rewards = [0]
states= []

def basic_policy(P,M):
    if P<=2:
        if M<5:
            return 11
        elif M<10:
            return 8
        elif M<20:
            return 7
        elif M<60:
            return 6
        else:
            return 5
    if  2<P<3:
        if M<10:
            return 11
        elif M<20:
            return 7
        elif M<30:
            return 2
        else:
            return 0
    if P>=3:
        if M<15:
            return 11
        elif M<20:
            return 2
        else:
            return 0
    else:
        return 0


env = LakeLoadEnv()
env.start_at_state(1,50)
for i_episode in range(1):
    observation = env.start_at_state(0,50)
    for t in range(2000):

        action = basic_policy(observation[0],observation[1])
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(observation[0])
        print("State : {} -- Reward: {} -- {} -- {}".format(observation, reward, done, info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

plt.plot(rewards)
plt.show()
plt.plot(states)
plt.show()
