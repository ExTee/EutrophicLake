import gym
from lake import LakeLoadEnv
import matplotlib.pyplot as plt


env = LakeLoadEnv()
env.reset()
rewards = [0]

for i_episode in range(5):
    observation = env.reset()
    for t in range(500):
        env.render()
        print(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(0)
        rewards.append(rewards[-1] + reward)
        #print("State : {} -- Reward: {} -- {} -- {}".format(observation, reward, done, info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

plt.plot(rewards)
plt.show()