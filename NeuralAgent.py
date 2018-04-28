import gym
from lake2 import LakeLoadEnv
import matplotlib.pyplot as plt
from Net import NeuralNetwork
import numpy as np


m = NeuralNetwork()
#m.load_data('20180420153744')
m.load_data('20180420154626')
m.train_model()


env = LakeLoadEnv()
env.reset()
rewards = [0]

for i_episode in range(1):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)

        
        inp = []
        for A in range(12):
            onehot = [0]*12
            onehot[A] = 1
            inp.append([observation[0],observation[1]] + onehot)
        
        predictions = m.model.predict(np.array(inp))
        print(predictions)


        action = np.argmax(predictions)
        print("Took action: " + str(action))

        observation, reward, done, info = env.step(action)
        print("Predicted Reward: {}  - Actual: {}".format(predictions[action], reward))
        rewards.append(rewards[-1] + reward)
        #print("State : {} -- Reward: {} -- {} -- {}".format(observation, reward, done, info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

plt.plot(rewards)
plt.show()