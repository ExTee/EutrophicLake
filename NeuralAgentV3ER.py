import gym
from lake import LakeLoadEnv
import matplotlib.pyplot as plt
from train_model import NeuralNetwork
import numpy as np
from random import shuffle

ALPHA = 0.3
GAMMA = 0.95
EPSILON = 0.2
NUM_EPISODES = 100
NUM_MOVES = 2000


m = NeuralNetwork()

'''
    Returns best action and Q value 
'''

#Stores (S,a,r,S')
class Sample(object):
    def __init__(self, SA, R, Sprime):
        self.SA = SA
        self.R = R
        self.Sprime = Sprime
        
    def target(self):
        A_prime, maxQ_S_prime = BestAction(self.Sprime)
        return self.R + GAMMA * maxQ_S_prime

def EpsilonGreedy(S):

    #looping through actions
    inp = []
    for A in range(12):
        onehot = [0]*12
        onehot[A] = 1
        inp.append([S[0],S[1]] + onehot)


    #predictions contains estimate of future value
    predictions = m.model.predict(np.array(inp))

    if (np.random.random_sample() < EPSILON):
        amax = np.random.random_integers(0,11)
    else:
        amax = np.argmax(predictions)

    return (amax, predictions[amax])

def RandomAction(S):

    #looping through actions
    inp = []
    for A in range(12):
        onehot = [0]*12
        onehot[A] = 1
        inp.append([S[0],S[1]] + onehot)


    #predictions contains estimate of future value
    predictions = m.model.predict(np.array(inp))

    amax = np.random.random_integers(0,11)

    return (amax, predictions[amax])

def BestAction(S):
    inp = []

    #looping through actions
    for A in range(12):
        onehot = [0]*12
        onehot[A] = 1
        inp.append([S[0],S[1]] + onehot)


    #predictions contains estimate of future value
    predictions = m.model.predict(np.array(inp))


    amax = np.argmax(predictions)

    return (amax, predictions[amax])

def Encode(S,A):
    A_onehot = [0]*12
    A_onehot[A] = 1

    encoded = [[S[0], S[1]] + A_onehot]
    encoded = np.array(encoded)
    return encoded


env = LakeLoadEnv()
env.reset()
rewards = [0]

samples = []

for i_episode in range(NUM_EPISODES):
    print(i_episode)
    batch_x = []
    batch_y = []

    observation = env.reset()

    for t in range(NUM_MOVES):
        #env.render()
        
        S = env.state

        #Start with random actions
        if(i_episode <0):
            A, _ = RandomAction(S)
        else:
            A, _ = EpsilonGreedy(S)

        SA_vector = Encode(S,A)
        Q_SA = m.model.predict(SA_vector)[0]


        S_prime, R, done, info = env.step(A)
        

        A_prime, maxQ_S_prime = BestAction(S_prime)

        target = R + GAMMA * maxQ_S_prime
        rewards.append(R)
        #print("Action: {}".format(A))

        #add to batch

        temp_x = SA_vector.tolist()[0]
        temp_y = target
        batch_x.append(temp_x)
        batch_y.append(temp_y)
        
        S = Sample(temp_x, R, S_prime)
        samples.append(S)

        #Perform one gradient step
        #m.model.train_on_batch(SA_vector, [target])

        if done:
            env.reset()
            #print("Episode finished after {} timesteps".format(t+1))
    #print(batch_y)
    
    shuffle(samples)
    chosen = samples[:NUM_MOVES]
    #pick num_moves at random
    trainX = [el.SA for el in chosen]
    trainY = [el.target() for el in chosen]
    m.model.train_on_batch(np.array(trainX),np.array(trainY))
    batch_x = []
    batch_y = []


env.close()


plt.plot(rewards)
plt.show()

env.reset()
env.state = np.array([10,20])
rewards = []
phase = []
actions = []
for i in range(300):
    S = env.state
    #env.render()
    A, _ = BestAction(S)
    S_prime, R, done, info = env.step(A)
    rewards.append(R)
    phase.append(S_prime)
    actions.append(A)
    if done:
        break
phase = np.array(phase)
plt.plot(rewards)
plt.show()
plt.quiver(phase[:-1, 0], phase[:-1, 1], phase[1:, 0]-phase[:-1, 0], phase[1:, 1]-phase[:-1, 1], scale_units='xy', angles='xy', scale=1, color='g')
plt.show()