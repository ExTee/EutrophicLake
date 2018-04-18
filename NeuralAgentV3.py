import gym
from lake import LakeLoadEnv
import matplotlib.pyplot as plt
from train_model import NeuralNetwork
import numpy as np

ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.1

m = NeuralNetwork()
#m.load_data('20180417193726')
#m.train_model()


'''
    Returns best action and Q value 
'''
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

for i_episode in range(50):
    observation = env.reset()


    for t in range(50):
        env.render()
        
        S = env.state

        #Start with random actions
        if(i_episode <10):
            A, _ = RandomAction(S)
        else:
            A, _ = EpsilonGreedy(S)

        SA_vector = Encode(S,A)
        Q_SA = m.model.predict(SA_vector)[0]


        S_prime, R, done, info = env.step(A)


        A_prime, maxQ_S_prime = BestAction(S_prime)

        target = R + GAMMA * maxQ_S_prime
        rewards.append(R)
        print("Action: {}".format(A))

        #Perform one gradient step
        m.model.train_on_batch(SA_vector, [target])
        

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


env.close()

plt.plot(rewards)
plt.show()