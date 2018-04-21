import gym
from lake import LakeLoadEnv
import matplotlib.pyplot as plt
from Net import NeuralNetwork
import numpy as np

ALPHA = 0.3
GAMMA = 0.9                 #Discount Rate 
EPSILON = 0.1               #Epsilon for Epsilon-Greedy
NUM_EPISODES = 100         #Number of total episodes
NUM_MOVES = 2000            #Number of actions performed per episode
NUM_EXPLORATION = 2       #Number of episodes where actions are chosen randomly

#Our network for function approximation
m = NeuralNetwork()

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

def Avg(S):
    inp = []

    #looping through actions
    for A in range(12):
        onehot = [0]*12
        onehot[A] = 1
        inp.append([S[0],S[1]] + onehot)


    #predictions contains estimate of future value
    predictions = m.model.predict(np.array(inp))

    avg = sum(predictions)/ len(predictions)

    return avg

def Encode(S,A):
    A_onehot = [0]*12
    A_onehot[A] = 1

    encoded = [[S[0], S[1]] + A_onehot]
    encoded = np.array(encoded)
    return encoded

def learn_Sarsa():

    env = LakeLoadEnv()
    env.reset()
    rewards = [0]

    for i_episode in range(NUM_EPISODES):

        batch_x = []
        batch_y = []

        observation = env.reset()
        #observation = env.start_at_state(np.random.random_sample() * 5, np.random.random_sample() * 149 + 1)
        print(env.state)

        for t in range(NUM_MOVES):
            env.render()
            
            S = env.state

            #Start with random actions
            if(i_episode < NUM_EXPLORATION):
                A, _ = RandomAction(S)
            else:
                A, _ = EpsilonGreedy(S)

            SA_vector = Encode(S,A)
            Q_SA = m.model.predict(SA_vector)[0]


            S_prime, R, done, info = env.step(A)


            A_prime, maxQ_S_prime = BestAction(S_prime)



            #SARSA update
            target = R + GAMMA * Avg(S_prime)
            if i_episode > 80 and t>1:
                rewards.append(R)
            print("Action: {}".format(A))

            #add to batch

            temp_x = SA_vector.tolist()[0]
            temp_y = target
            batch_x.append(temp_x)
            batch_y.append(temp_y)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        #Performs a gradient step at the end of an episode, then reset our batches
        m.model.train_on_batch(np.array(batch_x),np.array(batch_y))
        batch_x = []
        batch_y = []
        

    env.close()

    #return rewards
    return rewards

'''
    Use the play function to play according to a specific start state
    @ start_state   : Starting (P,M)
    @ moves         : number of moves allowed
'''
def play(start_state, moves):

    env = LakeLoadEnv()
    env.reset()
    rewards = [0]

    for i_episode in range(1):

        observation = env.start_at_state(start_state[0],start_state[1])

        for t in range(moves):
            env.render()
            
            #get current state
            S = env.state

            #Only choose the best action
            A, _ = BestAction(S)

            #Apply the best action
            S_prime, R, done, info = env.step(A)

            #Store reward
            rewards.append(rewards[-1]+R)

            print("Action: {}".format(A))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

    #return rewards
    return rewards

def main():
    #Learn the environment
    R1 = learn_Sarsa()

    #Plot the cumulative rewards
    plt.plot(R1)
    plt.ylim([-20,20])
    plt.show()

    #Random play at an arbitrary location
    #R2 = play((0.01, 137.12), 1000)
    #plt.plot(R2)
    #plt.show()

    #m.save_model()


if __name__ == '__main__':
    main()
