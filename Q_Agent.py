import gym
from lake2 import LakeLoadEnv
import matplotlib.pyplot as plt
from Net import NeuralNetwork
import numpy as np

ALPHA = 0.3
GAMMA = 0.9                 #Discount Rate 
EPSILON = 0.1               #Epsilon for Epsilon-Greedy
NUM_EPISODES = 20          #Number of total episodes
NUM_MOVES = 500            #Number of actions performed per episode
NUM_EXPLORATION = 0       #Number of episodes where actions are chosen randomly

#Our network for function approximation
m = NeuralNetwork()
FILEOUT = ''


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

def learn_Q():
    #Load the data from our samples
    m.load_data('20180420154626')
    m.train_model()
    #m.load_model('./networks/20180420162633.h5')
    FILEOUT = m.save_model()

    env = LakeLoadEnv()
    env.reset()
    rewards = [0]
    rewards_nc = [0]

    for i_episode in range(NUM_EPISODES):

        batch_x = []
        batch_y = []

        observation = env.reset()
        observation = env.start_at_state(np.random.random_sample() * 5, np.random.random_sample() * 149 + 1)
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

            #Q learning update
            target = R + GAMMA * maxQ_S_prime

            #SARSA update
            #target = R + GAMMA * Avg(S_prime)

            rewards.append(rewards[-1]+R)
            rewards_nc.append(R)
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
    return rewards, rewards_nc

'''
    Use the play function to play according to a specific start state
    @ start_state   : Starting (P,M)
    @ moves         : number of moves allowed
'''
def play(start_state, moves):

    #load our trained model
    m.load_model('./networks/20180420162633.h5')
    print("Loading {}".format(FILEOUT))
    #m.load_model('./networks/20180423134758.h5')

    env = LakeLoadEnv()
    env.reset()
    rewards = [0]

    for i_episode in range(5):

        #observation = env.start_at_state(start_state[0],start_state[1])
        observation = env.reset()
        for t in range(200):
            env.render()
            
            #get current state
            S = env.state

            #Only choose the best action
            A, _ = BestAction(S)

            #Apply the best action
            S_prime, R, done, info = env.step(A)

            #Store reward
            if t>20:
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
    Rewards_cum, Rewards = learn_Q()

    #Plot the cumulative rewards
    
    plt.figure(figsize=(14,10))
    plt.plot(Rewards_cum)
    plt.title('Q Learning with Function Approximation')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.savefig('Q_train_Rcum.png')
    plt.show()
    
    plt.figure(figsize=(14,10))
    plt.plot(Rewards)
    plt.title('Q Learning with Function Approximation')
    plt.xlabel('Steps')
    plt.ylabel('Reward at each step')
    plt.savefig('Q_train_R.png')
    plt.show()
    
    
    #Random play at an arbitrary location
    R2 = play((5, 35), 100)
    plt.figure(figsize=(14,10))
    plt.plot(R2)
    plt.title('Policy applied to a random starting state')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.savefig('Q_test_Rcum.png')
    plt.show()
    



    #m.save_model()


if __name__ == '__main__':
    main()
