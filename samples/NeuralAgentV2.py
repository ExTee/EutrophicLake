import gym
from lake import LakeLoadEnv
import matplotlib.pyplot as plt
from train_model import NeuralNetwork
import numpy as np

ALPHA = 0.3
GAMMA = 0.9

m = NeuralNetwork()
#m.load_data('20180417193726')
#m.train_model()


env = LakeLoadEnv()
env.reset()
rewards = [0]

for i_episode in range(20):
    observation = env.reset()

    new_data_x = []
    new_data_y = []
    R = 0
    Q = 0
    S = (0,0)

    for t in range(500):
        env.render()
        
        inp = []

        #looping through actions
        for A in range(12):
            onehot = [0]*12
            onehot[A] = 1
            inp.append([observation[0],observation[1]] + onehot)

        #predictions contains estimate of future value
        predictions = m.model.predict(np.array(inp))

        #Update Q
        new_Q = (1-ALPHA) * Q + ALPHA * (R + GAMMA * max(predictions))


        #Choose action
        if(i_episode < 5):
            A = env.action_space.sample()
        else:
            A = np.argmax(predictions)

        print("Took action: " + str(A))
        S_prime, R_prime, done, info = env.step(A)
    
        #Create a new data point
        new_action = [0]*12
        new_action[A] = 1
        new_data_x.append([S[0], S[1]] + new_action)
        new_data_y.append(new_Q)
        rewards.append(rewards[-1] + R_prime)

        #update variables
        R = R_prime
        Q = new_Q
        S = S_prime




        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    #clean and feed the new data
    '''    
    new_x = []
    new_y = []
    for i in range(1, len(new_data_x)):
        if(new_data_y[i][0] > 0):
            new_x.append(new_data_x[i])
            new_y.append(new_data_y[i][0])
    train_x = np.array(new_x)       
    train_y = np.array(new_y)
    print(train_x,train_y)
    

    
    m.model.fit(train_x,train_y, epochs=20, batch_size=20,verbose=1)
    '''
     
    train_x = np.array(new_data_x[1:])
    train_y = np.array(new_data_y[1:])
    print(train_x)
    print(train_y)

    #m = NeuralNetwork()
    m.model.fit(train_x,train_y, epochs=20, batch_size=20,verbose=1)
    

env.close()

plt.plot(rewards)
plt.show()