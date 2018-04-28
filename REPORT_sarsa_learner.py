import seaborn as sns
import gym
import matplotlib.pyplot as plt
import numpy as np
from lake import LakeLoadEnv
from Net import NeuralNetwork

ALPHA = 0.3
GAMMA = 0.9                 #Discount Rate
EPSILON = 0.1               #Epsilon for Epsilon-Greedy
NUM_EPISODES = 100          #Number of total episodes
NUM_MOVES = 1000            #Number of actions performed per episode
NUM_EXPLORATION = 2       #Number of episodes where actions are chosen randomly

#Our network for function approximation
m = NeuralNetwork()

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
        # observation = env.start_at_state(np.random.random_sample() * 5, np.random.random_sample() * 149 + 1)
        print(env.state)

        for t in range(NUM_MOVES):
            # env.render()
            S = env.state

            # Start with random actions
            if (i_episode < NUM_EXPLORATION):
                A, _ = RandomAction(S)
            else:
                A, _ = EpsilonGreedy(S)

            SA_vector = Encode(S, A)
            Q_SA = m.model.predict(SA_vector)[0]

            S_prime, R, done, info = env.step(A)

            A_prime, maxQ_S_prime = BestAction(S_prime)

            # SARSA update
            target = R + GAMMA * Avg(S_prime)

            rewards.append(rewards[-1] + R)
            # print("Action: {}".format(A))

            # add to batch

            temp_x = SA_vector.tolist()[0]
            temp_y = target
            batch_x.append(temp_x)
            batch_y.append(temp_y)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        # Performs a gradient step at the end of an episode, then reset our batches
        m.model.train_on_batch(np.array(batch_x), np.array(batch_y))
        batch_x = []
        batch_y = []

    env.close()

    # return rewards
    return rewards

def play(start_state, moves):
    env = LakeLoadEnv()
    env.reset()
    total_rewards = [0]
    step_rewards = []
    phase = []
    for i_episode in range(1):
        observation = env.start_at_state(start_state[0], start_state[1])
        for t in range(moves):
            # env.render()

            # get current state
            S = env.state
            # Only choose the best action
            A, _ = BestAction(S)
            # Apply the best action
            S_prime, R, done, info = env.step(A)
            # Store reward
            total_rewards.append(total_rewards[-1] + R)
            step_rewards.append(R)
            phase.append(S_prime)
            # print("Action: {}".format(A))
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    # return rewards
    return total_rewards, step_rewards, phase




R1 = learn_Sarsa()
rewards,step_reward,phase = play((1,30), 200)

plt.subplot(211)
plt.plot(rewards)
plt.title("Total rewards over time - SARSA Agent")
plt.xlabel("steps")
plt.ylabel("total reward")
plt.subplot(212)
plt.plot(step_reward)
plt.title("Step reward over time - SARSA Agent")
plt.xlabel("steps")
plt.ylabel("reward at step t")
plt.tight_layout()
plt.show()
phase = np.array(phase)
plt.quiver(phase[:-1, 0], phase[:-1, 1], phase[1:, 0]-phase[:-1, 0], phase[1:, 1]-phase[:-1, 1], scale_units='xy', angles='xy', scale=1, color='g')
plt.title("Trajectory of SARSA Agent")
plt.xlabel("P")
plt.ylabel("M")
plt.show()
gym.logger.set_level(50)

Ms = np.linspace(0,200,10)
Ps = np.linspace(0,7,15)
all_rewards = []

for m_val in reversed(Ms):
  m_reward = []
  for p_val in Ps:
    tot = 0
    for i_ep in range(5):
        tot = tot + play((p_val,m_val), 200)[0][-1]
    rewards = tot
    m_reward.append(rewards)
  all_rewards.append(m_reward)

all_rewards = np.matrix(all_rewards)
ax = sns.heatmap(all_rewards, linewidth=0.5)
plt.xticks([x + 0.5 for x in range(15)],[round(x,1) for x in Ps])
plt.yticks([1.05*x + 0.5 for x in range(10)],[int(x) for x in Ms])
plt.title ("Total reward over 10 episodes depending on starting state")
plt.xlabel("starting P value")
plt.ylabel("Starting M value")
plt.show()