import numpy as np
import seaborn as sns
import gym
from lake import LakeLoadEnv
from matplotlib import pyplot as plt
def policy(P,M):
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
env.reset()
rewards = [0]
step_reward = []
phase = []
for i_episode in range(5):
    observation = env.reset()
    print("Initial state :{}".format(env.state))
    for t in range(200):
        #env.render()
        action = policy(observation[0],observation[1])
        observation, reward, done, info = env.step(action)
        rewards.append(rewards[-1] + reward)
        step_reward.append(reward)
        phase.append(observation)
        print("Action taken : {}".format(action))
        print("P: {} -- M: {} -- Reward: {}".format(round(observation[0],3),round(observation[1],3), round(reward,2)))
        if done:
            print("Episode finished after {} timesteps\n".format(t+1))
            break
env.close()
plt.subplot(211)
plt.plot(rewards)
plt.title("Total rewards over time - Ecologist Agent")
plt.xlabel("steps")
plt.ylabel("total reward")
plt.subplot(212)
plt.plot(step_reward)
plt.title("Step reward over time - Ecologist Agent")
plt.xlabel("steps")
plt.ylabel("reward at step t")
plt.tight_layout()
plt.show()
phase = np.array(phase)
plt.quiver(phase[:-1, 0], phase[:-1, 1], phase[1:, 0]-phase[:-1, 0], phase[1:, 1]-phase[:-1, 1], scale_units='xy', angles='xy', scale=1, color='g')
plt.title("Trajectory of Ecologist Agent")
plt.xlabel("P")
plt.ylabel("M")
plt.show()
gym.logger.set_level(50)

Ms = np.linspace(0,200,10)
Ps = np.linspace(0,7,15)
all_rewards = []

for m_val in Ms:
  m_reward = []
  for p_val in Ps:
    env = LakeLoadEnv()
    env.start_at_state(float(p_val),float(m_val))
    rewards = 0
    step_reward = []
    phase = []
    for i_episode in range(10):
        observation = env.reset()
        for t in range(200):
            action = policy(observation[0],observation[1])
            observation, reward, done, info = env.step(action)
            rewards+= reward
            if done:
                break
    env.close()
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