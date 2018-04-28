# -*- coding: utf-8 -*-
"""
Created on Tues Apr 17 16:11:03 2018

@author: XT

Run this file in order
"""

import gym
from lake2 import LakeLoadEnv
import numpy as np
import time


def exhaustiveSampling(num_iterations):
	env = LakeLoadEnv()
	env.reset()

	data_x = []
	data_y = []

	for P in np.linspace(0.0, env.pThresh, 35):
		for M in np.linspace(0.0, env.mThresh, 300):
			S = env.start_at_state( P, M)

			for t in range(num_iterations):
				env.render()
				for A in range(12):
					S_prime, R, done, info = env.step(A)

					if (R > -100):
						one_hot = [0]*12
						one_hot[A] = 1

						X = [S[0],S[1]] + one_hot
						Y = [R]

						data_x.append(X)
						data_y.append(Y)
	env.close()

	return data_x,data_y

		
'''
	Interacts with the environment randomly to gather samples
'''
def randomPlay(num_iterations):

	#load lake
	env = LakeLoadEnv()
	env.reset()

	#State and Action pairs stored as data
	data_x = []
	data_y = []

	for i_episode in range(10000):
		S = env.reset()

		for t in range(num_iterations):
			env.render()

			A = env.action_space.sample()

			#one-hot encode action
			one_hot = [0]*12
			one_hot[A] = 1


			S_prime, R, done, info = env.step(A)

			#store the data in an array which corresponds to one entry
			X = [S[0],S[1]] + one_hot
			Y = [R]

			#append the entry to our data
			data_x.append(X)
			data_y.append(Y)

			print("S: {}	A: {}	R:{}".format(S,A,R))


			S = S_prime

			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	env.close()
	return (data_x,data_y)

'''
	Saves the samples generated in a csv file
'''
def saveCSV(data_x,data_y):
	data_x = np.asarray(data_x)
	data_y = np.asarray(data_y)

	timestr = time.strftime("%Y%m%d%H%M%S")
	file_out = "./samples/" + timestr

	np.savetxt(file_out + "_X" + ".csv", data_x, delimiter=",")
	np.savetxt(file_out + "_Y" + ".csv", data_y, delimiter=",")

	print("Samples Saved in {}".format(file_out))


def main():
	#x,y = randomPlay(2)
	x,y = exhaustiveSampling(1)
	saveCSV(x,y);

if __name__ == '__main__':
	main()

