# -*- coding: utf-8 -*-
"""
Created on Tues Apr 17 16:11:03 2018

@author: XT

Run this file in order
"""

import gym
from lake import LakeLoadEnv
import numpy as np
import time


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

	for i_episode in range(1):
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
	x,y = randomPlay(2000)
	saveCSV(x,y);

if __name__ == '__main__':
	main()

