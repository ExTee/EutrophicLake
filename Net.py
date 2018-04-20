import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
import time

class NeuralNetwork():

	def __init__(self):
		self.model = self.create_model()

	def create_model(self):
		model = Sequential()

		model.add(Dense(64, input_shape = (14,), activation = "relu"))
		model.add(Dense(32))
		model.add(Dense(1))

		model.compile(
			loss = 'mean_squared_error',
			optimizer = 'adam',
			metrics = ['mae']
			)

		return model

	def plot(self):
		plot_model(self.model, to_file='model.png', show_shapes=True)

	def train_model(self):
		self.model.fit(self.train_x, self.train_y,epochs=10, batch_size=20, verbose=1)

	def load_data(self, filename):

		self.train_x = genfromtxt('./samples/' + filename + '_X.csv', delimiter=',')
		self.train_y = genfromtxt('./samples/' + filename + '_Y.csv', delimiter=',')

		#print(train_y.shape)
		#print(train_y)
		#print(train_x.shape)
		#print(train_x)
		#return train_x, train_y
	def save_model(self):
		timestr = time.strftime("%Y%m%d%H%M%S")
		file_out = './networks/' + timestr + '.h5'
		self.model.save(file_out)


def main():
	#Uncomment the lines below to show our network
	#m = NeuralNetwork()
	#m.plot()

	m = NeuralNetwork()
	m.load_data('20180417193726')
	m.train_model()
	#m.save_model()


if __name__ == '__main__':
	main()




