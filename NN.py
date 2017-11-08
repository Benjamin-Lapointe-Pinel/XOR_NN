import math
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
	def __init__(self, layer_sizes):
		if not isinstance(layer_sizes, list):
			raise TypeError("layer_sizes must be a list")
		if not all(isinstance(item, int) for item in layer_sizes):
			raise TypeError("layer_sizes must be a list of integer")

		self.LEARNING_RATE = 1
		self.NBR_OF_LAYER = len(layer_sizes)
		
		self.W = []
		for i, j in enumerate(layer_sizes):
			if i == self.NBR_OF_LAYER - 1:
				break
			self.W.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))

	@staticmethod
	def activation(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def activation_derivative(x):
		return NeuralNetwork.activation(x) * (1 - NeuralNetwork.activation(x))

	def feed_forward(self, X):
		self.S = np.empty(self.NBR_OF_LAYER - 1, dtype=object)
		self.A = np.empty(self.NBR_OF_LAYER, dtype=object)
		self.A[0] = X
		for i, weights in enumerate(self.W):
			self.S[i] = np.dot(self.A[i], weights)
			self.A[i + 1] = NeuralNetwork.activation(self.S[i])
		return self.A[i + 1]

	def backpropagate(self, Y, P):
		for i in range(1, self.NBR_OF_LAYER):
			if i == 1:
				delta = -(Y - P) * self.activation_derivative(self.S[-i])
			else:
				delta = np.dot(delta, np.transpose(self.W[-i + 1])) * self.activation_derivative(self.S[-i])
			gradient = np.dot(np.transpose(self.A[-i - 1]), delta)
			self.W[-i] -= gradient * self.LEARNING_RATE

	def train(self, X, Y, epsilon):
		errors = []
		while True:
			P = self.feed_forward(X)
			self.backpropagate(Y, P)
			error = np.sum((Y - P) ** 2) / (2 * len(self.A[-1]))
			errors.append(error)
			if error < epsilon:
				return errors
#XOR
X = np.array(([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]), dtype=float)
Y = np.array(([0.0], [1.0], [1.0], [0.0]), dtype=float)
#Score test
#X = np.array(([3 / 24, 5 / 24], [5 / 24, 1 / 24], [10 / 24, 2 / 24], [8 / 24, 3 / 24]), dtype=float)
#Y = np.array(([0.75], [0.82], [0.93], [1.00]), dtype=float)

NN = NeuralNetwork([2, 3, 1])
errors = NN.train(X, Y, 0.001)
Z = NN.feed_forward(X)
print(Z)
print("Rounded")
print(Z.round())

plt.scatter(range(len(errors)), errors, 1)
plt.show()