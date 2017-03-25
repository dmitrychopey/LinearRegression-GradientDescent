from numpy import *
import matplotlib.pyplot as plt 


def compute_error(m, b, data):
	error = 0
	for i in range(0, len(data)):
		x = data[i, 0]
		y = data[i, 1]
		error += (y - (x * m + b)) ** 2
	return error / float(len(data))	

def draw_chart(m, b, data):
	x = data[:,0]
	y = data[:,1]
	print(x)
	print(y)
	plt.scatter(x, y)
	plt.plot(x, predict(m, b, x))
	plt.show()

def step_gradient(current_b, current_m, data, learning_rate):
	db = 0
	dm = 0

	N = float(len(data))
	for i in range(0, len(data)):
		x = data[i, 0]
		y = data[i, 1]
		db += -(2/N) * (y - ((current_m * x) + current_b))
		dm += -(2/N) * x * (y - ((current_m * x) + current_b))
	new_b = current_b - (learning_rate * db)
	new_m = current_m - (learning_rate * dm)
	return [new_b, new_m]

def gradient_descent(data, initial_b, initial_m, learning_rate, iterations):
	b = initial_b
	m = initial_m
	for i in range(iterations):
		b, m = step_gradient(b, m, array(data), learning_rate)

	return [b, m]	

def predict(b, m, x):
	y = []
	for i in range(len(x)):
		y.append(x[i] * m + b)
	return y

def run():
	data = genfromtxt("data.csv", delimiter=",")
	learning_rate = 0.0001
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	iterations = 25000 
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, data) ))
	print("Running...")
	[b, m] = gradient_descent(data, initial_b, initial_m, learning_rate, iterations)

	print ("After {0} iterations, b = {1}, m = {2}, error = {3}, with learning_rate = {4}".format(iterations, b, m, compute_error(b, m, data), learning_rate))
	draw_chart(m, b, data)

if __name__ == '__main__':
    run()