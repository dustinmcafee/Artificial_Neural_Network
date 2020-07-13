#!/usr/bin/python3	
from sklearn import preprocessing               # For standardizing the dataset
import sys
import random
import numpy as np
import math
import time
 
# Global parameters
# -----> Adjust Architecture Here <-----
inputDimension = 0
neuron_per_layer = 3
numLayers = 3	# <-- Num hidden layers + 1
l_rate = 0.1
tol = 0.001
n_epoch = 1000
n_k_folds = 5

# Other Globals used by system: DO NOT TOUCH
W = [None]*(numLayers + 1)
z2 = [None]*numLayers
delta = [None]*(numLayers + 1)
d_W = [None]*(numLayers + 1)

# Split data into Training and Testing sets
# Input: dataset: dataset to split
#	 split: how many elements to give to the testing dataste
# Return training dataset, testing dataset
def dataSplit(dataset, split):
	#Copy the data as to not randomize the original set
	data = dataset.copy()
	np.random.shuffle(data)

	#Split the randomized dataset into a training dataset and a testing dataset
	valid, train = data[:split,:], data[split:,:]
	return train, valid

# Train ANN for a number of epochs
def train_network(train):
	global inputDimension
	global n_epoch
	print("    Training Network")
	try:
		y = np.array(train, dtype=float)[:,-1].copy().reshape(np.size(np.array(train)[:,-1]), 1)
		X = np.delete(train, -1, 1)
		inputDimension = np.size(X, 1)
		init_weights_NN()
		for j in range(n_epoch):
			if((j+1) % 10 == 0 or j == 0):
				print("      Epoch:", j + 1)
			i = 0
			sum_err_sqr = 0
			outputs = []
			for row in X:
				output = forward(row)[0]
				outputs.append(float(output))
				backward(row, y[i], output)

				sum_err_sqr += (output - y[i])**2
				i += 1

			if((j+1) % 10 == 0 or j == 0):
				print("        Sum of Errors Squared:", sum_err_sqr)
			if(sum_err_sqr < tol):
				break

	except KeyboardInterrupt:
		pass
	return sum_err_sqr

# Helper function for getPerformanceMetrics.
# Computes number of correct/incorrect predictions,
# as well as true positives/negatives, and false positives/negatives
# Input: validData: the valid list of categories
#	 predictions: the predicted list of categories
# Return: number of correct, incorrect, true positives,
#	  true negatives, false positives, and false negatives
def getPerformanceStatistics(validData, predictions):
	correct = 0
	incorrect = 0
	fNeg = 0
	fPos = 0
	tPos = 0
	tNeg = 0
	for x in range(len(validData)):
		if(validData[x] == predictions[x]):
			correct += 1
			if(predictions[x] == 0):			#True negative
				tNeg +=1
			else:						#True positive
				tPos +=1
		else:
			incorrect += 1
			# 0 == non-spam (negative), 1 == spam (positive)
			if(predictions[x] == 0):			#False negative
				fNeg +=1
			else:						#False positive
				fPos +=1
	return correct, incorrect, tPos, tNeg, fPos, fNeg

#Report performance metrics from the validation dataset and list of predictions
#Returns tuple
def getPerformanceMetrics(validData, predictions):
	P, N, TP, TN, FP, FN = getPerformanceStatistics(validData, predictions)
	accuracy = (P / (P + N)) * 100.0
	if((TP + FN) == 0):
		TPR = np.inf
	else:
		TPR = TP / (TP + FN) * 100.0	#True positive rate; recall; sensitivity
	if((TP + FP) == 0):
		PPV = np.inf
	else:
		PPV = TP / (TP + FP) * 100.0	#Positive predictive value; precision
	if((TN + FP) == 0):
		TNR = np.inf
	else:
		TNR = TN / (TN + FP) * 100.0	#True negative rate; specificity
	if((PPV + TPR) == 0):
		F = np.inf
	else:
		F = 2 * PPV * TPR / (PPV + TPR)	#F1 Score
	return (P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F)

# Split a training dataset into k folds for use with K-Fold Cross-Validation
# Input: train_data: the training data to split into training data/validation data pairs
#	 k: the number of 'folds' of data (Number of partitions)
# Return: 2 Dimensional Array of k partitions of train_data
def k_fold_split(train_data):
	global n_k_folds
	size = int(len(train_data) / n_k_folds)
	# Copy the data as to not randomize the original training dataset
	data_copy = train_data.copy()
	split_data = []

	#From 0 to k-1
	for i in range(n_k_folds):
		fold = []
		#(Slow but works) Partition data into k folds
		while len(fold) < size:
			#Randomly partitions the data
			index = random.randrange(len(data_copy))
			#Pop and Append
			fold.append(data_copy.pop(index))
		#Return 2-Dimensional Array
		split_data.append(fold)
	return split_data

# Evaluate an algorithm using a cross validation split
def cross_validate_back_prop(training_dataset):
	# Split training dataset for k-fold cross-validation (Create validation/training pairs)
	global n_k_folds
	folds = k_fold_split(training_dataset)
	scores = []
	i = 0
	errors = []
	print("Performing K-Fold Validation with K:", n_k_folds, "Folds", int(len(training_dataset) / n_k_folds), "size each")
	for fold in folds:
		i += 1
		print("  Fold:", i)
		train = list(folds)
		train.remove(fold)
		train = sum(train, [])
		valid = []
		for row in fold:
			row_copy = list(row)
			valid.append(row_copy)
			row_copy[-1] = None
		predicted, err = back_prop(train, valid)
		errors.append(err)
		actual = [row[-1] for row in fold]

		accuracy = getPerformanceMetrics(actual, predicted)
		scores.append(accuracy)
	mean_err = np.mean(np.array(errors))
	return scores, mean_err
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_prop(train, test):
	#n_inputs = len(train[0]) - 1
	#n_categories = len(set([row[-1] for row in train]))
	err = train_network(train)
	out_predict = []

	X = np.delete(test, -1, 1)
	for row in X:
		out_predict.append(forward(row)[0])
	out_predict = np.round(np.array(out_predict))

	return out_predict, err


# Normalize the category column of the data.
#	Subtracts 1 from the last column until
#	the minimum of the column is 0. This makes it
#	easier to use the categories as indices for arrays
# Input: data: the data matrix
def normalize_categories(data):
	while(np.amin(data[:,-1]) > 0):
		data[:,-1] = data[:,-1] - 1

# Z-Normalize the columns of the data, except the last column
# Input: data: the data matrix
def normalize_dataset(data):
	cat = data[:,-1].copy().reshape(np.size(data, 0), 1)
	data = np.delete(data, -1, 1)
	data = preprocessing.normalize(data)
	data = np.hstack((data, cat))
	return data

def init_weights_NN():
	global W
	for i in range(numLayers + 1):
		if(i == 0):
			W[i] = None	# <-- We follow closely to the algorithm given in class: There is no W[0]
		elif(i == 1):
			W[i] = np.array([[np.random.uniform(-0.01, 0.01) for i in range(inputDimension + 1)] for i in range(neuron_per_layer)]).T
		elif(i == numLayers + 1):
			W[i] = np.array([[np.random.uniform(-0.01, 0.01) for i in range(neuron_per_layer + 1)] for i in range(1)]).T
		else:
			W[i] = np.array([[np.random.uniform(-0.01, 0.01) for i in range(neuron_per_layer + 1)] for i in range(neuron_per_layer)]).T

def forward(X):
	global W
	global z2
	z = np.array([])
	try:
		z2[0] = np.insert(X, 0, 1).copy().reshape(np.size(X, 0) + 1, np.size(X, 1))
	except:
		z2[0] = np.insert(X, 0, 1).copy().reshape(np.size(X, 0) + 1, 1)
	for i in range(1, numLayers):
		try:
			z = np.matmul(z2[i-1].astype(float).T, W[i].astype(float))
		except:
			z = np.matmul(z2[i-1].astype(float), W[i].astype(float))
		z2[i] = sigmoid(z)
		z2[i] = np.insert(z2[i], 0, 1)
	y = np.matmul(W[-1].T, z2[-1])

	return y

# Activation Function
def sigmoid(s):
	return 1/(1+np.exp(-s))

# Derivative of Sigmoid Function
def sigmoidPrime(s):
	return s * (1 - s)

# Backward Propgate (Updates weights online training)
def backward(X, y, o):
	global d_W
	global delta
	global W
	global z2
	global l_rate
	try:
		inputs = X.copy().reshape(np.size(X, 0), np.size(X, 1))
	except:
		inputs = X.copy().reshape(np.size(X, 0), 1)
	for i in reversed(range(1, numLayers + 1)):
		if(i == numLayers):
			delta[i] = (y - o)[0]
			try:
				d_W[i] = float(l_rate*delta[i]) * z2[i-1]
			except:
				d_W[i] = np.matmul(l_rate*delta[i], z2[i-1].T)
		else:
			try:
				try:
					delta[i] = np.matmul(np.matmul(W[i+1].T, delta[i+1]), sigmoidPrime(z2[i]))
				except:
					delta[i] = np.matmul(np.matmul(W[i+1], delta[i+1]).T, sigmoidPrime(z2[i]))
			except:
				try:
					delta[i] = np.matmul(np.matmul(W[i+1], delta[i+1].T), sigmoidPrime(z2[i]))
				except:	#Delta is a floating point, not a vector
					delta[i] = np.matmul(W[i+1].T * float(delta[i+1]), sigmoidPrime(z2[i]))
			z2[i-1] = z2[i-1].reshape(np.size(z2[i-1], 0), 1)
			d_W[i] = np.matmul(l_rate*np.array(delta[i]).reshape(len(delta[i]), 1), z2[i-1].T).T

	for i in range(1, numLayers + 1):
		# delta_Wi = learning_rate * (ri - yi) * z
		# Wi = Wi + delta_Wi
		d_W[i] = np.array(d_W[i])
		try:
			d_W[i] = d_W[i].reshape(np.size(d_W[i], 0), np.size(d_W[i], 1))
		except:
			d_W[i] = d_W[i].reshape(np.size(d_W[i], 0), 1)
		W[i] = W[i] + d_W[i]

def main():
	random.seed(1)
	global n_k_folds
	global l_rate
	global neuron_per_layer
	global n_epoch

	# load and prepare data
	dataset = np.genfromtxt('input/spambase.data', delimiter=',')

	# Load or generate training/testing data
	if(len(sys.argv) == 2):
		if(int(sys.argv[1]) == 1):
			trainData, testData = dataSplit(dataset, 600)
			np.savetxt("input/train/TrainingData.txt", np.matrix(trainData), delimiter=',', fmt='%3.4f')
			np.savetxt("input/test/TestingData.txt", np.matrix(testData), delimiter=',', fmt='%3.4f')
		else:
			trainData = np.genfromtxt('input/train/TrainingData.txt', delimiter=',')
			testData = np.genfromtxt('input/test/TestingData.txt', delimiter=',')
	else:
		trainData = np.genfromtxt('input/train/TrainingData.txt', delimiter=',')
		testData = np.genfromtxt('input/test/TestingData.txt', delimiter=',')

	# Print Dataset sizes
	print(np.size(trainData, 1), "Dimensions")
	print(np.size(trainData, 0), "Triaining Set Observations")
	print(np.size(testData, 0), "Test Set Observations")
	normalize_categories(dataset)
	normalize_categories(trainData)
	normalize_categories(testData)
	dataset = normalize_dataset(dataset)
	trainData = normalize_dataset(trainData)
	testData = normalize_dataset(testData)

	# Log information
	print("Performing Back Propogation on", n_epoch, "epochs")
	print("K-Fold Cross-Validation with", n_k_folds, "Folds")
	print("Learning Rate:", l_rate)
	print("Number of neurons per Hidden Layer:", neuron_per_layer)
	print("Number of Hidden Layers:", numLayers - 1)

	# Perform K-fold cross validation on back propagation
	metrics, mean_err = cross_validate_back_prop(dataset.tolist())
	elem = np.mean(np.array(metrics), 0)
	P = elem[0]
	N = elem[1]
	TP = elem[2]
	TN = elem[3]
	FP = elem[4]
	FN = elem[5]
	accuracy = elem[6]
	TPR = elem[7]
	PPV = elem[8]
	TNR = elem[9]
	F = elem[10]
	print('Average Performance Metrics:')
	print('Average K-Fold Output SSQ Errors:', mean_err)
	print('Accuracy: ' + "{0:.2f}".format(round(accuracy,2)) + '%')
	print('True Positives: ' + str(TP))
	print('True Negatives: ' + str(TN))
	print('False Positives: ' + str(FP))
	print('False Negatives: ' + str(FN))
	print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(TPR,2)) + '%')
	print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(PPV,2)) + '%')
	print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(TNR,2)) + '%')
	print('F1 Score: ' + "{0:.2f}".format(round(F,2)) + '%')
	print('\n\n')

	#Perform on Test Data
	predictions, err = back_prop(trainData, testData)
	P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F = getPerformanceMetrics(testData[:,-1], predictions)
	print('Testing Set Performance Metrics:')
	print('Output SSQ Errors:', err)
	print('Accuracy: ' + "{0:.2f}".format(round(accuracy,2)) + '%')
	print('True Positives: ' + str(TP))
	print('True Negatives: ' + str(TN))
	print('False Positives: ' + str(FP))
	print('False Negatives: ' + str(FN))
	print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(TPR,2)) + '%')
	print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(PPV,2)) + '%')
	print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(TNR,2)) + '%')
	print('F1 Score: ' + "{0:.2f}".format(round(F,2)) + '%')

	#Save Predictions of Test Data File
	np.savetxt("output/ANN_Predictions.txt", predictions, delimiter=',', fmt='%d')

main()
