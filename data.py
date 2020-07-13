#!/usr/bin/python3
"""
Name: Dustin Mcafee
Class: COSC 528 - Introduction to Machine Learning
Assignment: Project 3 - Project data to Principal Components
"""

import numpy as np
from numpy import genfromtxt
import sys

def main():
	PC = int(sys.argv[1])
	if(PC < 1):
		print("First input argument must be the number of Principal Components to project data onto")
		return

	# Data has already been preprocessed (no missing values or ID columns or headers)
	# Data is assumed to already be standardized in some fashion
	#my_data = genfromtxt('input/test/TestingData.txt', delimiter=',')
	my_data = genfromtxt('input/train/TrainingData.txt', delimiter=',')

	# Last column is predicted value, delete it
	labels = my_data[:,-1].copy()
	my_data = np.delete(my_data, -1, 1)

	#Variables
	N = np.size(my_data, 0)
	labels = labels.reshape(N,1)
	dimensions = np.size(my_data, 1)
	print(dimensions, "Dimensions")
	print(N, "Observations")

	#Singular Value Decomposition
	u, s, vh = np.linalg.svd(my_data, full_matrices=True)
	ss = np.square(s)

	#Percentage of Variance for each k PC
	ss_sum = np.sum(ss)
	ss_percent = np.divide(ss,ss_sum) * 100
	sum = 0
	for i in range(0, PC):
		sum = sum + ss_percent[i]
		
	print("First", PC, "PC covers", sum, "percent of the Variance")

	#Project to first PCs
	v = np.transpose(vh)
	rang = range(0,PC)
	v_pc = v[:, rang]
	data_pc = np.matmul(my_data, v_pc)

	#Save output Files
	np.savetxt("output/Singular_Values.txt", s, delimiter=',', fmt='%3.4f')
	np.savetxt("output/Singular_Values_Percent_Variance.txt", ss_percent, delimiter=',', fmt='%3.4f')
	np.savetxt("input/train/TrainingData_Project.txt", np.hstack((data_pc, labels)), delimiter=',', fmt='%3.4f')
	#np.savetxt("input/test/TestingData_Project.txt", np.hstack((data_pc, labels)), delimiter=',', fmt='%3.4f')

main()
