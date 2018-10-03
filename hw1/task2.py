import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import svm
import random
import math

# define constant
CODE_SETOSA = 0;
CODE_VERSICOLOR = 1;
CODE_VIRGINICA = 2;
epochs = 20000;

def draw_datadots(trainX, trainY):

	for i in range(len(trainX)):
		if(trainY[i] == 0):
			plt.scatter(trainX[i][1], trainX[i][2], color="red");
		elif(trainY[i] == 1):
			plt.scatter(trainX[i][1], trainX[i][2], color="blue");
		elif(trainY[i] == 2):
			plt.scatter(trainX[i][1], trainX[i][2], color="orange");


def draw_bonudaries(ws):
	x_val = np.array(range(4,8))

	# first line
	slope = - (ws[0][1] / ws[0][2]);
	intercept = -(ws[0][0] / ws[0][2]);
	plt.plot(x_val, slope * x_val + intercept, color = "red");

	# 2nd line
	slope = - (ws[1][1] / ws[1][2]);
	intercept = -(ws[1][0] / ws[1][2]);
	plt.plot(x_val, slope * x_val + intercept, color = "blue");

	# 3rd line
	slope = - (ws[2][1] / ws[2][2]);
	intercept = -(ws[2][0] / ws[2][2]);
	plt.plot(x_val, slope * x_val + intercept, color = "orange");

def split_train_test(x,y):
	
	trainX = [];
	testX = [];
	trainY = [];
	testY = [];

	for index in range (len(y)):
		if (0 <= index < 40) or (50 <= index < 90) or (100 <= index < 140):
			trainX.append(x[index])
			trainY.append(y[index])
		else:
			testX.append(x[index])
			testY.append(y[index])

	return (np.array(trainX), np.array(testX), np.array(trainY), np.array(testY));


def draw_result(trainX, trainY, ws):

	plt.grid();
	plt.title('Q2: Training data along with decision boundaries');
	plt.xlabel('x1');
	plt.ylabel('x2');
	draw_datadots(trainX, trainY);
	draw_bonudaries(ws);
	plt.show()

def compute_gradient(input_x, output_y, j_value, ws, lamda):
	
	gradient = 0
	for i in range(len(output_y)):
		target = output_y[i];

		# if the current target is the same as the w that we are updating
		if (target == j_value):

			# need to go through each ws
			for k in range(3):

				# I am taking the gradient of the j_value, here the target is the same as j_value
				# if the the k of the current ws[k] is the not same as the target, we compute for gradient
				if (k != target):
					if ((np.inner(ws[target], input_x[i]) - np.inner(ws[k], input_x[i])) < 1):
						gradient +=  (0 - input_x[i]);
					else:
						gradient += 0;
		else:
			
			# Note that target != j_value
			for k in range(3):

				# I am taking the gradient of the j_value, here the target is Different From j_value
				# if the the k of the current ws[k] is the not same as the target, we compute for gradient
				if (k != target):
					# if the current k is the same as the j_value, it means that we want to preserve this
					if ( (k == j_value) and ((np.inner(ws[target], input_x[i]) - np.inner(ws[k], input_x[i])) < 1) ):
						gradient +=  (input_x[i]);
				# In all other cases, we do not update gradient


	gradient *= lamda;			
	gradient += (ws[j_value]);

	return gradient;

def SVM(input_x, output_y, lamda, learning_rate):

	class_num = 3;
	ws = np.array([np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0]))]);
	ws_new = a = np.empty_like (ws);

	for index in range(epochs):

		ws[:] = ws_new
		gradients = np.array([np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0]))]);

		# calculate gradient for w0, w1, and w2, since there are three weights
		for j in range (class_num):
			# the current ws being updated is j
			gradients[j] = compute_gradient(input_x, output_y, j, ws, lamda);

		
		# update gradient here
		for j in range (class_num):
			ws_new[j] = ws[j] - learning_rate * gradients[j];


		if index % 10000 == 0:
			print ("w values difference is " + str([np.linalg.norm(ws_new[0] - ws[0], ord = 1), np.linalg.norm(ws_new[1] - ws[1], ord = 1),np.linalg.norm(ws_new[2] - ws[2], ord = 1)]))

		if np.linalg.norm(ws_new.flatten() - ws.flatten(), ord = 1) < 0.00001:
			print("gradient descent terminated after " + str(index) + " iterations");
			break;

	# if something is satisfied, break

	return ws_new;


def Test_SVM(ws, testX, testY):

	error_count = 0;

	for index in range(len(testX)):

		value0 = np.inner(ws[0], testX[index]);
		value1 = np.inner(ws[1], testX[index]);
		value2 = np.inner(ws[2], testX[index]);

		predicted_label = 0;

		if (value0 >= value1) and (value0 >= value2):
			predicted_label = 0
			# print ("loop " + str(index) + " predict label is " + str(predicted_label))
		elif (value1 >= value0) and (value1 >= value2):
			predicted_label = 1;
			# print ("loop " + str(index) + " predict label is " + str(predicted_label))
		elif (value2 >= value0) and (value1 >= value0):
			predicted_label = 2;
			# print ("loop " + str(index) + " predict label is " + str(predicted_label))

		if predicted_label != testY[index]:
			error_count += 1;

	return float(error_count) / len(testX);


def main (lamda, learning_rate):
	final_train_error = 1;
	final_test_error = 1;
	final_ws = np.array([np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0]))]);
	final_lamda = 0;
	(trainX, testX, trainY, testY) = split_train_test(x, y)

#############################################################
	lamda = 0.5;
	ws = SVM(trainX, trainY, lamda, learning_rate);
	train_error = Test_SVM (ws, trainX, trainY);
	test_error = Test_SVM (ws, testX, testY);

	if (train_error < final_train_error):
		final_ws[:] = ws;
		final_train_error = train_error;
		final_test_error = test_error;
		final_lamda = lamda;

	print ("--------------- when lamda is 0.5 ---------------")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
	print ("------------------------------------------")
#############################################################
	lamda = 2.0;
	ws = SVM(trainX, trainY, lamda, learning_rate);
	train_error = Test_SVM (ws, trainX, trainY);
	test_error = Test_SVM (ws, testX, testY);

	if (train_error < final_train_error):
		final_ws[:] = ws;
		final_train_error = train_error;
		final_test_error = test_error;
		final_lamda = lamda;

	print ("--------------- when lamda is 2.0 ---------------")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
	print ("------------------------------------------")
#############################################################
	lamda = 5.0;
	ws = SVM(trainX, trainY, lamda, learning_rate);
	train_error = Test_SVM (ws, trainX, trainY);
	test_error = Test_SVM (ws, testX, testY);

	if (train_error < final_train_error):
		final_ws[:] = ws;
		final_train_error = train_error;
		final_test_error = test_error;
		final_lamda = lamda;

	print ("--------------- when lamda is 5.0 ---------------")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
	print ("------------------------------------------")
#############################################################
	lamda = 10.0;
	ws = SVM(trainX, trainY, lamda, learning_rate);
	train_error = Test_SVM (ws, trainX, trainY);
	test_error = Test_SVM (ws, testX, testY);

	if (train_error < final_train_error):
		final_ws[:] = ws;
		final_train_error = train_error;
		final_test_error = test_error;
		final_lamda = lamda;

	print ("--------------- when lamda is 10.0 ---------------")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
	print ("------------------------------------------")
#############################################################
	print;
	print ("-----------------------END-----------------------------")
	print ("The final ws0 is " + str(final_ws[0]));
	print ("The final ws1 is " + str(final_ws[1]));
	print ("The final ws2 is " + str(final_ws[2]));
	print ("the FINAL train_error is " + str(final_train_error));
	print ("the FINAL test_error is " + str(final_test_error));
	print ("the FINAL lamda is " + str(final_lamda));

	draw_result(trainX, trainY, final_ws);

	print;
	print;
	print;


if __name__ == '__main__':

	print "start loading....";
	
	data = load_iris();

	labels = data.target_names;
	x = data.data;
	y = data.target;

	# use the first w as the bias term
	x = np.hstack((np.ones((len(x),1)), x));

	print labels;
	
	main(0.5, 0.000005);
