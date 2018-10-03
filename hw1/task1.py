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
epochs = 50000;

def draw_datadots(trainX, trainY):

	for i in range(len(trainX)):
		if(trainY[i] == 0):
			plt.scatter(trainX[i][1], trainX[i][2], color="red");
		elif(trainY[i] == 1):
			plt.scatter(trainX[i][1], trainX[i][2], color="blue");
		elif(trainY[i] == 2):
			plt.scatter(trainX[i][1], trainX[i][2], color="orange");


def draw_bonudaries(ws0, ws1, ws2):
	x_val = np.array(range(4,8))

	# first line
	slope = - (ws0[1] / ws0[2])
	intercept = -(ws0[0] / ws0[2]);
	plt.plot(x_val, slope * x_val + intercept, color = "red");

	# 2nd line
	slope = - (ws1[1] / ws1[2])
	intercept = -(ws1[0] / ws1[2]);
	plt.plot(x_val, slope * x_val + intercept, color = "blue");

	# 3rd line
	slope = - (ws2[1] / ws2[2])
	intercept = -(ws2[0] / ws2[2]);
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

'''
This function is used to preprocess targets for one vs. all operation
'''
def convert_labels(y, option):
	
	newY = [None] * len(y);

	for i in range(len(y)):
		if(y[i] == option):
			newY[i] = +1;
		else:
			newY[i] = -1;
	
	return newY;

'''
This function performs the SVM operation
'''
def SVM(input_x, output_y, lamda, learning_rate):

	ws = np.zeros(len(x[0]))
	ws_new = ws;

	gradient = 0;

	for index in range(epochs):

		# ws_new gives the old value to ws
		ws = ws_new;
		gradient = 0;

		for i in range(len(output_y)):
			if (1 - output_y[i] * np.inner(ws,input_x[i])) > 0:
				gradient += 0 - lamda * output_y[i] * input_x[i];
			else:
				gradient += 0;

		gradient += ws;

		ws_new = ws - learning_rate * gradient;

		if index % 10000 == 0:
			# print ("gradient is " + str(gradient));
			print ("w values difference is " + str(np.linalg.norm(ws_new - ws, ord = 1)));


		if np.linalg.norm(ws_new - ws, ord = 1) < 0.00001:
			print("gradient descent terminated after " + str(index) + " iterations");
			break;

	# print ("The final ws is " + str(ws_new));
	return ws_new;


def Test_SVM(ws0, ws1, ws2, testX, testY):

	error_count = 0;

	# compute the value given from three model
	for index in range(len(testX)):

		value0 = np.inner(ws0, testX[index]);
		value1 = np.inner(ws1, testX[index]);
		value2 = np.inner(ws2, testX[index]);

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

def draw_result(trainX, trainY, ws0, ws1, ws2):

	plt.grid();
	plt.title('Training data along with decision boundaries');
	plt.xlabel('x1');
	plt.ylabel('x2');
	draw_datadots(trainX, trainY);
	draw_bonudaries(ws0 ,ws1, ws2);
	plt.show()

def main(lamda, learning_rate):
	final_train_error = 1;
	final_test_error = 1;
	ws0 = np.zeros(len(x[0]))
	ws1 = np.zeros(len(x[0]))
	ws2 = np.zeros(len(x[0]))
	final_lamda = 999;

#######################################################################
	print;
	lamda = 0.5
	corrected_y = convert_labels(y, CODE_SETOSA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	a_ws0 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VERSICOLOR)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	a_ws1 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VIRGINICA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	a_ws2 = SVM(trainX, trainY, lamda, learning_rate)

	(trainX, testX, trainY, testY) = split_train_test(x, y)
	train_error = Test_SVM(a_ws0, a_ws1, a_ws2, trainX, trainY) 
	test_error = Test_SVM(a_ws0, a_ws1, a_ws2, testX, testY)

	if (train_error < final_train_error):
		ws0 = a_ws0
		ws1 = a_ws1
		ws2 = a_ws2
		final_train_error = train_error
		fina_test_error = test_error
		final_lamda = lamda;

	print ("when lamda is 0.5...")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
#######################################################################
	print;
	lamda = 2.0
	corrected_y = convert_labels(y, CODE_SETOSA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	b_ws0 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VERSICOLOR)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	b_ws1 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VIRGINICA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	b_ws2 = SVM(trainX, trainY, lamda, learning_rate)

	(trainX, testX, trainY, testY) = split_train_test(x, y)
	train_error = Test_SVM(b_ws0, b_ws1, b_ws2, trainX, trainY) 
	test_error = Test_SVM(b_ws0, b_ws1, b_ws2, testX, testY)

	if (train_error < final_train_error):
		ws0 = b_ws0
		ws1 = b_ws1
		ws2 = b_ws2
		final_train_error = train_error
		fina_test_error = test_error
		final_lamda = lamda;
		
	print ("when lamda is 2.0...")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
#######################################################################
	print;
	lamda = 5.0
	corrected_y = convert_labels(y, CODE_SETOSA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	c_ws0 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VERSICOLOR)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	c_ws1 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VIRGINICA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	c_ws2 = SVM(trainX, trainY, lamda, learning_rate)

	(trainX, testX, trainY, testY) = split_train_test(x, y)
	train_error = Test_SVM(c_ws0, c_ws1, c_ws2, trainX, trainY) 
	test_error = Test_SVM(c_ws0, c_ws1, c_ws2, testX, testY)

	if (train_error < final_train_error):
		ws0 = c_ws0
		ws1 = c_ws1
		ws2 = c_ws2
		final_train_error = train_error
		fina_test_error = test_error
		final_lamda = lamda;
		
	print ("when lamda is 5.0...")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
#######################################################################
	print;
	lamda = 10.0
	corrected_y = convert_labels(y, CODE_SETOSA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	d_ws0 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VERSICOLOR)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	d_ws1 = SVM(trainX, trainY, lamda, learning_rate)

	corrected_y = convert_labels(y, CODE_VIRGINICA)
	(trainX, testX, trainY, testY) = split_train_test(x, corrected_y)
	d_ws2 = SVM(trainX, trainY, lamda, learning_rate)

	(trainX, testX, trainY, testY) = split_train_test(x, y)
	train_error = Test_SVM(d_ws0, d_ws1, d_ws2, trainX, trainY) 
	test_error = Test_SVM(d_ws0, d_ws1, d_ws2, testX, testY)

	if (train_error < final_train_error):
		ws0 = d_ws0
		ws1 = d_ws1
		ws2 = d_ws2
		final_train_error = train_error
		fina_test_error = test_error
		final_lamda = lamda;

	print ("when lamda is 10.0...")
	print ("train_error is " + str(train_error));
	print ("test_error is " + str(test_error));
#######################################################################
	print;
	print ("-----------------------END-----------------------------")
	print ("The final ws0 is " + str(ws0));
	print ("The final ws1 is " + str(ws1));
	print ("The final ws2 is " + str(ws2));
	print ("the FINAL train_error is " + str(final_train_error));
	print ("the FINAL test_error is " + str(final_test_error));
	print ("the FINAL lamda is " + str(final_lamda));

	draw_result(trainX, trainY, ws0, ws1, ws2)

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
