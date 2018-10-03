import re
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

def convert_labels(y, option):
	
	newY = [None] * len(y);

	for i in range(len(y)):
		if(y[i] == option):
			newY[i] = +1;
		else:
			newY[i] = -1;
	
	return newY;

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


def read_iris():
	data = load_iris()
	labels = data.target_names;
	x = data.data;
	y = data.target;	

	# makes the y uniform
	for i in range(len(y)):
		y[i] += 1;

	return split_train_test(x, y)

def read_dna(filename, n_examples, n_features):
    F = open(filename)
    labels = np.zeros(n_examples)
    data = np.zeros([n_examples,n_features])
    
    i = 0
    for str_line in F.readlines():
        line0 = map(int, filter(None, re.split('\ |:1', str_line.strip())))
        labels[i] = line0.pop(0)

        for j in line0:
            data[i][j-1] += 1.0
        i += 1
    return labels, data

# only labels for tr_labels are 1, 2, 3
def train_one_vs_all(train_x, train_y, test_x, test_y, my_max_depth):

	train_y_active_1 = convert_labels(train_y, 1);
	train_y_active_2 = convert_labels(train_y, 2);
	train_y_active_3 = convert_labels(train_y, 3);


	clf = RandomForestClassifier(max_depth=my_max_depth, random_state=0)
	clf.fit (train_x, train_y_active_1);

	clf2 = RandomForestClassifier(max_depth=my_max_depth, random_state=0)
	clf2.fit (train_x, train_y_active_2);

	clf3 = RandomForestClassifier(max_depth=my_max_depth, random_state=0)
	clf3.fit (train_x, train_y_active_3);

	prediction_1 = clf.predict_proba(test_x)
	prediction_2 = clf2.predict_proba(test_x)
	prediction_3 = clf3.predict_proba(test_x)


	train_prediction_1 = clf.predict_proba(train_x)
	train_prediction_2 = clf2.predict_proba(train_x)
	train_prediction_3 = clf3.predict_proba(train_x)

	test_error_count = 0;
	train_error_count = 0;

	# test error
	for i in range(len(prediction_1)):
		if prediction_1[i][1] > prediction_2[i][1] and prediction_1[i][1] > prediction_3[i][1]:

			# print (str(prediction_1[i][1]));
			if test_y[i] != 1:
				test_error_count += 1;
		elif prediction_2[i][1] >  prediction_1[i][1] and prediction_2[i][1] > prediction_3[i][1]:
			
			# print (str(prediction_2[i][1]));
			if test_y[i] != 2:
				test_error_count += 1;
		elif prediction_3[i][1] >  prediction_1[i][1] and prediction_3[i][1] > prediction_2[i][1]:

			# print (str(prediction_3[i][1]));
			if test_y[i] != 3:
				test_error_count += 1;
		else:
			print ("----------- Impossible ------------");
	

	# train error
	for i in range(len(train_prediction_1)):
		if train_prediction_1[i][1] > train_prediction_2[i][1] and train_prediction_1[i][1] > train_prediction_3[i][1]:

			# print (str(train_prediction_1[i][1]));
			if train_y[i] != 1:
				train_error_count += 1;
		elif train_prediction_2[i][1] >  train_prediction_1[i][1] and train_prediction_2[i][1] > train_prediction_3[i][1]:
			
			# print (str(train_prediction_2[i][1]));
			if train_y[i] != 2:
				train_error_count += 1;
		elif train_prediction_3[i][1] >  train_prediction_1[i][1] and train_prediction_3[i][1] > train_prediction_2[i][1]:

			# print (str(train_prediction_3[i][1]));
			if train_y[i] != 3:
				train_error_count += 1;
		else:
			print ("----------- Impossible ------------");

	print "Training Error is ", train_error_count / float(len(train_prediction_1))
	print "Test Error is ", test_error_count / float(len(prediction_1))


def train_Explicit_Multiclass(train_x, train_y, test_x, test_y, my_max_depth):

	clf = RandomForestClassifier(max_depth=my_max_depth, random_state=0);
	clf.fit(train_x, train_y);

	train_prediction = clf.predict(train_x)
	test_prediction = clf.predict(test_x)

	train_error_count = 0;
	test_error_count = 1;

	for i in range (len(train_prediction)):
		if train_y[i] != train_prediction[i]:
			train_error_count += 1;

	for i in range (len(test_prediction)):
		if test_y[i] != test_prediction[i]:
			test_error_count += 1;

	print "Training Error is ", train_error_count / float (len(train_prediction))
	print "Test Error is ", test_error_count / float (len(test_prediction))


(trainX, testX, trainY, testY) = read_iris()

tr_labels, tr_data = read_dna("dna.scale.txt",2000,180)
te_labels, te_data = read_dna("dna.scale.t",1186,180)

print;

print "--------- Running Random Forest on iris data One-vs-All --------------";
train_one_vs_all(trainX, trainY, testX, testY, 2);

print;
print;

print "--------- Running Random Forest on DNA data One-vs-All  --------------";
train_one_vs_all(tr_data, tr_labels, te_data, te_labels, 5);

print;
print;

print "--------- Running Random Forest on iris data Explicit Multiclass --------------";
train_Explicit_Multiclass(trainX, trainY, testX, testY, 2);

print;
print;

print "--------- Running Random Forest on DNA data Explicit Multiclass  --------------";
train_Explicit_Multiclass(tr_data, tr_labels, te_data, te_labels, 5);



print;
print;
print;












# print "The first 3 train labels: ", tr_labels[1:3]
# print "The last 3 train labels:  ", tr_labels[-4:-1]
# print "The first 3 train data: ", tr_data[1:3]
# print "The last 3 train data:  ", tr_data[-4:-1]
# print "The first 3 test labels: ", te_labels[1:3]
# print "The last 3 test labels:  ", te_labels[-4:-1]
# print "The first 3 test data: ", te_data[1:3]
# print "The last 3 test data:  ", te_data[-4:-1]










