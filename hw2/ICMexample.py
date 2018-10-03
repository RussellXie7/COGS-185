import dlib
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline
import timeit
import csv
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
onehot_encoder= OneHotEncoder(sparse=False)

Niter = 4                 # The hyper-parameter for icm search

class ThreeClassClassifierProblem:
    def separation_oracle(self, idx, current_solution):
        samp = self.samples[idx]
        psi = [0]*self.num_dimensions
        loss = 0.0
        max_scoring_label = [0]*L # Initialize max_scoring_label for icm search
        for k in range(Niter):
            for iL in range(L):   # Iterate over the window length
                for i in range(K):# Change differnet label for the search of a structured label
                    tmp_label = list(max_scoring_label) # New a list to avoid modifying the max_scoring_label
                    tmp_label[iL] = i # Take turns to modify the structured label from left to right. The guessed structured label.
                    tmp_psi = self.make_psi(samp, tmp_label) # Make a new Psi for the guessed structured label
                    score1 = dlib.dot(current_solution, tmp_psi)

                    #if self.labels[idx] != tmp_label: # Add the conditional "1"
                    #    score1 += 1
                    for j in range(L):
                        if self.labels[idx][j] != tmp_label[j]:
                            score1 += 1

                    if loss < score1: # Search for the maximum and update loss, max_scoring_label, and psi
                        loss = score1
                        max_scoring_label[iL] = i
                        psi = tmp_psi

        return loss, psi

predictions = []
for samp in samples:
    prediction = [0]*L # Initialize max_scoring_label for icm search
    #Niter = 4                 # The hyper-parameter for icm search
    max1 = 0                  # The max value during maximizing our target function
    for k in range(Niter):
        for iL in range(L):   # Iterate over the window length
            for i in range(K):# Change differnet label for the search of a structured label
                tmp_label = list(prediction)
                tmp_label[iL] = i
                psi1 = problem.make_psi(samp, tmp_label)
                score1 = dlib.dot(weights, psi1)

                if max1 < score1:
                    max1 = score1
                    prediction[iL] = i

    predictions.append(prediction)

print("weights", weights)

print labels
print predictions
print "training accuracy=", accuracy_score(predictions, labels)


te_samples = X_test_struct.astype(float).tolist()
te_labels = y_test_struct.astype(float).tolist()

te_predictions = []
for samp in te_samples:
    te_prediction = [0]*L # Initialize max_scoring_label for icm search
    #Niter = 4                 # The hyper-parameter for icm search
    max1 = 0                  # The max value during maximizing our target function
    for k in range(Niter):
        for iL in range(L):   # Iterate over the window length
            for i in range(K):# Change differnet label for the search of a structured label
                tmp_label = list(te_prediction)
                tmp_label[iL] = i
                psi1 = problem.make_psi(samp, tmp_label)
                score1 = dlib.dot(weights, psi1)

                if max1 < score1:
                    max1 = score1
                    te_prediction[iL] = i
    te_predictions.append(te_prediction)

print te_labels
print te_predictions
print "test accuracy=", accuracy_score(te_predictions, te_labels)
