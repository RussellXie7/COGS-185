import numpy as np
from sklearn.svm import LinearSVC
from pystruct.datasets import load_letters
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)

print X.shape
print y.shape

def convert_letter_to_dict(letter):
    d = {}
    for ind, i in enumerate(letter):
        d["p_"+str(ind)] = i
    return d

def word2features(word, i):
    letter = word[i]
    features = {
#         'bias':1,
    'letter': convert_letter_to_dict(letter)
    }
    if i > 0:
        letter = word[i-1]
        features.update({
        '-1:letter': str(convert_letter_to_dict(letter))
        })
    if i < len(word)-1:
        letter = word[i+1]
        features.update({
        '+1:letter': str(convert_letter_to_dict(letter))
        })
    return features


def create_word_features(data):
    return [word2features(data, i) for i in range(len(data))]

print "-------- Trainig Started ----------"
X_features = [create_word_features(word) for word in X]
X_features = np.array(X_features)

import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=True)

X_train, X_test = X_features[folds == 1], X_features[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]
y_tr = []
for y_i in y_train:
    z = []
    for i in y_i:
        z.append(str(i))
    y_tr.append(z)

y_te = []
for y_i in y_test:
    z = []
    for i in y_i:
        z.append(str(i))
    y_te.append(z)
cnt=0
a=0
for xseq, yseq in zip(X_train, y_tr):
#     print(xseq)
#     print(yseq)
    cnt +=1
    ystr = [str(i) for i in yseq]
    if(len(ystr)!=len(xseq)):
       print(cnt)
       continue
    a+=1
    trainer.append(xseq, ystr)
print(a)
print(len(y_train))
trainer.set_params({
    'c1': 0.10,
    'c2': 1e-3,
    'max_iterations': 100,
    'feature.possible_transitions': True
})
trainer.train('ocr.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('ocr.crfsuite')

def calc_acc(data, y):
    tot=0.0
    cor=0.0
    for i,d in enumerate(data):
        prediction = tagger.tag(d)
        cor += np.sum(np.array(y)[i]==np.array(prediction))
        tot += len(y[i])
    return (cor/tot)*100.0

print "-------- Getting Result ----------"
print("Train acc:", calc_acc(X_train, y_tr))
print("Test acc:", calc_acc(X_test, y_te))
