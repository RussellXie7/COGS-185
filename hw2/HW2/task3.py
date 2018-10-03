import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import dlib

window_size = 2

def main():

    # for simple testing
    # I recommend try training_sample['feature'][:30] and testing_sample['feature'][:30]
    # and training_sample['labels'][:30] and testing_sample['labels'][:30]
    # should give a high training accuracy and a low testing accuracy

    # need to read 4001 + 1001 = 5002
    dataset1 = read_OCR('OCRdataset/letter.data', 5002, 128)

    (training_sample,testing_sample) = recompose_data(dataset1);

    print "total training samples: ", len(training_sample['words'])

    print "total testing samples: ",len(testing_sample['words'])

    # problem = MultiClassClassifierProblem(training_sample['feature'][:30],training_sample['labels'][:30])
    problem = MultiClassClassifierProblem(training_sample['feature'],training_sample['labels'])
    weights = dlib.solve_structural_svm_problem(problem)

    # get training accuracy
    predictions = []

    # for samp in training_sample['feature'][:30]:
    for samp in training_sample['feature']:
        prediction = [0] * window_size

        Nither = 4
        max1 = 0

        for k in range(Nither):
            for iL in range(window_size):
                for i in range(27):
                    temp_label = list(prediction)
                    temp_label[iL] = i
                    psi1= problem.make_psi(samp, temp_label)
                    score1 = dlib.dot(weights,psi1)

                    if max1 < score1:
                        max1 = score1
                        prediction[iL] = i

        predictions.append(prediction)


    # print("weights", weights)
    print predictions
    # print "training accuracy=", accuracy_score(predictions, training_sample['labels'][:30])
    print "training accuracy=", accuracy_score(predictions, training_sample['labels'])

    # get testing accuracy
    te_predictions = []
    # for samp in testing_sample['feature'][:30]:
    for samp in testing_sample['feature']:
        te_prediction = [0] * window_size
        Nither = 4
        max1 = 0

        for k in range(Nither):
            for iL in range(window_size):
                for i in range(27):
                    temp_label = list(te_prediction)
                    temp_label[iL] = i
                    psi1 = problem.make_psi(samp, temp_label)
                    score1 = dlib.dot(weights, psi1)

                    if max1 < score1:
                        max1 = score1
                        te_prediction[iL] = i
        te_predictions.append(te_prediction)


    # print te_labels
    # print te_predictions
    # print "test accuracy=", accuracy_score(te_predictions, testing_sample['labels'][:30])
    print "test accuracy=", accuracy_score(te_predictions, testing_sample['labels'])


def accuracy_score(predictions, labels):
    if len(predictions) != len(labels):
        print "---------- Weird -------------"

    error_count = 0;
    total = len(predictions) * 2

    for i in range (len(predictions)):
        if predictions[i][0] != labels[i][0]:
            error_count = error_count + 1

        if predictions[i][1] != labels[i][1]:
            error_count = error_count + 1

    return ( 1 - float(error_count)/float(total))

def atoi(a):
    return int(ord(a)-ord('a'))
def itoa(i):
    return chr(i+ord('a'))

def iors(s):
    try:
        return int(s)
    except ValueError: # if it is a string, return a string
        return s


def recompose_data(in_dataset):

    training_set = {}
    training_set['words'] = ["" for x in range(4000)]
    training_set['labels'] = []
    training_set['feature'] = []
    training_set['change_labels'] = []

    testing_set = {}
    testing_set['words'] = ["" for x in range(1000)]
    testing_set['labels'] = []
    testing_set['feature'] = []
    testing_set['change_labels'] = []



    id = in_dataset['ids'][0]
    next_id = in_dataset['next_ids'][0]

    # 1 stands for true
    prev_null = 0

    train_index = 0;
    test_index = 0;

    # 5002 letters intotal, getting training here
    for i in range(1, 4001):

        id = in_dataset['ids'][next_id-1]
        next_id = in_dataset['next_ids'][next_id-1]

        if next_id == -1:
            # now we reach the end of this word
            prev_null = 1
            next_id = id+1

            training_set['words'][train_index] = itoa(in_dataset['labels'][id - 2]) + itoa(in_dataset['labels'][id - 1])
            training_set['feature'].append([])
            training_set['feature'][-1].append(in_dataset['features'][id-2])
            training_set['feature'][-1].append(in_dataset['features'][id-1])

            if in_dataset['labels'][id - 2] != in_dataset['labels'][id - 1]:
                training_set['change_labels'].append(1)
            else:
                training_set['change_labels'].append(0)

            training_set['labels'].append([in_dataset['labels'][id - 2], in_dataset['labels'][id - 1]])

            train_index  = train_index + 1
            if (train_index >= 4000):
                break

            # note that label for _ is 26
            training_set['words'][train_index] = itoa(in_dataset['labels'][id - 1]) + "_"

            training_set['feature'].append([])
            training_set['feature'][-1].append(in_dataset['features'][id-1])
            training_set['feature'][-1].append(np.zeros(128))
            training_set['change_labels'].append(1)
            training_set['labels'].append([in_dataset['labels'][id - 1], 26])

            train_index  = train_index + 1
            if (train_index >= 4000):
                break

            continue

        # if the prev one is 1, then prev label is _
        if prev_null == 1:
            training_set['words'][train_index] = "_" + itoa(in_dataset['labels'][id - 1])

            training_set['feature'].append([])
            training_set['feature'][-1].append(np.zeros(128))
            training_set['feature'][-1].append(in_dataset['features'][id-1])
            training_set['change_labels'].append(1)
            training_set['labels'].append([26,in_dataset['labels'][id - 1]])

            prev_null = 0
        else:
            training_set['words'][train_index] = itoa(in_dataset['labels'][id - 2]) + itoa(in_dataset['labels'][id - 1])

            training_set['feature'].append([])
            training_set['feature'][-1].append(in_dataset['features'][id-2])
            training_set['feature'][-1].append(in_dataset['features'][id-1])

            if in_dataset['labels'][id - 2] != in_dataset['labels'][id - 1]:
                training_set['change_labels'].append(1)
            else:
                training_set['change_labels'].append(0)

            training_set['labels'].append([in_dataset['labels'][id - 2], in_dataset['labels'][id - 1]])

        train_index  = train_index + 1;
        if (train_index >= 4000):
            break

    for i in range(0, 1000):
        id = in_dataset['ids'][next_id-1]
        next_id = in_dataset['next_ids'][next_id-1]

        if next_id == -1:
            next_id = id + 1
            prev_null = 1
            testing_set['words'][test_index] = itoa(in_dataset['labels'][id - 2]) + itoa(in_dataset['labels'][id - 1])

            testing_set['feature'].append([])
            testing_set['feature'][-1].append(in_dataset['features'][id-2])
            testing_set['feature'][-1].append(in_dataset['features'][id-1])

            if in_dataset['labels'][id - 2] != in_dataset['labels'][id - 1]:
                testing_set['change_labels'].append(1)
            else:
                testing_set['change_labels'].append(0)

            testing_set['labels'].append([in_dataset['labels'][id - 2], in_dataset['labels'][id - 1]])

            test_index = test_index + 1
            if(test_index >= 1000):
                break
            testing_set['words'][test_index] = itoa(in_dataset['labels'][id - 1]) + "_"

            testing_set['feature'].append([])
            testing_set['feature'][-1].append(in_dataset['features'][id-1])
            testing_set['feature'][-1].append(np.zeros(128))
            testing_set['change_labels'].append(1)
            testing_set['labels'].append([in_dataset['labels'][id - 1], 26])


            test_index = test_index + 1
            if(test_index >= 1000):
                break

            continue

        if prev_null == 1:
            testing_set['words'][test_index] = "_" + itoa(in_dataset['labels'][id - 1])

            testing_set['feature'].append([])
            testing_set['feature'][-1].append(np.zeros(128))
            testing_set['feature'][-1].append(in_dataset['features'][id-1])
            testing_set['change_labels'].append(1)
            testing_set['labels'].append([26,in_dataset['labels'][id - 1]])

            prev_null = 0
        else:
            testing_set['words'][test_index] = itoa(in_dataset['labels'][id - 2]) + itoa(in_dataset['labels'][id - 1])


            testing_set['feature'].append([])
            testing_set['feature'][-1].append(in_dataset['features'][id-2])
            testing_set['feature'][-1].append(in_dataset['features'][id-1])

            if in_dataset['labels'][id - 2] != in_dataset['labels'][id - 1]:
                testing_set['change_labels'].append(1)
            else:
                testing_set['change_labels'].append(0)

            testing_set['labels'].append([in_dataset['labels'][id - 2], in_dataset['labels'][id - 1]])


        test_index = test_index + 1
        if(test_index >= 1000):
            break

    return (training_set,testing_set);



def read_OCR(filename, n_examples, n_features):
    F = open(filename)
    dataset = {}
    dataset['ids'] = np.zeros(n_examples, dtype=int)
    dataset['labels'] = np.zeros(n_examples,dtype=int)
    dataset['next_ids'] = np.zeros(n_examples,dtype=int)
    dataset['word_ids'] = np.zeros(n_examples,dtype=int)
    dataset['positions'] = np.zeros(n_examples,dtype=int)
    dataset['folds'] = np.zeros(n_examples,dtype=int)
    dataset['features'] = np.zeros([n_examples,n_features])

    i = 0
    for str_line in F.readlines():
        line0 = map(iors, filter(None, re.split('\t', str_line.strip())))

        dataset['ids'][i] = line0.pop(0)
        dataset['labels'][i] = atoi(line0.pop(0))
        dataset['next_ids'][i] = line0.pop(0)
        dataset['word_ids'][i] = line0.pop(0)
        dataset['positions'][i] = line0.pop(0)
        dataset['folds'][i] = line0.pop(0)
        if len(line0) != 128:  # Sanity check of the length
            print len(line0)

        for j, v in enumerate(line0):
            dataset['features'][i][j] = v
        i += 1
        if i == n_examples:
            break

    return dataset




#
# next_id1 = dataset1['next_ids'][0]
#
# for i in range(1,20):
#     idl = dataset1['ids'][next_idl-1]
#     next_id1 = dataset1['next_ids'][nextid1-1]
# Below is used for checking the loaded data
#

# print "The first 10 ids:",dataset1['ids'][:10]
# print "The shape of features:", dataset1['features'].shape
#
# print "ids[0]=",dataset1['ids'][0]
# print "labels[0]=", dataset1['labels'][0]
# print "The letter is ", itoa(dataset1['labels'][1])
# print "next_ids[0]=",dataset1['next_ids'][0]
# # print "The feature of the first example:",dataset1['features'][0]
# # Show the matrix into an image
# imshow(dataset1['features'][1].reshape(16,8), cmap='gray')
# # plt.show()
#
# string1 = itoa(dataset1['labels'][0])
#
# plt.figure()
# plt.subplot(4,5,1)
# imshow(dataset1['features'][0].reshape(16,8), cmap='gray')
# id1 = dataset1['ids'][0]
# next_id1 = dataset1['next_ids'][0]
#
# # Iterated Conditional Mode to avoid exhaustive mode
# # plt.show()
# # if the next_id1 is -1 then the word ends
# # print "------------- ", dataset1['ids'][], " ----------------- "
# for i in range(1,5000):
#     id1 = dataset1['ids'][next_id1-1]
#     next_id1 = dataset1['next_ids'][next_id1-1]
#
#     # print "id1", id1, "next_id1",  next_id1
#     # print "word id is ", dataset1['word_ids'][id1-1]
#     if next_id1 == -1: # Skip the next_id pointing to nothing
#         feature1 = np.zeros((16,8),dtype=int)
#         label1 = '_'
#         string1 += label1
#         # plt.subplot(4,5,i+1)
#         # imshow(feature1, cmap='gray')
#         next_id1 = id1+1
#         # print "**************** it is passed *********************";
#         continue
#
#     string1 += itoa(dataset1['labels'][next_id1-1])
#
#     # plt.subplot(4,5,i+1)
#     # imshow(dataset1['features'][next_id1-1].reshape(16,8), cmap='gray')
#     # plt.show()
# print string1

class MultiClassClassifierProblem:
    C = 1

    # sample = training_sample['feature'] -> (4000, 2, 128)
    # labels = training_sample['labels'] -> (4000, 2)
    def __init__(self, samples, labels):
        # dlib.solve_structural_svm_problem() expects the class to have
        # num_samples and num_dimensions fields.  These fields should contain
        # the number of training samples and the dimensionality of the PSI
        # feature vector respectively.
        self.num_samples = len(samples)
        # we have 27 labels including the dummy label, change_labels is 1
        self.num_dimensions = 128 * 27 * 2 + 1

        self.samples = samples #(4000, 2, 128)
        self.labels = labels #(4000, 2)


    # x -> (2,128)
    # label -> (2)
    def make_psi(self, x, label):
        """Compute PSI(x,label)."""
        psi = dlib.vector()
        # Set it to have 9 dimensions.  Note that the elements of the vector
        # are 0 initialized.
        psi.resize(self.num_dimensions)

        # first
        label_num = label[0]
        # psi[:label_num * 128] = label_num * 128 * [0]
        for index in range (128):
            psi[label_num*128 + index] = x[0][index]
        # psi[label_num*128, (label_num+1)*128] = x[0].tolist()
        # psi[(label_num+1)*128:] = (26 - label_num) * 128 * [0]

        label_num = label[1]
        for index in range (128):
            psi[label_num*128 + 128 * 27 + index] = x[1][index]

        # get changing label
        if label[0] != label[1]:
            psi[-1] = 1
        else:
            psi[-1] = 0

        return psi




    def get_truth_joint_feature_vector(self, idx):
        return self.make_psi(self.samples[idx], self.labels[idx])



    def separation_oracle(self, idx, current_solution):
        # self.sample[idx] -> (2,128)
        samp = self.samples[idx]

        Niter = 4;

        # implementing ICM here:
        psi = [0] * self.num_dimensions
        loss = 0.0

        # window length
        max_scoring_label = [0] * window_size

        for k in range(Niter):
            for iL in range(window_size):

                for i in range(27):
                    temp_label = list(max_scoring_label)
                    temp_label[iL] = i
                    temp_psi = self.make_psi(samp,temp_label)
                    score1 = dlib.dot(current_solution, temp_psi)

                    for j in range(window_size):
                        if self.labels[idx][j] != temp_label[j]:
                            score1 += 1

                    if loss < score1: # Search for the maximum and update loss, max_scoring_label, and psi
                        loss = score1
                        max_scoring_label[iL] = i
                        psi = temp_psi

        return loss, psi


if __name__ == "__main__":
    main()
