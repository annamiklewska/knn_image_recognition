import pickle as pkl
import numpy as np
import time

import predict as p


def read_file(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def get_accuracy(test_data, test_labels, predictions):
    correct = 0
    td = len(test_data)
    tp = len(predictions)
    if td != tp:
        print("PROBLEM: ", td, tp)
    for x in range(len(test_data)):
        a = test_labels[x]
        b = predictions[x]
        if a == b:
            correct += 1
    return (correct / float(len(test_data))) * 100.0


def get_best_k(test_data, test_labels, train_data, train_labels):
    arr = []
    for i in range(100):
        arr.append(get_accuracy(test_data, test_labels, p.get_prediction(test_data, train_data, train_labels, i)))
    best_accuracy = np.amax(np.array(arr))
    best_k = np.where(arr == best_accuracy)
    print("Best accuracy, best k: ", best_accuracy, best_k)
    return best_k

'''
all_data = p.read_file('data_20k_3.pkl')
test_data = all_data[:2500]
train_data = all_data[10000:10700]

#file_name = 'testdata_2500.pkl'
#pickle.dump(test_data, open(file_name, 'wb'))
file_name = 'traindata_700.pkl'
pickle.dump(train_data, open(file_name, 'wb'))


all_labels = p.read_file('labels_20k_3.pkl')
test_labels = all_labels[0:2500]
train_labels = all_labels[10000:10700]

#file_name = 'testlabels_2500.pkl'
#pickle.dump(test_labels, open(file_name, 'wb'))
file_name = 'trainlabels_700.pkl'
pickle.dump(train_labels, open(file_name, 'wb'))
'''

'''
start = time.time()

data = p.read_file('testdata_2500.pkl')
labels = p.read_file('testlabels_2500.pkl')

print("data done")

prediction = p.predict(data)

end = time.time() - start
print("Time of execution: ", end)

accuracy = get_accuracy(data, labels, prediction)
print("Accuracy: ", accuracy, "%")

print("Predicted labels: ", prediction)
print("Actual labels: ", labels)
'''


def main():

    start = time.time()

    all_data = read_file('data_20k_3.pkl')
    test_data = all_data[:2500]
    #train_data = all_data[2500:4000]

    all_labels = read_file('labels_20k_3.pkl')
    test_labels = all_labels[0:2500]
    #train_labels = all_labels[25000:4000]

    #data = read_file('testdata_2500.pkl')
    #labels = read_file('testlabels_2500.pkl')

    print("data done")

    prediction = p.predict(test_data)

    end = time.time() - start
    print("Time of execution: ", end)

    accuracy = get_accuracy(test_data, test_labels, prediction)
    print("Accuracy: ", accuracy, "%")

    print("Predicted labels: ", prediction)
    print("Actual labels: ", test_labels)


if __name__ == '__main__':
    main()

