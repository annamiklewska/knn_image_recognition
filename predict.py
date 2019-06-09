import pickle as pkl
import numpy as np


def read_file(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def euclidean_dist(instance1, instance2):
    return np.linalg.norm(np.subtract(instance1, instance2))


def get_distances(data, instance):
    return np.array([euclidean_dist(data[i], instance) for i in range(data.shape[0])])


def add_labels(distances, labels):
    return np.array([(distances[i], labels[i]) for i in range(distances.shape[0])])


def get_sorted_distances_with_labels(distances_with_labels):
    order = np.argsort(distances_with_labels[:,0], kind='mergesort')
    return distances_with_labels[order]


def get_neighbors(sorted_distances_with_labels, k):
    return np.array([sorted_distances_with_labels[i] for i in range(k)])


def pick_label(neighbors):
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    k = neighbors.shape[0]
    for i in range(k):
        labels[int(neighbors[i][1])] += 1
    best = np.where(labels == np.amax(labels))
    while best[0].shape[0] > 1:
        best = pick_label(get_neighbors(neighbors, k - 1))
        if np.isscalar(best):
            break
    if np.isscalar(best):
        return best
    else:
        return best[0][0]


def predict(x):

    """
    Function takes images as the argument. They are stored in the matrix X (NxD).
    Function returns a vector y (Nx1), where each element of the vector is a class numer {0, ..., 9} associated with recognized type of cloth.
    :param x: matrix NxD
    :return: vector Nx1
    """
    k = 5
    data = read_file('traindata_700.pkl')

    labels = read_file('trainlabels_700.pkl')
    
    return np.transpose(np.array([[pick_label(get_neighbors(get_sorted_distances_with_labels(add_labels(get_distances(data, x[i]), labels)), k)) for i in range(x.shape[0])]]))
