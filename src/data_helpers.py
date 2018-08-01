'''
Contains functions for loading CIFAR-10 data, and preparing data_sets.

Modification of Wolfgang Beyer's code form his tutorial:
"Simple Image Classification Models for the CIFAR-10 dataset using TensorFlow".
'''

import numpy as np
import pickle
import sys


def load_CIFAR10_batch(filename):
    '''
    Loads all the data from a single CIFAR-10 batch.

    Args:
        filename: The OS file path towards the CIFAR-10 batch.

    Returns:
        x: A dictionary containing the batch's images.
           (An image is a 3072 array of floats, 3072=32*32*3)
        y: A dictionary containing the batch's labels.
           (A label is an integer from 0 to 9)
    '''
    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            batch_dict = pickle.load(f)
        else:
            batch_dict = pickle.load(f, encoding='latin1')
        x = batch_dict['data']
        y = batch_dict['labels']
        x = x.astype(float)
        y = np.array(y)
    return x, y


def load_data():
    '''
    Loads all the CIFAR-10 data, merging the training batches together.

    Args:
        filename: The OS file path towards the CIFAR-10 batch.

    Returns:
        data_dict: A dictionary containing the following dictionnaries:
            images_train: All the images from the training set, 50000 arrays.
            labels_train: The labels for the training set images, 50000 ints.
            images_test: All the images from the testing batches, 10000 arrays.
            labels_test: The labels for the testing set images, 10000 ints.
            classes: All t10 possible class names, one for each label 0 to 9.
    '''
    xs = []
    ys = []
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    x_test, y_test = load_CIFAR10_batch('cifar-10-batches-py/test_batch')

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']

    # Normalize the data.
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_test': x_test,
        'labels_test': y_test,
        'classes': classes
    }
    return data_dict


def generate_random_batches(data, batch_size):
    '''
    Uses the CIFAR-10 data to generate random batches with the same given size.

    Args:
        data: The CIFAR-10 data from which the random batches are generated.
        batch_size: The size of the random batches that will be produced.

    Returns:
        A generator that randomly generates batches, then yields them.
    '''
    data = np.array(data)
    index = len(data)
    while True:
        index += batch_size
        if (index + batch_size > len(data)):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
        yield data[index:index + batch_size]
