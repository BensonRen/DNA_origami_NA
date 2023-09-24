import os
import sys
import numpy as np
import scipy.io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=1, test_ratio=0.05, trainsetsize=None):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        random_state=rand_seed)
    if trainsetsize is not None:
        x_train = x_train[:trainsetsize, :]
        y_train = y_train[:trainsetsize, :]
    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train), len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

def normalize_np(x):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        assert np.max(x[:, i]) == 1, 'your normalization is wrong'
        assert np.min(x[:, i]) == -1, 'your normalization is wrong'
    return x

 
def read_DNA_origami_0203(flags, test_mode=False):
    # mode = 'test/' if test_mode else 'train/'
    data_dir = os.path.join('../', 'Simulated_DataSets', 'DNA_origami')
    x_file = os.path.join(data_dir, 'test' if test_mode else 'train', 'data_x.csv')
    data_x = pd.read_csv(x_file, header=None, sep=' ').values
    data_y = pd.read_csv(x_file.replace('data_x', 'data_y'), header=None, sep=' ').values
    if test_mode:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.99)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

def read_DNA_origami_various_feature_combination(flags, test_mode=False, trainsetsize=None):
    # mode = 'test/' if test_mode else 'train/'
    data_dir = os.path.join('../', 'Simulated_DataSets', 'DNA_origami')
    x_file = os.path.join(data_dir, 'test' if test_mode else 'train', flags.data_set.replace('DNA_','') + '_data_x.csv')
    data_x = pd.read_csv(x_file, header=None, sep=' ').values
    data_y = pd.read_csv(x_file.replace('data_x', 'data_y'), header=None, sep=' ').values
    if test_mode:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.99)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio, trainsetsize=trainsetsize)


def read_data(flags, test_mode=False, trainsetsize=None):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    :param flags: The input flag of the input data set
    :param test_mode: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    if  flags.data_set == 'DNA_origami':
        train_loader, test_loader = read_DNA_origami_0203(flags, test_mode=test_mode)
    elif 'DNA' in flags.data_set:
        train_loader, test_loader = read_DNA_origami_various_feature_combination(flags, test_mode=test_mode, trainsetsize=trainsetsize)
    else:
        sys.exit("Your flags.data_set entry is not correct, check again!")
    return train_loader, test_loader

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class SimulatedDataSet_class_1d_to_1d(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class SimulatedDataSet_class(Dataset):
    """ The simulated Dataset Class for 1d output purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]


class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]

class SimulatedDataSet_regress_1d_input(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind, :]
