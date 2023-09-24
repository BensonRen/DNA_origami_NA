"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
import numpy as np
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import NA
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

def training_from_flag(flags, trainsetsize=None):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, trainsetsize=trainsetsize)
    print("Making network now")


    # Just to make sure the last dimension of the linear layer matches the data
    for x, y in train_loader:
        flags.linear[-1] = y.size(1)
        break

    # Make Network
    ntwk = Network(NA, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()
    # ntwk.train_reverse()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)


        
def retrain_different_dataset(index):
    """
    This function is to evaluate all different datasets in the model with one function call
    """
    from utils.helper_functions import load_flags
    #model_name = 'models/cur_best_RDF_model_DNA'
    model_name =  'models/best_DNA_Bond_orientation'
    flags = load_flags(model_name)
    #flags = load_flags(os.path.join("models", "retrain0" + eval_model))
    flags.model_name = 'retrain_{}'.format(index)
    flags.data_dir = '../'
    flags.lr = 1e-3
    flags.eval_step = 2
    flags.batch_size = 256
    flags.train_step = 500
    flags.test_ratio = 0.1
    training_from_flag(flags)

def retrain_with_different_data_size():
    """
    The function to extract the effect of dataset size
    """
    from utils.helper_functions import load_flags
    #model_name = 'models/cur_best_RDF_model_DNA'
    model_name =  'models/best_DNA_Bond_orientation'
    # model_name =  'models/RDF_Angle_0305_best'
    flags = load_flags(model_name)
    #flags = load_flags(os.path.join("models", "retrain0" + eval_model))
    flags.data_dir = '../'
    flags.lr = 1e-3
    flags.eval_step = 2
    flags.batch_size = 256
    flags.train_step = 500
    flags.test_ratio = 0.1

    k = 7
    for trainsetsize in np.arange(k*1000, (k+1)*1000, 200):
        if trainsetsize == 0:
            continue
        flags.model_name = 'retrain_dataset_size_{}'.format(trainsetsize)
        training_from_flag(flags, trainsetsize=trainsetsize)

def hypersweep():
    # [1e-3, 5e-3, 5e-4, 1e-4]
    #lr = 1e-3
    #lr = 5e-3
    #lr = 5e-4
    lr = 1e-3
    num_nurons = 1000
    # Sweep the model size
    for num_layers in range(5, 10):
    # for num_layers in range(4, 10):
    #for num_layers in range(7, 15):
        #for bs in [100, 500, 1000]:
        for num_nurons in [100, 200, 500, 1000]:
        # for num_nurons in [100, 1000]:
        # for num_nurons in [200, 500]:
            # for lr in [1e-3, 5e-3, 5e-4, 1e-4]:
            # for reg_scale in [0]:
            for reg_scale in [0, 5e-4]:
            # for reg_scale in [1e-3]:
            #for reg_scale in [1e-2, 5e-3]:
            #for reg_scale in [ 8e-1]:
                for trail in range(1):
                #for trail in range(1, 2):
                    flags = flag_reader.read_flag()
                    dim_x, dim_y = flags.linear[0], flags.linear[-1]
                    # Changing
                    flags.linear = [num_nurons for i in range(num_layers)]
                    flags.linear[0] = dim_x
                    flags.linear[-1] = dim_y
                    flags.lr = lr
                    #flags.batch_size = bs
                    flags.reg_scale = reg_scale
                    flags.model_name = '{}/layer_{}_reg_{}_bs_{}/{}_size_{}x{}_lr_{}_decay_{}reg_{}_bs_{}_epoch{}_trail_{}'.format(flags.data_set, len(flags.linear), flags.reg_scale, flags.batch_size, flags.data_set, len(flags.linear), flags.linear[1], 
                                    flags.lr, flags.lr_decay_rate, flags.reg_scale, flags.batch_size, flags.train_step, trail)

                    training_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    retrain_with_different_data_size()

    # training_from_flag(flags)
    #retrain_different_dataset(3)
    # hypersweep()
    # Do the retraining for all the data set to get the training 
    # base = 10
    # j = 9
    # for i in range(j*base, (j+1)*base):
    # #for i in [10]:
    #     retrain_different_dataset(i)
    #retrain_different_dataset(0)
    # for worse_index in [10, 50, 100]:
    #retrain_different_dataset_notrain(100)
