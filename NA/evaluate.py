"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper
import torch


# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from thop import profile, clever_format

    
def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False, save_misc=False, 
                MSE_Simulator=False, save_Simulator_Ypred=True, init_lr=0.1, BDY_strength=1):

    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param multi_flag: The switch to turn on if you want to generate all different inference trial results
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print(model_dir)
    flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.backprop_step = eval_flags.backprop_step
    flags.test_ratio = get_test_ratio_helper(flags)

    if flags.data_set == 'meta_material':
        save_Simulator_Ypred = False
        print("this is MM dataset, setting the save_Simulator_Ypred to False")
    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1
    flags.lr = 0.1
    flags.BDY_strength = BDY_strength
    flags.eval_batch_size = eval_flags.eval_batch_size
    flags.train_step = eval_flags.train_step

    print(flags)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, test_mode=True)#, eval_data_all=eval_data_all)
    print("Making network now")
    
    # Make Network
    ntwk = Network(NA, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    

    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        dest_dir = '/home/sr365/DNA_NA/NA/multi_eval'
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        dest_dir += flags.data_set
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        #pred_file, truth_file = ntwk.evaluate(save_dir='/work/sr365/multi_eval/NA/' + flags.data_set, save_all=True,
        pred_file, truth_file = ntwk.evaluate(save_dir=dest_dir, save_all=True,
                                                save_misc=save_misc, MSE_Simulator=MSE_Simulator,save_Simulator_Ypred=save_Simulator_Ypred)
    else:
        pred_file, truth_file = ntwk.evaluate(save_misc=save_misc, MSE_Simulator=MSE_Simulator, save_Simulator_Ypred=save_Simulator_Ypred)

        # Plot the MSE distribution
        plotMSELossDistrib(pred_file, truth_file, flags)
        print("Evaluation finished")

def evaluate_from_file(model_dir, eval_file, y_labels_include, 
                       multi_flag=False,  init_lr=0.1, BDY_strength=1, loss_weight=None):

    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print(model_dir)
    flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.backprop_step = 300

    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1
    flags.lr = 0.1
    flags.BDY_strength = BDY_strength
    flags.eval_batch_size = eval_flags.eval_batch_size
    flags.train_step = eval_flags.train_step

    # Read the new data
    data = pd.read_csv(eval_file, index_col=0)
    # Normalize it
    prev_min = np.load('../Simulated_DataSets/DNA_origami/nn_feature_extracted_ver2min_val.npy')
    prev_max = np.load('../Simulated_DataSets/DNA_origami/nn_feature_extracted_ver2max_val.npy')
    data.iloc[:, 1:] -= prev_min
    data.iloc[:, 1:] /= (prev_max - prev_min)

    # Only take the relevant values
    data_y_cols = []
    for keyword in y_labels_include:
        data_y_cols += [col for col in data.columns if keyword in col]
    # Remove these two columns as they are always 0 and would cause trouble to normalization
    if 'RDF_0' in data_y_cols:
        data_y_cols.remove('RDF_0')
        data_y_cols.remove('RDF_1')
        # Only take the first 10 points RDF_2-11
        for i in range(12, 37):
            data_y_cols.remove('RDF_{}'.format(i))
    if 'Voronoi_volume_0' in data_y_cols:
        data_y_cols.remove('Voronoi_volume_0')
        data_y_cols.remove('Voronoi_volume_1')
        data_y_cols.remove('Voronoi_volume_2')

    data_y = data[data_y_cols].values
    data_x = np.zeros(len(data_y)) # Dummy data_x
    test_loader = torch.utils.data.DataLoader(data_reader.SimulatedDataSet_regress_1d_input(data_x, data_y),
                                              batch_size=1)
    
    print("Making network now")
    
    # Make Network
    ntwk = Network(NA, flags, None, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    

    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        dest_dir = '/home/sr365/DNA_NA/NA/multi_eval_'
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        dest_dir += flags.data_set
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        #pred_file, truth_file = ntwk.evaluate(save_dir='/work/sr365/multi_eval/NA/' + flags.data_set, save_all=True,
        pred_file, truth_file = ntwk.evaluate(save_dir=dest_dir, save_all=True,
                                                save_misc=False, MSE_Simulator=False,
                                                save_Simulator_Ypred=True, loss_weight=loss_weight)
    else:
        pred_file, truth_file = ntwk.evaluate(save_misc=False, MSE_Simulator=False, 
                                              save_Simulator_Ypred=True, loss_weight=loss_weight)

        # Plot the MSE distribution
        plotMSELossDistrib(pred_file, truth_file, flags)
        print("Evaluation finished")

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    eval_flags = flag_reader.read_flag()

    # evaluate_from_model('models/augmented_best_model', multi_flag=False, eval_data_all=False,
    #                      save_misc=False)

    # evaluate_from_model('models/cur_best_RDF_model', multi_flag=False, eval_data_all=False,
    #                      save_misc=False)

    # evaluate_from_model('models/retrain_0', multi_flag=True, eval_data_all=False,
    #                      save_misc=False)

    # combination = ['RDF']
    # combination = ['Voronoi_volume']
    # combination = ['Bond_orientation']
    # combination = ['RDF', 'Voronoi_volume']     # Inference time get out of memory???
    # combination = ['RDF', 'Bond_orientation']
    # combination = ['Voronoi_volume', 'Bond_orientation']
    combination = ['RDF', 'Voronoi_volume', 'Bond_orientation']

    model_name = 'best_DNA_' + '_'.join(combination)
    evaluate_from_file(model_name, '../Simulated_DataSets/DNA_origami/fake_crystal_changed_scaling.csv', #fake_crystal.csv', 
                       combination, multi_flag=True)#, loss_weight=np.exp(-np.arange(35)/3))