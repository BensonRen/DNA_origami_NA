"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
from NA import flag_reader
from NA.class_wrapper import Network
from NA.model_maker import NA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
import torch
# Libs
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

def predict_from_model(pre_trained_model, Xpred_file, no_plot=True, load_state_dict=None):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param Xpred_file: The Prediction file position
    :param no_plot: If True, do not plot (For multi_eval)
    :param load_state_dict: The new way to load the model for ensemble MM
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    print(pre_trained_model)
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode
    flags.test_ratio = 0.1              #useless number  

    # Get the data, this part is useless in prediction but just for simplicity
    #train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    
    if not no_plot:
        # Plot the MSE distribution
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=False, load_state_dict=load_state_dict)
        flags.eval_model = pred_file.replace('.','_') # To make the plot name different
        plotMSELossDistrib(pred_file, truth_file, flags)
    else:
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=True, load_state_dict=load_state_dict)
    
    print("Evaluation finished")

    return pred_file, truth_file, flags


def predict_all(models_dir="data"):
    """
    This function predict all the files in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


if __name__ == '__main__':
    predict_from_model("models/best_DNA_RDF_Voronoi_volume_Bond_orientation", 
                       'data/DNA_Xpred.csv', no_plot=False, load_state_dict=None)
