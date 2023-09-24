"""
This is the helper functions for evaluation purposes

"""
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def correct_dataset_name(name):
    output_name = name.replace('Peurifoy', 'nano')
    output_name = output_name.replace('Yang_sim', 'AEM')
    output_name = output_name.replace('Yang', 'AEM')
    output_name = output_name.replace('meta_material', 'AEM')
    return output_name

def get_test_ratio_helper(flags):
    """
    The unified place for getting the test_ratio the same for all methods for the dataset,
    This is for easier changing for multi_eval
    """
    if flags.data_set == 'ballistics':
        #return 0.00781                       # 100 in total
        #return 0.02
        #return 0.01
        return 0.1
        #return 0.039                        # 500 in total
    elif flags.data_set == 'sine_wave':
        #return 0.0125                        # 100 in total
        #return 0.02
        #return 0.1
        return 0.1
        #return 0.0625                        # 500 in total
    elif flags.data_set == 'robotic_arm':
        #return 0.02
        return 0.1                          # 500 in total
        #return 0.01
        #return 0.01                          # 100 in total
    elif flags.data_set == 'meta_material':
        return 0.1
        #return 0.02
        #return 0.1                        # 10000 in total for Meta material
    else:
        print("Your dataset is none of the artificial datasets")
        return None

def compare_truth_pred(pred_file, truth_file, cut_off_outlier_thres=None, quiet_mode=False):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    if isinstance(pred_file, str):      # If input is a file name (original set up)
        try:
            pred = np.loadtxt(pred_file, delimiter=' ')
        except:
            pred = np.loadtxt(pred_file, delimiter=',')
        try:
            truth = np.loadtxt(truth_file, delimiter=' ')
        except:
            truth = np.loadtxt(truth_file, delimiter=',')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        print('In the compare_truth_pred function, your input pred and truth is neither a file nor a numpy array')
    if not quiet_mode:
        print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    if len(np.shape(pred)) == 1:
        # Due to Ballistics dataset gives some non-real results (labelled -999)
        valid_index = pred != -999
        if (np.sum(valid_index) != len(valid_index)) and not quiet_mode:
            print("Your dataset should be ballistics and there are non-valid points in your prediction!")
            print('number of non-valid points is {}'.format(len(valid_index) - np.sum(valid_index)))
        pred = pred[valid_index]
        truth = truth[valid_index]
        # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
        pred = np.reshape(pred, [-1,1])
        truth = np.reshape(truth, [-1,1])
    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    if cut_off_outlier_thres is not None:
        mse = mse[mse < cut_off_outlier_thres]
        mae = mae[mae < cut_off_outlier_thres]

        
    return mae, mse


def plotMSELossDistrib(pred_file, truth_file, flags, save_dir='data/'):
    mae, mse = compare_truth_pred(pred_file, truth_file)
    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(mse)))
    eval_model_str = flags.eval_model.replace('/','_')
    plt.savefig(os.path.join(save_dir,
                            '{}.png'.format(eval_model_str)))
    print('(Avg MSE={:.4e})'.format(np.mean(mse)))
    mse_file_name = truth_file.replace('Ytruth','mse_list')
    with open(mse_file_name, 'a') as msef:
        np.savetxt(msef, mse)

