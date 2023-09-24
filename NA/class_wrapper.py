"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
from math import inf
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
# Own module
from utils.time_recorder import time_keeper

class Network(object):   
    # @Poan this is initializing the model, simply read through this is fine, just some settings
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        #self.log = SummaryWriter(self.ckpt_dir)                 # Create a summary writer for keeping the summary to the tensor board
        if not os.path.isdir(self.ckpt_dir) and not inference_mode:
            os.makedirs(self.ckpt_dir)
        self.best_validation_loss = float('inf')                # Set the BVL to large number
        self.CELoss = None                              # THis is for classification tasks

    def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        return model

    def make_loss(self, logit=None, labels=None, G=None, return_long=False, inverse=False, loss_weight=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param larger_BDY_penalty: For only filtering experiments, a larger BDY penalty is added
        :param return_long: The flag to return a long list of loss in stead of a single loss value,
                            This is for the forward filtering part to consider the loss
        :return: the total loss
        """
        if logit is None:
            return None
        square_diff = torch.square(logit - labels)
        if loss_weight is not None:
            square_diff *= torch.tensor(loss_weight, requires_grad=False, device='cuda')
        MSE_loss = torch.mean(square_diff)
        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_upper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = 0.1*torch.sum(BDY_loss_all)
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)


    def build_tensor(self, nparray, requires_grad=False):
        # @Poan Feel free to leave this unchanged
        # THis is helper function that put a numpy array into tensor so that we can pass gradient to it
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)


    def make_optimizer(self, optimizer_type=None):
        # @Poan Feel free to leave this unchanged
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # For eval mode to change to other optimizers
        if  optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        # @Poan Feel free to leave this unchanged
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        # @Poan Feel free to leave this unchanged
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def load(self):
        # @Poan Feel free to leave this unchanged
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))

    def augment_geometry(self, geometry):
        """
        DNA origma data augmentation, rotating 90 degrees would be equivalent to original structure
        """
        rotate_num = np.random.randint(0, 4)
        geometry[:, rotate_num * 3: 12], geometry[:, :rotate_num * 3] = geometry[:, :(4-rotate_num) * 3], geometry[:, (4-rotate_num) * 3:12]
        return geometry

    def train(self, data_augmentation=True):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        # @Poan this is the major training loop here
        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda().float()                          # Put data onto GPU
                    spectra = spectra.cuda().float()                            # Put data onto GPU
                if data_augmentation:                               # Implement the data augmentation
                    geometry = self.augment_geometry(geometry)
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

             # @Poan this is the validation loop, NOT ACTUAL EVALUATION (where you input spectra and model output geometry)
            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda().float()
                        spectra = spectra.cuda().float()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra, inverse=True)                   # compute the loss
                    test_loss += loss                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                # self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        #self.log.close()
        tk.record(1)                    # Record at the end of the training

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, 
                save_Simulator_Ypred=True, loss_weight=None):
        """
        The function to evaluate how good the Neural Adjoint is and output results
        :param save_dir: The directory to save the results
        :param save_all: Save all the results instead of the best one (T_200 is the top 200 ones)
        :param MSE_Simulator: Use simulator loss to sort (DO NOT ENABLE THIS, THIS IS OK ONLY IF YOUR APPLICATION IS FAST VERIFYING)
        :param save_misc: save all the details that are probably useless
        :param save_Simulator_Ypred: Save the Ypred that the Simulator gives
        (This is useful as it gives us the true Ypred instead of the Ypred that the network "thinks" it gets, which is
        usually inaccurate due to forward model error)
        :return:
        """
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 300
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                Xpred  = self.evaluate_one(spectra.float(), save_dir=save_dir, save_all=save_all, ind=ind, loss_weight=loss_weight)
                #tk.record(ind)                          # Keep the time after each evaluation for backprop
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                np.savetxt(fxp, Xpred)
        return Ypred_file, Ytruth_file

    
    def evaluate_one(self, target_spectra, save_dir='data/', save_all=False, ind=None, modulized=False,
                        init_from_Xpred=None, save_MSE_each_epoch=False, loss_weight=None):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to 
        :param save_dir: The directory to save to when save_all flag is true
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        :param FF(forward_filtering): [default to be true for historical reason] The flag to control whether use forward filtering or not
        :param save_MSE_each_epoch: To check the MSE progress of backprop
        :param save_output: Default to be true, however it is turned off during the timing function since the buffer time is not considered
        """
        
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))
        # Initialize the geometry_eval or the initial guess xs
        geometry_eval, real_value = self.initialize_geometry_eval()

        # In case the input geometry eval is not of the same size of batch size, modify batch size
        self.flags.eval_batch_size = geometry_eval.size(0)

        # Set up the learning schedule and optimizer
        self.optm_eval = self.make_optimizer_eval(real_value)
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([geometry_eval.size(0) ,-1]) #self.flags.eval_batch_size, -1])
        
        # Extra for early stopping
        loss_list = []

        saved_model_str = self.saved_model.replace('/', '_') + '_modulized_inference_' + str(ind)

        # Begin NA
        for i in range(self.flags.backprop_step):
            # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
            if init_from_Xpred is None:
                geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
            else:
                geometry_eval_input = geometry_eval
            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)                     # Get the output
            ###################################################
            # Boundar loss controled here: with Boundary Loss #
            ###################################################
            loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input, loss_weight=loss_weight)         # Get the loss
            loss.backward()                                             # Calculate the Gradient
            self.optm_eval.step()  # Move one step the optimizer
            loss_np = loss.data
            self.lr_scheduler.step(loss_np)
            # Extra step of recording the MSE loss of each epoch
            loss_list.append(np.copy(loss_np.cpu()))
        
        if save_MSE_each_epoch:
            with open('data/{}_MSE_progress_point_{}.txt'.format(self.flags.data_set ,ind),'a') as epoch_file:
                np.savetxt(epoch_file, loss_list)

        # Get the last epoch of MSE
        if init_from_Xpred is None:
            geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
        else:
            geometry_eval_input = geometry_eval
        logit = self.model(geometry_eval_input)                     # Get the output
        loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input, loss_weight=loss_weight)         # Get the loss

        if save_all:                # If saving all the results together instead of the first one,  # @Poan this is necessary as we want multiple geometry output for one spectra
            ranked_Xpred, sort_index = self.forward_filter(Yp=logit.cpu().data.numpy(), 
                                Yt=target_spectra_expand.cpu().data.numpy(),
                                Xp=geometry_eval_input.cpu().data.numpy(), loss_weight=loss_weight)

            print('shape of ranked_XPred', np.shape(ranked_Xpred))
            BP_FF_folder_name = save_dir
            
            Xpred_file = os.path.join(BP_FF_folder_name, 'test_Xpred_point{}.csv'.format(saved_model_str))
            
            with open(Xpred_file, 'a') as fxp:
                print(ranked_Xpred)
                np.savetxt(fxp, ranked_Xpred[:1000, :])  # @Poan  we are taking top 1000 ones here
            
            # Saving the Ypred file as well, but due to size limit, only top 20
            Ypred_file = Xpred_file.replace('Xpred','Ypred')
            with open(Ypred_file, 'a') as fyp:
                np.savetxt(fyp, logit.cpu().data.numpy()[sort_index[:20], :])
                
        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()
        
        # calculate the MSE list and get the best one
        ranked_Xpred, sort_index = self.forward_filter(Yp=Ypred, Yt=target_spectra_expand.cpu().data.numpy(),
                            Xp=geometry_eval_input.cpu().data.numpy(), loss_weight=loss_weight)
        
        return ranked_Xpred[0]
            

    def forward_filter(self, Yp, Yt, Xp, loss_weight=None):
        square_diff = np.square(Yp - Yt)
        if loss_weight is not None:
            square_diff *= loss_weight
        MSE_list = np.mean(square_diff, axis=1)
        BDY_list = self.get_boundary_loss_list_np(Xp)
        BDY_strength = 10
        MSE_list += BDY_list * BDY_strength
        sort_index = np.argsort(MSE_list)
        ranked_Xpred = Xp[sort_index]
        return ranked_Xpred, sort_index


    def get_boundary_loss_list_np(self, Xpred):
        """
        Return the boundary loss in the form of numpy array
        :param Xpred: input numpy array of prediction
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_upper_bound()
        X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
        BDY_loss = np.mean(np.maximum(0, np.abs(Xpred - X_mean) - 0.5*X_range), axis=1)
        return BDY_loss
        

    def initialize_geometry_eval(self, num_init=50):
        """
        Initialize the geometry eval according to different dataset. These 2 need different handling
        :return: The initialized geometry eval
        """        
        # DNA specific build a large tensor with all combinations
        permutation = torch.tensor(np.load('permutation.npy'), requires_grad=False, 
                                   device='cuda', dtype=torch.float)
        geometry_eval_list = []
        for _ in range(num_init):
            real_value = torch.tensor(np.random.random([4096, 1]), requires_grad=True,
                                    device='cuda', dtype=torch.float)
            geometry_eval_cur = torch.cat([permutation, real_value], axis=1)
            geometry_eval_list.append(geometry_eval_cur)
        geometry_eval = torch.cat(geometry_eval_list, axis=0)
        return geometry_eval, real_value

    def initialize_from_uniform_to_dataset_distrib(self, geometry_eval):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param geometry_eval: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_upper_bound()
        geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        return geometry_eval_input

    
    def get_boundary_lower_bound_upper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        x_dim = 13
        return np.array([1] * x_dim), np.array([0] * x_dim), np.array([1] * x_dim)

    def predict(self, Xpred_file, no_save=False, load_state_dict=None):
        """
        The prediction function, takes Xpred file and write Ypred file using trained model
        :param Xpred_file: Xpred file by (usually VAE) for meta-material
        :param no_save: do not save the txt file but return the np array
        :param load_state_dict: If None, load model using self.load() (default way), If a dir, load state_dict from that dir
        :return: pred_file, truth_file to compare
        """
        print("entering predict function")
        if load_state_dict is None:
            self.load()         # load the model in the usual way
        else:
            self.model.load_state_dict(torch.load(load_state_dict))
        Ypred_file = Xpred_file.replace('Xpred', 'Ypred')
        Ytruth_file = Ypred_file.replace('Ypred', 'Ytruth')
        Xpred = pd.read_csv(Xpred_file, header=None, delimiter=',')     # Read the input
        if len(Xpred.columns) == 1: # The file is not delimitered by ',' but ' '
            Xpred = pd.read_csv(Xpred_file, header=None, delimiter=' ')
        # DNA Origami takes 13 dimensional input
        Xpred = np.reshape(Xpred.values, [-1, 13])
        print('Xpred shape now = ', np.shape(Xpred))
        Xpred_tensor = torch.from_numpy(Xpred).to(torch.float)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            Xpred_tensor = Xpred_tensor.cuda()
        # Put into evaluation mode
        self.model.eval()
        Ypred = self.model(Xpred_tensor)
        print(Ypred.cpu().data.numpy())
        if load_state_dict is not None:
            Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + load_state_dict[-7:-4])
        elif self.flags.model_name is not None:
            Ypred_file = Ypred_file.replace('Ypred', os.path.basename(self.flags.model_name))
        if no_save:                             # If instructed dont save the file and return the array
            return Ypred.cpu().data.numpy(), Ytruth_file
        print('Ypred file:', Ypred_file)
        np.savetxt(Ypred_file, Ypred.cpu().data.numpy())

        return Ypred_file, Ytruth_file

    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
