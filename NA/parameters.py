"""
Params for Back propagation model
"""

# Different feature set version
# DATA_SET = 'DNA_RDF'
# DATA_SET = 'DNA_Bond_orientation'
# DATA_SET = 'DNA_Voronoi_volume'
# DATA_SET = 'DNA_RDF_Voronoi_volume'
# DATA_SET = 'DNA_RDF_Bond_orientation'
# DATA_SET = 'DNA_Voronoi_volume_Bond_orientation'
DATA_SET = 'DNA_RDF_Voronoi_volume_Bond_orientation'

# LINEAR = [13,  1000, 1000, 1000, 1000, 1000, 1000, 1000, 156]
#LINEAR = [13,  1000, 1000, 1000, 1000, 1000, 1000, 1000, 37]
LINEAR = [13,  1000, 1000, 1000, 1000, 1000, 1000, 1000, 95]

# @Poan, below is empty unless if you want to use convolutional layers in your network, which is also find 
CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

TEST_RATIO = 0.2

# Model Architectural Params for meta_material data Set
USE_LORENTZ = False

# Optimizer Params
OPTIM = "Adam"      # @Poan Optimizer, Adam is usually pretty good so dont need to touch
REG_SCALE = 0       # @Poan L2 regularization strength, you can tune this if the model overfit
BATCH_SIZE = 128    # @Poan this is batch size of training, smaller means train longer and more noise gradient, however smaller can lead to better local minimum
EVAL_BATCH_SIZE = 2048   # @Poan This is the number of initializations that we will try. I don;t think this is used in our DNA case (remember we hard code 12d binary and let 1d free varying)
EVAL_STEP = 5           # @Poan This is the number of every X epochs where the model would run validation data duringn training
TRAIN_STEP = 300    # @Poan for training stage
BACKPROP_STEP = 300   # @Poan This is during evaluation, how many backprop steps you want it to run
LEARN_RATE = 1e-2   # @Poan Learning rate, this is kind of important and need to be tuned, 1e-2 / 1e-3 are usually good starting point
LR_DECAY_RATE = 0.8 # @Poan LR(learning rate) needs to decay so that model recognize lower local minimum
STOP_THRESHOLD = 1e-5 # @Poan this is not useful, ignore

# Data specific Params # @Poan I don't think these are useful now 
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 310 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME =  DATA_SET + '_reverse_aux_loss_{}'.format(0.1)     # @Poan this is what name would be saved in default setting 
DATA_DIR = '../'                                               # All simulated simple dataset
GEOBOUNDARY =[-1,1,-1,1]    # @Poan This is the boundary loss setting, geometry boundary
NORMALIZE_INPUT = True 

# Running specific
USE_CPU_ONLY = False
EVAL_MODEL = "mm"       # @Poan THis is for evaluation only, which model you want to eval, usually it is rewritten in the evaluate.py