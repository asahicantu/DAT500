from tensorflow.keras.layers import LSTM

##########################################################
###  DEFINE TRAINING DATA FRAME PARAMETERS
##########################################################

# Number of iterations
N_STEPS = 100
# Window of 'n' next days to load as values to predict
WINDOW_OFFSET = 90

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# Select default feature columns to train the neural network
FEATURE_COLS = ['high','low','open','close', 'volume']


##########################################################
###  DEFINE DEFAULT NEURAL NETWORK PARAMETERS 
##########################################################

N_LAYERS = 3

# RNN=LSTM BEST SUITABLE FOR TIME SERIES PREDICTION
CELL = LSTM
# NUMBER OF "NEURONS TO CREATE"
N_UNITS = 256

DROPOUT = 0.4

##########################################################
###  DEFINE THE TRAINING PARAMETERS
##########################################################

# Loss type function, default is mean square error
LOSS = "mse"
OPTIMIZER = "rmsprop"
BATCH_SIZE = 64
EPOCHS = 300



RESULTS_PATH = "results"
LOG_PATH = "logs"
DATA_PATH = "data"
    
