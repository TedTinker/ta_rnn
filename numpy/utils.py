# We will predict future text with numpy-based RNN using this algorithm.

# 1:    RandomizeWeight(); /* initialize weight and bias */
# 2:    RandomizeBias();
# 3:    ReadTeachingData(readfileptr); /* read teaching data from file */
# 4:    RandomizeInitStateAllSeqs(); /* randomize initial step context state for all sequences */
#        /* Learning of maxSeq sequences */
#        for (epoch=0; epoch<maxEpoch; ++epoch) {
#            for (iseq=0; iseq<maxSeq; ++iseq) { /* maxSeq is max # of sequences trained */
#                /* Forward computation */
#                for (step=0; step<maxStep[iseq]; ++step)
# 5:                forwardCompt(step, iseq); /* input => Hidden, Hidden => output */
#          
#                /* BackPropThroughTime computation */
#                for (step=maxStep[iseq]-1; step>=0; --step)
# 6:                backProp(step, iseq); /* output => hidden, hidden => input */
#          
# 7:            updateInitState(iseq);
#            }
# 8:        updateweight();
# 9:        updatebias();
#       }
# 10:   SaveWeightBias(fileptr);
# 11:   SaveActivationHiddenOutforAllStepsAllSeqs(act-fileptr);

import numpy as np
import pickle

# Define some constants
input_dim = 27              # Input feature dimension
hidden_dim = 50             # Hidden layer dimension
output_dim = 27             # Output dimension
max_epoch = 10000             # Number of training epochs
learning_rate = 0.01        # How to scale backpropogation
maxStep = 30                # Example sequence lengths (list of lengths for each sequence)

# Activation functions
def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / exps.sum(axis=0)



# 10 and 11: Save model and hidden state
def save_model(model, hidden_state):
    with open('saved/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'hidden_state': hidden_state}, f)

# Load model and hidden state
def load_model():
    with open('saved/model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['hidden_state']