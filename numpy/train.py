#%% 

#       /* Learning of maxSeq sequences */
#       for (epoch=0; epoch<maxEpoch; ++epoch) {
#           for (iseq=0; iseq<maxSeq; ++iseq) { /* maxSeq is max # of sequences trained */
#               /* Forward computation */
#               for (step=0; step<maxStep[iseq]; ++step)
# 5:                forwardCompt(step, iseq); /* input => Hidden, Hidden => output */
#          
#               /* BackPropThroughTime computation */
#               for (step=maxStep[iseq]-1; step>=0; --step)
# 6:                backProp(step, iseq); /* output => hidden, hidden => input */
#          
# 7:            updateInitState(iseq);
#           }
# 8:        updateweight();
# 9:        updatebias();
#       }
# 10:   SaveWeightBias(fileptr);
# 11:   SaveActivationHiddenOutforAllStepsAllSeqs(act-fileptr);

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from get_data import *
from model import *


    
def epoch(epoch_num, total_epochs):
    global hidden_state
    x_seq = X[epoch_num * maxStep : epoch_num * maxStep + maxStep]
    y_seq = Y[epoch_num * maxStep : epoch_num * maxStep + maxStep]
        
    hidden_state_cache = []
    y_pred_cache = []
    model.start_gradients()
    
    # 5: Forward passes
    for step in range(maxStep):
        hidden_state, y_pred, c_pred = model.forward(x_seq[step], hidden_state)
        hidden_state_cache.append(hidden_state)
        y_pred_cache.append(y_pred)
    
    # Gradient regarding hidden state
    dh_next = np.zeros_like(hidden_state)
    
    # 6: Backpropogation
    for step in reversed(range(maxStep)):
        # Gradients regarding outputs
        dy = y_pred_cache[step] - y_seq[step]
        model.d_hidden_y_weights += np.outer(dy, hidden_state_cache[step])
        model.d_output_bias += dy
        
        # Gradients regarding hidden state
        dh = np.dot(model.hidden_y_weights.T, dy) + dh_next
        dtanh = (1 - hidden_state_cache[step] ** 2) * dh
        model.d_hidden_bias += dtanh
        model.d_hidden_x_weights += np.dot(dtanh, x_seq[step].T)
        model.d_hidden_hidden_weights += np.dot(dtanh, hidden_state_cache[step - 1].T) if step > 0 else 0
        
        # Update dh_next for the next time step
        dh_next = np.dot(model.hidden_hidden_weights.T, dtanh)
        
    model.clip_gradients()
        
    # 8 and 9: Update weights and biases
    model.hidden_x_weights -= learning_rate * model.d_hidden_x_weights
    model.hidden_hidden_weights -= learning_rate * model.d_hidden_hidden_weights
    model.hidden_y_weights -= learning_rate * model.d_hidden_y_weights
    model.hidden_bias -= learning_rate * model.d_hidden_bias
    model.output_bias -= learning_rate * model.d_output_bias
    
    if(total_epochs % 100 == 0):
        print("REAL:", "".join([one_hot_to_char(x) for x in x_seq]))
        print("PRED: ", "".join([one_hot_to_char(y) for y in y_pred_cache]))
    
    error = dy.sum()
    return(error)
    
    
total_epochs = 0
epoch_num = 0
training_error = []



#%% 

# Train the RNN
for i in range(max_epoch):
    total_epochs += 1
    epoch_num += 1
    if(epoch_num * maxStep + maxStep >= len(X)):
        epoch_num = 0
    training_error.append(epoch(epoch_num, total_epochs))

plt.plot(training_error)
plt.show()
plt.close()
    
# 10 and 11: Save model and hidden state
save_model(model, hidden_state)


# %%
