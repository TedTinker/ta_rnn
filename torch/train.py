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
from torch import nn 
from torch.optim import Adam

from utils import *
from get_data import *
from model import *



loss_function = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=learning_rate)


    
def epoch(epoch_num, total_epochs):
    global hidden_state
    hidden_state = hidden_state.detach()
    x_seq = X[epoch_num * maxStep : epoch_num * maxStep + maxStep]
    y_seq = Y[epoch_num * maxStep : epoch_num * maxStep + maxStep]
    
    y_pred_cache = []
    
    # 5: Forward passes
    for step in range(maxStep):
        hidden_state, y_pred, c_pred = model.forward(x_seq[step], hidden_state)
        # Compute the loss for this step
        # Reshape y_pred to [batch_size, output_dim]
        y_pred = y_pred.squeeze(0)  # y_pred: [1, 1, output_dim] -> [1, output_dim]
        y_pred_cache.append(y_pred)
        
    logits = torch.stack(y_pred_cache)  # Shape: [maxStep, output_dim]
    targets = y_seq.argmax(dim=-1)  # Convert one-hot to class indices, Shape: [maxStep]
    loss = loss_function(logits, targets)

    
    # 6: Backpropogation
    optimizer.zero_grad()  # Reset gradients before each epoch
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    if(total_epochs % 100 == 0):
        print("Epoch:", total_epochs)
        print("REAL:", "".join([one_hot_to_char(x) for x in x_seq]))
        print("PRED: ", "".join([one_hot_to_char(y) for y in y_pred_cache]))
    
    return(loss.item())
    
    
total_epochs = 0
epoch_num = 0
training_error = []




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
