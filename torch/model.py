#%%

# 1:    RandomizeWeight(); /* initialize weight and bias */
# 2:    RandomizeBias();
# 5:    forwardCompt(step, iseq); /* input => Hidden, Hidden => output */

from torch import nn 

from utils import *
from get_data import *



class RNN_Model(nn.Module):

    def __init__(self):
        super(RNN_Model, self).__init__()  
        
        self.a = nn.Sequential(
            nn.Embedding(
                num_embeddings = input_dim,
                embedding_dim = char_encode_size),
            nn.ReLU(),
            nn.Linear(
                in_features = char_encode_size,
                out_features = 128),
            nn.ReLU(),)
            
        self.b = nn.GRU(
            input_size = 128,
            hidden_size = hidden_dim,
            batch_first = True)
        
        self.c = nn.Sequential(
            nn.Linear(
                in_features = hidden_dim, 
                out_features = 128),
            nn.ReLU(),
            nn.Linear(
                in_features = 128, 
                out_features = output_dim))
        
        # 1 and 2: Randomize weights
        self.apply(init_weights)
                
    # 5: Forward computation
    def forward(self, x, hidden_state):
        x = x.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)
        a = self.a(x)
        b, hidden_state = self.b(a, hidden_state)
        c = self.c(b)
        y_pred = torch.sigmoid(c).squeeze(0).squeeze(0)
        hidden_state = hidden_state.squeeze(0).squeeze(0)
        c_pred = one_hot_to_char(y_pred)
        return(hidden_state, y_pred, c_pred)
        




model = RNN_Model()
    


if __name__ == "__main__":
    # Display shapes of initialized weights and biases
    print(model)
    
    # Show forward computation
    h, y_pred, c_pred = model.forward(X[0], hidden_state)

    # Print the predicted output and corresponding character
    print("\nForward Model Output:")
    print(f"Hidden State (h): {h.shape}")
    print(f"Predicted One-hot Output (y): {y_pred.shape}")
    print(f"Predicted Character: {one_hot_to_char(y_pred)}")

# %%
