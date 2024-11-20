#%%

# 1:    RandomizeWeight(); /* initialize weight and bias */
# 2:    RandomizeBias();
# 5:    forwardCompt(step, iseq); /* input => Hidden, Hidden => output */

import numpy as np

from utils import *
from get_data import *



class RNN_Model:
    
    def __init__(self): 
        # 1: Randomly begin weights
        self.hidden_x_weights = np.random.randn(hidden_dim, input_dim) * 0.01       # Input to hidden weights
        self.hidden_hidden_weights = np.random.randn(hidden_dim, hidden_dim) * 0.01 # Hidden to hidden weights
        self.hidden_y_weights = np.random.randn(output_dim, hidden_dim) * 0.01      # Hidden to output weights
        # 2: Randomly begin biases
        self.hidden_bias = np.random.randn(hidden_dim, 1) * 0.01                    # Hidden layer bias
        self.output_bias = np.random.randn(output_dim, 1) * 0.01                    # Output layer bias

    # 5: Forward computation
    def forward(self, x, previous_hidden_state):
        hidden_state = tanh(np.dot(self.hidden_x_weights, x) + np.dot(self.hidden_hidden_weights, previous_hidden_state) + self.hidden_bias)
        y_pred = softmax(np.dot(self.hidden_y_weights, hidden_state) + self.output_bias)
        c_pred = one_hot_to_char(y_pred)
        return hidden_state, y_pred, c_pred
    
    def start_gradients(self):
        self.d_hidden_x_weights         = np.zeros_like(self.hidden_x_weights)
        self.d_hidden_hidden_weights    = np.zeros_like(self.hidden_hidden_weights) 
        self.d_hidden_y_weights         = np.zeros_like(self.hidden_y_weights)
        self.d_hidden_bias              = np.zeros_like(self.hidden_bias)
        self.d_output_bias              = np.zeros_like(self.output_bias)
        
    def clip_gradients(self):
        self.d_hidden_x_weights         = np.clip(self.d_hidden_x_weights, -1, 1)
        self.d_hidden_hidden_weights    = np.clip(self.d_hidden_hidden_weights, -1, 1)
        self.d_hidden_y_weights         = np.clip(self.d_hidden_y_weights, -1, 1)
        self.d_hidden_bias              = np.clip(self.d_hidden_bias, -1, 1)
        self.d_output_bias              = np.clip(self.d_output_bias, -1, 1)

model = RNN_Model()
    


if __name__ == "__main__":
    # Display shapes of initialized weights and biases
    print("Shapes of Weights and Biases:")
    print(f"hidden_x_weights: {model.hidden_x_weights.shape}")
    print(f"hidden_hidden_weights: {model.hidden_hidden_weights.shape}")
    print(f"hidden_y_weights: {model.hidden_y_weights.shape}")
    print(f"hidden_bias: {model.hidden_bias.shape}")
    print(f"output_bias: {model.output_bias.shape}")
    
    # Example inputs
    x = np.random.randn(input_dim, 1)  # Random input vector (input_dim x 1)
    h_prev = np.zeros((hidden_dim, 1)) # Initial hidden state (hidden_dim x 1)

    # Show forward computation
    h_prev = hidden_state  # Initialize hidden state
    h, y_pred, c_pred = model.forward(X[0].reshape(-1, 1), hidden_state)

    # Print the predicted output and corresponding character
    print("\nForward Model Output:")
    print(f"Hidden State (h): {h.shape}")
    print(f"Predicted One-hot Output (y): {y_pred.shape}")
    print(f"Predicted Character: {one_hot_to_char(y_pred)}")
