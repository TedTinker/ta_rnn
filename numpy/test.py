#%%

import numpy as np

from utils import *
from model import *
from get_data import *

# Load the trained model
model, hidden_state = load_model()

def generate_text(seed_text, num_steps):

    # Initialize the hidden state
    hidden_state = np.random.randn(hidden_dim, 1) * 0.01
    
    # Forward passes to update hidden state
    for char in seed_text[:-1]:
        one_hot = char_to_one_hot(char).reshape(-1, 1)
        hidden_state, y_pred, c_pred = model.forward(one_hot, hidden_state)

    # Generate num_steps characters
    new_chars = []
    char = seed_text[-1]
    for _ in range(num_steps):
        # Perform forward computation
        one_hot = char_to_one_hot(char).reshape(-1, 1)
        hidden_state, y_pred, c_pred = model.forward(one_hot, hidden_state)
        char = c_pred
        new_chars.append(c_pred)

    print(seed_text + " | " + "".join(new_chars))


if __name__ == "__main__":
    # Input seed text
    seed = "ONCE UPON A TIME"
    steps = 100  # Number of characters to generate
    
    # Generate text
    generate_text(seed, steps)
# %%
