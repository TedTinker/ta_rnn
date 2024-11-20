#%%

# 3:    ReadTeachingData(readfileptr); /* read teaching data from file */
# 4:    RandomizeInitStateAllSeqs(); /* randomize initial step context state for all sequences */

import numpy as np
import re
import requests

from utils import *

# 3: Get the text of "The Tale of Peter Rabbit"
url = "https://www.gutenberg.org/cache/epub/14838/pg14838.txt"
response = requests.get(url)
text = response.text

# Data cleaning
text = text.upper()                                 # Convert to uppercase
start_index = text.find("ONCE UPON A TIME")         # Remove everything before "ONCE UPON A TIME"
end_index = text.rfind("THE END")                   # Remove everything after "THE END"
text = text[start_index:end_index + len("THE END")]
text = re.sub(r'[^A-Z\s]', '', text)                # Remove all characters except letters and spaces
text = re.sub(r'\bILLUSTRATION\b', '', text)        # Remove the word "ILLUSTRATION"
text = re.sub(r'\s+', ' ', text).strip()            # Replace multiple whitespace with a single space

# Define a dictionary to map each character to an index
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}  

# Function to convert a character to a one-hot vector
def char_to_one_hot(char):
    one_hot = np.zeros(len(chars))
    one_hot[char_to_index[char]] = 1
    return one_hot

# Function to convert a one-hot vector back to a character
def one_hot_to_char(one_hot_vector):
    index = np.argmax(one_hot_vector)
    return index_to_char[index]

# Convert the cleaned text to a list of one-hot vectors
one_hot_vectors = np.array([char_to_one_hot(char) for char in text])

X = one_hot_vectors[:-1][:, :, np.newaxis]
Y = one_hot_vectors[1:][:, :, np.newaxis]



# 4: Single hidden state for each sequence
hidden_state = np.random.randn(hidden_dim, 1) * 0.01



if __name__ == "__main__":
    # Print the first few characters and their one-hot encoding
    for i in range(10):
        print(f"Character: {text[i]}, One-hot: {one_hot_vectors[i]}")

    # Print sizes of inputs (X) and outputs to predict (Y)
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    
