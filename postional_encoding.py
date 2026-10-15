import numpy as np
import math

block_size = 256   # context length
n_embd = 384
step_size = 1.0491

def make_postional_encoding():
    postional_encoded_embd = []
    for pos in range(block_size):        # loop over 256 time steps!
        one_stamp = []
        n = 1
        for slot in range(n_embd):
            if slot % 2 == 0:
                one_stamp.append(math.sin(pos / n))
            else:
                one_stamp.append(math.cos(pos / n))
                n *= step_size
        postional_encoded_embd.append(one_stamp)
    return np.array(postional_encoded_embd)

postional_encoding_matrix = make_postional_encoding()
print(f"New shape: {postional_encoding_matrix.shape}")  # (256, 384)
np.save("postional_encoding_embd", postional_encoding_matrix)