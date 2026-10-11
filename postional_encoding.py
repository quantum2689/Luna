import numpy as np
import math

rng = np.random.default_rng()
step_size = 1.0491

embedding_matrix = rng.standard_normal((65,384))


def make_postional_encoding():
    postional_encoded_embd = []
    for pos, word_vector in enumerate(embedding_matrix):
        one_stamp = []
        n = 1
        for slot in range(len(word_vector)):
            if slot % 2 == 0:
                one_stamp.append(math.sin(pos / n))
            else:
                one_stamp.append(math.cos(pos / n))
                n *= step_size        # after the second slot of the pair
        postional_encoded_embd.append(one_stamp)
    return np.array(postional_encoded_embd)


postional_encoding_matrix = make_postional_encoding()

print(f"shape {postional_encoding_matrix.shape}")
print("\n")
np.save("postional_encoding_embd", postional_encoding_matrix)
