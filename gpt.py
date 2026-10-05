import numpy as np

## self attention
#soft max
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# self-attention
def self_attention(wq,wk,wv,x):
    # coumpute query-key-value
    q = np.dot(wq,x)
    k = np.dot(wk,x)
    v = np.dot(wv,x)

    # coumpute scores

    d_k = q.shape[-1]

    scores = np.dot(q ,k.T) / np.sqrt(d_k)
    # apply mask
    scores = np.where(True, scores, -1e9)

    # apply softmax

    weights = softmax(scores, axis=-1)

    # value coumputed
    return weights@ v

# relu activation function

def relu(z):    
    return np.maximum(0, z)

# mlp

def mlp_layer(w,b,x):
    z = np.dot(x,w) + b

    return relu(z)

