import numpy as np


# ---- config (same values as your hyperparameters; torch/device removed since NumPy is CPU only) ----
batch_size = 64      # independent sequences processed in parallel
block_size = 256     # maximum context length
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 65      # e.g. char-level tiny Shakespeare; set to your tokenizer size
head_size = 2

rng = np.random.default_rng()

#soft max
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# self-attention

class Head:
    def __init__(self, n_embd, head_size):

        self.head_size = head_size
        self.wq = rng.standard_normal((n_embd,head_size))
        self.wk = rng.random((n_embd,head_size))
        self.wV = rng.random((n_embd,head_size))
    def forward(self ,x):
        B,T,C = x.shape
        q = np.matmul(x,self.wq)
        k = np.matmul(x,self.wk)
        v = np.matmul(x,self.wk)

        score = q@k.swapaxes(-2, -1)/np.sqrt(self.head_size)

        causal = np.tril(np.ones((T, T), dtype=bool))
        score = np.where(causal, score, -np.inf)

        score = softmax(score) @ v
        return score

x = rng.standard_normal((2, 5, 16))
head = Head(n_embd=16, head_size=8)
print(head.forward(x)) 

def self_attention(wq,wk,wv,x):

    B, T, C = x.shape

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

