import numpy as np
from tokenizer import stoi , vocab_size

# ---- config (same values as your hyperparameters; torch/device removed since NumPy is CPU only) ----
batch_size = 64      # independent sequences processed in parallel
block_size = 256     # maximum context length
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 65      # e.g. char-level tiny Shakespeare; set to your tokenizer size
head_size = 2
dropout = 0.2

rng = np.random.default_rng()

# Embedding and postional encoding
class Embedding:
    def __init__(self):
        self.embedding_matrix = rng.standard_normal((vocab_size,n_embd))*0.02
        self.pe = np.load("postional_encoding_embd.npy")
    def forward(self,x):
        return self.embedding_matrix[x] + self.pe[x.shape[-1]]


#soft max
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# self-attention

class Head:
    def __init__(self, n_embd, head_size):

        self.head_size = head_size
        self.wq = rng.standard_normal((n_embd,head_size)) * 0.02
        self.wk = rng.standard_normal((n_embd,head_size)) * 0.02
        self.wV = rng.standard_normal((n_embd,head_size)) * 0.02
    def forward(self ,x):
        B,T,C = x.shape
        q = np.matmul(x,self.wq)
        k = np.matmul(x,self.wk)
        v = np.matmul(x,self.wV)

        score = q@k.swapaxes(-2, -1)/np.sqrt(self.head_size)

        causal = np.tril(np.ones((T, T), dtype=bool))
        score = np.where(causal, score, -np.inf)

        score = softmax(score) @ v
        return score

x = rng.standard_normal((2, 5, 16))
head = Head(n_embd=16, head_size=8)
print(head.forward(x)) 


class MultiHeadAttention:
    def __init__(self, n_embd, n_heads):
        head_size = n_embd // n_heads
        self.heads = [Head(n_embd, head_size) for _ in range(n_heads)]
        self.wo = rng.standard_normal((n_embd, n_embd)) * 0.02

    def forward(self, x):
        outs = [h.forward(x) for h in self.heads]   # each (B, T, head_size)
        out = np.concatenate(outs, axis=-1)         # (B, T, n_embd)
        return out @ self.wo                        # (B, T, n_embd)

x = rng.standard_normal((2, 5, 16))
mha = MultiHeadAttention(n_embd=16, n_heads=4)
print(mha.forward(x))

# layer Norm

class LayerNorm:
    def __init__(self, n_embd):
        self.eps = 1e-5
        self.beta = np.zeros((n_embd))
        self.gamma = np.ones((n_embd))
    def forward(self,x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * norm + self.beta

x = rng.standard_normal((n_embd))
print(LayerNorm(n_embd).forward(x))   

# FeedFoward network
# n_embd -
class FeedFoward:
    def __init__(self,n_embd):
        self.w1 = rng.standard_normal((n_embd, 4*n_embd))
        self.b1 = rng.standard_normal(4*n_embd)

        self.w2 = rng.standard_normal((4*n_embd, n_embd))
        self.b2 = rng.standard_normal((n_embd))
    def relu(self,x):
        return np.maximum(0,x)
    def forward(self,x):
        # first layer
        y = x @ self.w1 + self.b1
        y = self.relu(y)

        # second layer
        z = y @ self.w2 + self.b2

        return z

x = rng.standard_normal((n_embd))
net = FeedFoward(n_embd=n_embd)
print(net.forward(x))

# transfoermer block
class block:
    def __init__(self,n_embd, n_head):
        head_size = n_embd // n_head
        self.ma = MultiHeadAttention(n_embd,head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = LayerNorm((n_embd))
        self.ln2 = LayerNorm((n_embd))
    def forward(self,x):
        x = x + self.ma.forward(self.ln1.forward(x))
        x = x + self.ffwd.forward(self.ln2.forward(x))

        return x   

    
x = rng.standard_normal((2,5,n_embd))
B = block(n_embd,n_head)
print(B.forward(x))