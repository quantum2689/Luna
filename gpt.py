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

# Embedding and postional encoding
class Embedding:
    def __init__(self,vocab_size,n_embd):
        self.embedding_matrix = rng.standard_normal((vocab_size,n_embd))*0.02
        self.pe = np.load("postional_encoding_embd.npy")
    def forward(self,x):
        self.x = x
        return self.embedding_matrix[x] + self.pe[:x.shape[-1]]
    def backward(self, dout):
        self.dE = np.zeros_like(self.embedding_matrix)
        np.add.at(self.dE, self.x, dout) 
        return self.dE


#soft max
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def softmax_backward(y, dy):
    # in attention
    return y * (dy - np.sum(dy * y, axis=-1, keepdims=True))

# Dropout
class Dropout:
    def __init__(self,p,training):
        self.p = p
        self.training = training
        self.mask = None
    def forward(self,x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        else:
            self.mask = (rng.random(x.shape) >= self.p) / (1.0 - self.p)
            return x * self.mask
    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask

#d = Dropout(dropout,True)
#x = rng.standard_normal((2,5,n_embd))
#d.forward(x)

# self-attention

class Head:
    def __init__(self, n_embd, head_size,training):

        self.head_size = head_size
        self.wq = rng.standard_normal((n_embd,head_size)) * 0.02
        self.wk = rng.standard_normal((n_embd,head_size)) * 0.02
        self.wV = rng.standard_normal((n_embd,head_size)) * 0.02

        self.dropout = Dropout(dropout,training)
    def forward(self ,x):
        B,T,C = x.shape
        q = np.matmul(x,self.wq)
        k = np.matmul(x,self.wk)
        v = np.matmul(x,self.wV)

        score = q@k.swapaxes(-2, -1)/np.sqrt(self.head_size)

        causal = np.tril(np.ones((T, T), dtype=bool))
        score = np.where(causal, score, -np.inf)

        score = softmax(score)
        att = score
        score = self.dropout.forward(score)
        att_d = score
        self.cache = (x, q, k, v, att, att_d)
        return score @ v
    def backward(self, dout):
        x, q, k, v, att, att_d = self.cache
        datt_d = dout @ v.swapaxes(-2, -1)
        dv = att_d.swapaxes(-2, -1) @ dout
        dscore = softmax_backward(att, self.dropout.backward(datt_d)) / np.sqrt(self.head_size)
        dq = dscore @ k
        dk = dscore.swapaxes(-2, -1) @ q
        self.dwq = np.einsum("btc,bth->ch", x, dq)
        self.dwk = np.einsum("btc,bth->ch", x, dk)
        self.dwV = np.einsum("btc,bth->ch", x, dv)
        return dq @ self.wq.T + dk @ self.wk.T + dv @ self.wV.T

#x = rng.standard_normal((2, 5, 16))
#head = Head(n_embd=16, head_size=8,traning=True)
#print(head.forward(x)) 


class MultiHeadAttention:
    def __init__(self, n_embd, n_heads,training):
        head_size = n_embd // n_heads
        self.heads = [Head(n_embd, head_size,training=training) for _ in range(n_heads)]
        self.wo = rng.standard_normal((n_embd, n_embd)) * 0.02

    def forward(self, x):
        outs = [h.forward(x) for h in self.heads]   # each (B, T, head_size)
        out = np.concatenate(outs, axis=-1)         # (B, T, n_embd)
        self.concat = out
        return out @ self.wo                       # (B, T, n_embd)
    def backward(self, dout):
        self.dwo = np.einsum("btc,btd->cd", self.concat, dout)
        dconcat = dout @ self.wo.T
        hs = self.heads[0].head_size
        dx = 0
        for i, h in enumerate(self.heads):
            dx = dx + h.backward(dconcat[..., i*hs:(i+1)*hs])
        return dx
#x = rng.standard_normal((2, 5, 16))
#mha = MultiHeadAttention(n_embd=16, n_heads=4,training=True)
#print(mha.forward(x))

# layer Norm

class LayerNorm:
    def __init__(self, n_embd):
        self.eps = 1e-5
        self.beta = np.zeros((n_embd))
        self.gamma = np.ones((n_embd))
    def forward(self,x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        inv_std = 1.0 / np.sqrt(var + self.eps)
        norm = (x - mean) * inv_std
        self.xhat = norm
        self.inv_std = inv_std
        return self.gamma * norm + self.beta
    def backward(self, dout):
        N = self.xhat.shape[-1]
        self.dgamma = (dout * self.xhat).reshape(-1, N).sum(0)
        self.dbeta = dout.reshape(-1, N).sum(0)
        dxhat = dout * self.gamma
        return (self.inv_std / N) * (N * dxhat
                - dxhat.sum(-1, keepdims=True)
                - self.xhat * (dxhat * self.xhat).sum(-1, keepdims=True))

#x = rng.standard_normal((n_embd))
#print(LayerNorm(n_embd).forward(x))   

# FeedFoward network
# n_embd -
class FeedFoward:
    def __init__(self,n_embd,training):
        self.w1 = rng.standard_normal((n_embd, 4*n_embd))*0.02
        self.b1 = np.zeros(4*n_embd)

        self.w2 = rng.standard_normal((4*n_embd, n_embd))*0.02
        self.b2 = rng.standard_normal((n_embd))

        self.dropout = Dropout(dropout,training)

    def relu(self,x):
        return np.maximum(0,x)
    def forward(self,x):
        self.x = x
        # first layer
        self.y1 = x @ self.w1 + self.b1
        self.a = self.relu(self.y1)

        # second layer
        z = self.a @ self.w2 + self.b2
        z = self.dropout.forward(z)

        return z
    def backward(self, dout):
        dz = self.dropout.backward(dout)
        self.dw2 = np.einsum("btf,btc->fc", self.a, dz)
        self.db2 = dz.reshape(-1, dz.shape[-1]).sum(0)
        dy1 = (dz @ self.w2.T) * (self.y1 > 0)          # relu'
        self.dw1 = np.einsum("btc,btf->cf", self.x, dy1)
        self.db1 = dy1.reshape(-1, dy1.shape[-1]).sum(0)
        return dy1 @ self.w1.T

#x = rng.standard_normal((n_embd))
#net = FeedFoward(n_embd=n_embd,training=True)
#print(net.forward(x))

# transfoermer block
class Block:
    def __init__(self,n_embd, n_head,training):
        head_size = n_embd // n_head
        self.ma = MultiHeadAttention(n_embd,n_head,training=training)
        self.ffwd = FeedFoward(n_embd,training=training)
        self.ln1 = LayerNorm((n_embd))
        self.ln2 = LayerNorm((n_embd))
    def forward(self,x):
        x = x + self.ma.forward(self.ln1.forward(x))
        x = x + self.ffwd.forward(self.ln2.forward(x))

        return x
    def backward(self,dout):
        dx1 = dout + self.ln2.backward(self.ffwd.backward(dout))
        return dx1 + self.ln1.backward(self.ma.backward(dx1))


    
#x = rng.standard_normal((2,5,n_embd))
#B = Block(n_embd,n_head,training=True)
#print(B.forward(x))

# logits linear

class Lm_head:
    def __init__(self,n_embd, vocab_size):
        self.lm = rng.standard_normal((n_embd,vocab_size))*0.02
    def forward(self,x):
        self.x = x
        return x @ self.lm
    def backward(self, dlogits):
        self.dlm = np.einsum("btc,btv->cv", self.x, dlogits)
        return dlogits @ self.lm.T

# GPT

class GPT:
    def __init__(self,vocab_size,n_embd,n_head,n_layer,training):
        self.embedding = Embedding(vocab_size=vocab_size,n_embd=n_embd)
        self.blocks = [Block(n_embd=n_embd,n_head=n_head,training=training) for _ in range(n_layer)]
        self.lm_head = Lm_head(n_embd=n_embd,vocab_size=vocab_size)

    def forward(self,x):
        x = self.embedding.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        logits = self.lm_head.forward(x)

        return logits
    def backward(self,dout):
        dx = self.lm_head.backward(dout)
        for block in reversed(self.blocks):
            dx = block.backward(dx)            
        dx = self.embedding.backward(dx)
        return dx




#model = GPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer,training=True)
#ids = rng.integers(0, 65, size=(2, 5))
#logits = model.forward(ids)
#print("Logits shape:", logits.shape)   # (2, 5, 65)


#dlogits = np.ones_like(logits)
#dx = model.backward(dlogits)
#print("Backward pass completed successfully!")