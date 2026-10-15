import os
import numpy as np
from gpt import GPT, vocab_size, n_embd, n_head, n_layer
from tokenizer import *



model = GPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer,training=True)

num_steps = 500
learning_rate = 3e-4
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?


# Train and test splits
data = np.array(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # Select train or val data
    data = train_data if split == 'train' else val_data
    
    # Pick random starting indices in the dataset
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    
    # Slice inputs X and target Y (shifted by 1)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    
    return x, y

def compute_loss(logits, targets):
    # Softmax
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    # Cross-entropy loss
    B, T, V = logits.shape
    N = B * T
    flat_probs = probs.reshape(N, V)
    flat_targets = targets.reshape(N)
    loss = -np.mean(np.log(flat_probs[np.arange(N), flat_targets] + 1e-10))
    # Gradient: (probs - target) / total_tokens
    dlogits = flat_probs.copy()
    dlogits[np.arange(N), flat_targets] -= 1.0
    dlogits = (dlogits / N).reshape(B, T, V)
    return loss, dlogits



class AdamW:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, model):
        self.t += 1
        # Bias-corrected learning rate
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        # Collect all parameters: (key, param_array, grad_array, is_weight)
        params = [
            ("emb", model.embedding.embedding_matrix, model.embedding.dE, True),
            ("lm", model.lm_head.lm, model.lm_head.dlm, True),
        ]
        for i, b in enumerate(model.blocks):
            params += [
                (f"b{i}_w1", b.ffwd.w1, b.ffwd.dw1, True),
                (f"b{i}_b1", b.ffwd.b1, b.ffwd.db1, False),
                (f"b{i}_w2", b.ffwd.w2, b.ffwd.dw2, True),
                (f"b{i}_b2", b.ffwd.b2, b.ffwd.db2, False),
                (f"b{i}_wo", b.ma.wo, b.ma.dwo, True),
                (f"b{i}_ln1_g", b.ln1.gamma, b.ln1.dgamma, False),
                (f"b{i}_ln1_b", b.ln1.beta, b.ln1.dbeta, False),
                (f"b{i}_ln2_g", b.ln2.gamma, b.ln2.dgamma, False),
                (f"b{i}_ln2_b", b.ln2.beta, b.ln2.dbeta, False),
            ]
            for j, h in enumerate(b.ma.heads):
                params += [
                    (f"b{i}_h{j}_wq", h.wq, h.dwq, True),
                    (f"b{i}_h{j}_wk", h.wk, h.dwk, True),
                    (f"b{i}_h{j}_wv", h.wV, h.dwV, True),
                ]

        for key, p, g, is_weight in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)

            # Update biased 1st and 2nd moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g ** 2)

            # Decoupled weight decay (AdamW) - applied to weights, not biases/norms
            if is_weight and self.weight_decay > 0:
                p -= self.lr * self.weight_decay * p

            # Adam update step
            p -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)

def save_model(model, folder="model"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "weights.npz")
    weights = {
        "embedding": model.embedding.embedding_matrix,
        "lm_head": model.lm_head.lm,
    }
    for i, block in enumerate(model.blocks):
        weights[f"b{i}_w1"] = block.ffwd.w1
        weights[f"b{i}_b1"] = block.ffwd.b1
        weights[f"b{i}_w2"] = block.ffwd.w2
        weights[f"b{i}_b2"] = block.ffwd.b2
        weights[f"b{i}_wo"] = block.ma.wo
        for j, head in enumerate(block.ma.heads):
            weights[f"b{i}_h{j}_wq"] = head.wq
            weights[f"b{i}_h{j}_wk"] = head.wk
            weights[f"b{i}_h{j}_wv"] = head.wV
        weights[f"b{i}_ln1_gamma"] = block.ln1.gamma
        weights[f"b{i}_ln1_beta"]  = block.ln1.beta
        weights[f"b{i}_ln2_gamma"] = block.ln2.gamma
        weights[f"b{i}_ln2_beta"]  = block.ln2.beta

    np.savez(filepath, **weights)
    print(f"\nModel weights saved to {filepath}")

optimizer = AdamW(lr=learning_rate)

for step in range(num_steps):
    X, Y = get_batch("train")
    # A. Forward pass
    logits = model.forward(X)
    # B. Compute loss & gradient
    loss, dlogits = compute_loss(logits, Y)
    # C. Backward pass (calculates .dw1, .dwq, etc.)
    model.backward(dlogits)
    # D. Update weights with AdamW
    optimizer.step(model)
    if step % 10 == 0 or step == num_steps - 1:
        print(f"Step {step} | Loss: {loss:.4f}")

save_model(model)