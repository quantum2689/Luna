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



def update_weights(model, lr):
    # Embedding & LM Head
    model.embedding.embedding_matrix -= lr * model.embedding.dE
    model.lm_head.lm                 -= lr * model.lm_head.dlm
    # Transformer Blocks
    for block in model.blocks:
        # Feed-forward network
        block.ffwd.w1 -= lr * block.ffwd.dw1
        block.ffwd.b1 -= lr * block.ffwd.db1
        block.ffwd.w2 -= lr * block.ffwd.dw2
        block.ffwd.b2 -= lr * block.ffwd.db2
        # Attention projection
        block.ma.wo   -= lr * block.ma.dwo
        # Individual attention heads
        for head in block.ma.heads:
            head.wq -= lr * head.dwq
            head.wk -= lr * head.dwk
            head.wV -= lr * head.dwV
        # Layer norms
        block.ln1.gamma -= lr * block.ln1.dgamma
        block.ln1.beta  -= lr * block.ln1.dbeta
        block.ln2.gamma -= lr * block.ln2.dgamma
        block.ln2.beta  -= lr * block.ln2.dbeta

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

for step in range(num_steps):
    X, Y = get_batch("train")
    # A. Forward pass
    logits = model.forward(X)
    # B. Compute loss & gradient
    loss, dlogits = compute_loss(logits, Y)
    # C. Backward pass (calculates .dw1, .dwq, etc.)
    model.backward(dlogits)
    # D. Update weights directly
    update_weights(model, learning_rate)
    if step % 10 == 0 or step == num_steps - 1:
        print(f"Step {step} | Loss: {loss:.4f}")

save_model(model)