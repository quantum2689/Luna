#this code is the gpt.py and train.py transalted to pytorch using Antigravity
# why? because our trainig code works but it is to slow due to python is ver slow so we used pytorch to make it faster 
# the checkpoin trained using numpy is in old.npz 

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Hyperparameters (Identical to gpt.py / train.py)
# ==========================================
batch_size = 64        # independent sequences processed in parallel
block_size = 256       # maximum context length
n_embd = 384           # embedding dimension
n_head = 6             # number of attention heads
n_layer = 6            # number of transformer layers
dropout = 0.2          # dropout rate
learning_rate = 3e-4   # AdamW learning rate
num_steps = 5000        # number of training iterations
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# ==========================================
# 2. Tokenizer & Dataset Preparation
# ==========================================
try:
    from tokenizer import encode, decode, text
except ImportError:
    with open('/kaggle/input/datasets/intoai/input67/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

vocab_size = len(sorted(list(set(text))))
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 3. Model Architecture (Exact Replica of gpt.py)
# ==========================================

def get_sinusoidal_pe(block_size=256, n_embd=384, step_size=1.0491):
    """Generates the exact fixed positional encoding from postional_encoding.py."""
    if os.path.exists("postional_encoding_embd.npy"):
        pe = np.load("postional_encoding_embd.npy").astype(np.float32)
        return torch.tensor(pe, dtype=torch.float32)
    
    pe = []
    for pos in range(block_size):
        one_stamp = []
        n_val = 1.0
        for slot in range(n_embd):
            if slot % 2 == 0:
                one_stamp.append(math.sin(pos / n_val))
            else:
                one_stamp.append(math.cos(pos / n_val))
                n_val *= step_size
        pe.append(one_stamp)
    return torch.tensor(pe, dtype=torch.float32)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.inv_sqrt_hs = 1.0 / math.sqrt(self.head_size)

        # In gpt.py: Q, K, V have no bias and are multiplied as x @ W
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)  # wo

        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        # Linear projections & split into heads (B, n_head, T, head_size)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Dot-product attention identical to gpt.py
        att = (q @ k.transpose(-2, -1)) * self.inv_sqrt_hs
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-concatenate heads

        # Output projection
        return self.c_proj(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        # In gpt.py: w1 has shape (n_embd, 4*n_embd), w2 has shape (4*n_embd, n_embd) with ReLU
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ma = CausalMultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PyTorchGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.register_buffer("pe", get_sinusoidal_pe(block_size, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        # In gpt.py: lm_head has no bias and directly follows blocks without a final LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Standard normal * 0.02 initialization matching gpt.py
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pe[:T]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits


# ==========================================
# 4. Weight Saving in Exact NumPy .npz Format
# ==========================================
def save_model_to_npz(model, filepath="model/weights.npz"):
    """
    Saves PyTorch parameters into the NumPy .npz file format.
    Transposes weights where necessary because NumPy models compute `x @ W`
    while PyTorch computes `x @ W^T`.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    weights = {}

    # Token embeddings: (vocab_size, n_embd)
    weights["embedding"] = model.tok_emb.weight.detach().cpu().numpy().astype(np.float32)

    # LM Head: PyTorch (vocab_size, n_embd) -> Transposed to (n_embd, vocab_size)
    weights["lm_head"] = model.lm_head.weight.detach().cpu().numpy().T.astype(np.float32)

    for i, block in enumerate(model.blocks):
        # FeedForward weights & biases
        weights[f"b{i}_w1"] = block.ffwd.fc1.weight.detach().cpu().numpy().T.astype(np.float32)
        weights[f"b{i}_b1"] = block.ffwd.fc1.bias.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}_w2"] = block.ffwd.fc2.weight.detach().cpu().numpy().T.astype(np.float32)
        weights[f"b{i}_b2"] = block.ffwd.fc2.bias.detach().cpu().numpy().astype(np.float32)

        # Output projection wo
        weights[f"b{i}_wo"] = block.ma.c_proj.weight.detach().cpu().numpy().T.astype(np.float32)

        # Attention projections: Q, K, V
        wq = block.ma.q_proj.weight.detach().cpu().numpy().T.astype(np.float32)
        wk = block.ma.k_proj.weight.detach().cpu().numpy().T.astype(np.float32)
        wv = block.ma.v_proj.weight.detach().cpu().numpy().T.astype(np.float32)

        # Save merged matrices
        weights[f"b{i}_wq"] = wq
        weights[f"b{i}_wk"] = wk
        weights[f"b{i}_wv"] = wv

        # Also save per-head matrices for backward compatibility
        hs = block.ma.head_size
        for j in range(block.ma.n_head):
            weights[f"b{i}_h{j}_wq"] = wq[:, j * hs:(j + 1) * hs]
            weights[f"b{i}_h{j}_wk"] = wk[:, j * hs:(j + 1) * hs]
            weights[f"b{i}_h{j}_wv"] = wv[:, j * hs:(j + 1) * hs]

        # LayerNorm parameters: gamma (weight) and beta (bias)
        weights[f"b{i}_ln1_gamma"] = block.ln1.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}_ln1_beta"]  = block.ln1.bias.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}_ln2_gamma"] = block.ln2.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}_ln2_beta"]  = block.ln2.bias.detach().cpu().numpy().astype(np.float32)

    np.savez(filepath, **weights)
    print(f"\nModel weights saved to {filepath}")


# ==========================================
# 5. Training Loop (Exact Replica of train.py)
# ==========================================
model = PyTorchGPT(vocab_size, n_embd, n_head, n_layer, dropout).to(device)

# Weight decay setup identical to train.py: decay 2D weights, do not decay 1D biases / LayerNorms
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': 1e-2},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

print("Starting PyTorch training...")
model.train()
for step in range(num_steps):
    X, Y = get_batch("train")

    # A. Forward pass
    logits = model(X)

    # B. Compute loss (exact cross entropy)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

    # C. Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # D. Update weights with AdamW
    optimizer.step()

    if step % 10 == 0 or step == num_steps - 1:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# Save the trained weights to model/weights.npz
save_model_to_npz(model, filepath="model/weights.npz")