import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
block_size = 256       # maximum context length
n_embd = 384           # embedding dimension
n_head = 6             # number of attention heads
n_layer = 6            # number of transformer layers
dropout = 0.2          # dropout (set to 0 during eval)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. Tokenizer
# ==========================================
try:
    from tokenizer import encode, decode, vocab_size
except ImportError:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])


# ==========================================
# 3. Model Architecture
# ==========================================
def get_sinusoidal_pe(block_size=256, n_embd=384, step_size=1.0491):
    """Generates the fixed positional encoding from postional_encoding.py."""
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
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.inv_sqrt_hs = 1.0 / math.sqrt(self.head_size)

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.inv_sqrt_hs
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
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
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.register_buffer("pe", get_sinusoidal_pe(block_size, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pe[:T]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits


# ==========================================
# 4. Load Model from .npz Weights File
# ==========================================
def load_model(model, filepath="model/weights.npz"):
    """
    Loads weights from the NumPy .npz archive into the PyTorch GPT model.
    Transposes weights where appropriate (NumPy x @ W vs PyTorch x @ W^T).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Weights file not found at '{filepath}'. Please train and save weights first!"
        )

    data = np.load(filepath)
    with torch.no_grad():
        # Embedding: (vocab_size, n_embd)
        model.tok_emb.weight.copy_(torch.from_numpy(data["embedding"]))

        # LM Head: stored as (n_embd, vocab_size) in npz -> transpose to (vocab_size, n_embd)
        model.lm_head.weight.copy_(torch.from_numpy(data["lm_head"].T))

        for i, block in enumerate(model.blocks):
            # FeedForward weights & biases
            block.ffwd.fc1.weight.copy_(torch.from_numpy(data[f"b{i}_w1"].T))
            block.ffwd.fc1.bias.copy_(torch.from_numpy(data[f"b{i}_b1"]))
            block.ffwd.fc2.weight.copy_(torch.from_numpy(data[f"b{i}_w2"].T))
            block.ffwd.fc2.bias.copy_(torch.from_numpy(data[f"b{i}_b2"]))

            # Attention output projection wo
            block.ma.c_proj.weight.copy_(torch.from_numpy(data[f"b{i}_wo"].T))

            # Attention Q, K, V projections (supports both merged and per-head npz saves)
            if f"b{i}_wq" in data:
                block.ma.q_proj.weight.copy_(torch.from_numpy(data[f"b{i}_wq"].T))
                block.ma.k_proj.weight.copy_(torch.from_numpy(data[f"b{i}_wk"].T))
                block.ma.v_proj.weight.copy_(torch.from_numpy(data[f"b{i}_wv"].T))
            else:
                wq = np.concatenate([data[f"b{i}_h{j}_wq"] for j in range(n_head)], axis=1)
                wk = np.concatenate([data[f"b{i}_h{j}_wk"] for j in range(n_head)], axis=1)
                wv = np.concatenate([data[f"b{i}_h{j}_wv"] for j in range(n_head)], axis=1)
                block.ma.q_proj.weight.copy_(torch.from_numpy(wq.T))
                block.ma.k_proj.weight.copy_(torch.from_numpy(wk.T))
                block.ma.v_proj.weight.copy_(torch.from_numpy(wv.T))

            # LayerNorms
            block.ln1.weight.copy_(torch.from_numpy(data[f"b{i}_ln1_gamma"]))
            block.ln1.bias.copy_(torch.from_numpy(data[f"b{i}_ln1_beta"]))
            block.ln2.weight.copy_(torch.from_numpy(data[f"b{i}_ln2_gamma"]))
            block.ln2.bias.copy_(torch.from_numpy(data[f"b{i}_ln2_beta"]))

    print(f"Successfully loaded model weights from '{filepath}'.")


# ==========================================
# 5. Text Generation Function
# ==========================================
@torch.no_grad()
def generate(model, prompt="", max_new_tokens=100, temperature=0.8, top_k=20):
    """
    Generates text from a starting prompt and streams characters live to the console.
    """
    model.eval()
    tokens = encode(prompt)
    if len(tokens) == 0:
        tokens = [0]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to maximum context length
        idx_cond = idx[:, -block_size:]

        # Forward pass to obtain logits
        logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Top-k sampling filtering
        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_val)
            logits[logits < v[:, [-1]]] = float('-inf')

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append next token to running sequence
        idx = torch.cat((idx, next_token), dim=1)

        # Stream character live
        char = decode([next_token.item()])
        print(char, end="", flush=True)

    print()
    return decode(idx[0].tolist())


# ==========================================
# 6. Interactive Chat Interface
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("Initializing Luna Model (PyTorch Engine)...")
    print(f"Running on: {device.upper()}")
    print("=" * 50)

    # Initialize model in evaluation mode
    model = PyTorchGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.0
    ).to(device)

    # Load trained weights from model/weights.npz
    load_model(model, filepath="model/weights.npz")

    print("\nModel ready! Type your message to chat with Luna (type 'quit' or 'exit' to stop).\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        print("Luna: ", end="", flush=True)
        generate(model, prompt=user_input, max_new_tokens=100, temperature=0.8, top_k=20)
