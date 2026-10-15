import os
import numpy as np
from gpt import GPT, softmax, vocab_size, n_embd, n_head, n_layer, block_size
from tokenizer import encode, decode


def load_model(model, filepath="model/weights.npz"):
    """Loads saved numpy weights into the GPT model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Weights file not found at '{filepath}'. Please run 'train.py' first to train and save the model!"
        )

    data = np.load(filepath)
    model.embedding.embedding_matrix = data["embedding"]
    model.lm_head.lm = data["lm_head"]

    for i, block in enumerate(model.blocks):
        block.ffwd.w1 = data[f"b{i}_w1"]
        block.ffwd.b1 = data[f"b{i}_b1"]
        block.ffwd.w2 = data[f"b{i}_w2"]
        block.ffwd.b2 = data[f"b{i}_b2"]
        block.ma.wo   = data[f"b{i}_wo"]

        for j, head in enumerate(block.ma.heads):
            head.wq = data[f"b{i}_h{j}_wq"]
            head.wk = data[f"b{i}_h{j}_wk"]
            head.wV = data[f"b{i}_h{j}_wv"]

        block.ln1.gamma = data[f"b{i}_ln1_gamma"]
        block.ln1.beta  = data[f"b{i}_ln1_beta"]
        block.ln2.gamma = data[f"b{i}_ln2_gamma"]
        block.ln2.beta  = data[f"b{i}_ln2_beta"]

    print(f"Successfully loaded model weights from '{filepath}'.")


def generate(model, prompt="", max_new_tokens=100, temperature=0.8, top_k=None):
    """
    Generates text from a starting prompt using the trained model.
    Tokens are streamed to the console as they are produced.
    """
    tokens = encode(prompt)
    if len(tokens) == 0:
        # Default starting token if prompt is empty
        tokens = [0]

    idx = np.array([tokens], dtype=np.int64)  # shape (1, T)

    for _ in range(max_new_tokens):
        # Crop context to the maximum context length (block_size)
        idx_cond = idx[:, -block_size:]

        # Forward pass to get logits for all tokens
        logits = model.forward(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Optional top-k sampling
        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits.shape[-1])
            cutoff = np.partition(logits, -top_k_val, axis=-1)[:, -top_k_val]
            logits[logits < cutoff[:, None]] = -np.inf

        # Softmax probabilities
        probs = softmax(logits, axis=-1)[0]

        # Sample the next token according to the probability distribution
        next_token = np.random.choice(vocab_size, p=probs)

        # Append next token to the running sequence
        idx = np.concatenate([idx, [[next_token]]], axis=1)

        # Print the character live
        print(decode([next_token]), end="", flush=True)

    print()
    return decode(idx[0].tolist())


if __name__ == "__main__":
    print("=" * 50)
    print("Initializing Luna Model...")
    print("=" * 50)

    # Initialize model in evaluation mode (training=False disables dropout)
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        training=False
    )

    # Load trained weights
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
