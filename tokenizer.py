from collections import Counter

# Load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Start with your original character vocabulary setup
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Initial mappings (characters to unique integer IDs)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Convert the entire training text into a sequence of starting integer IDs
train_ids = [stoi[c] for c in text]

# ==========================================
# BYTE PAIR ENCODING (BPE) TRAINING LOOP
# ==========================================

# Target vocabulary size (Initial unique chars + custom merged subwords)
# Adjust this value higher (e.g., 200, 500, or 1000) for more compression
target_vocab_size = 300 
num_merges = target_vocab_size - vocab_size

# Map of (token_id_1, token_id_2) -> new_merged_token_id
merges = {} 

def get_stats(ids):
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge_tokens(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair and ids[i+1] == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

print(f"Initial unique characters: {vocab_size}")
print(f"Training BPE Tokenizer up to a vocabulary size of {target_vocab_size}...")

# Iteratively find the most common pair and mint a new token ID for it
for i in range(num_merges):
    stats = get_stats(train_ids)
    if not stats:
        break
    
    top_pair = max(stats, key=stats.get)
    new_id = vocab_size + i
    
    # Store the merge rule
    merges[top_pair] = new_id
    
    # Update our lookups to unpack the two child strings
    itos[new_id] = itos[top_pair[0]] + itos[top_pair[1]]
    # (Optional) Update stoi for completeness
    stoi[itos[new_id]] = new_id
    
    # Compress the training dataset sequence
    train_ids = merge_tokens(train_ids, top_pair, new_id)

print("BPE Training complete!\n")

# ==========================================
# REPLACEMENT ENCODE & DECODE FUNCTIONS
# ==========================================

def encode(s):
    """Takes a string, transforms it into base character IDs, and applies BPE merges."""
    # Fallback to a safe character lookup to prevent crashes on completely unseen letters
    ids = [stoi.get(c, 0) for c in s] 
    for pair, new_id in merges.items():
        ids = merge_tokens(ids, pair, new_id)
    return ids

def decode(l):
    """Takes a list of integer IDs and glues their string pieces back together."""
    return ''.join([itos.get(i, '') for i in l])


# ==========================================
# YOUR TEST CODE (NOW RUNNING BPE)
# ==========================================

encode_text = encode("low key chill")

print(f"encoded string: {encode_text}")
print("===========")
print(f"decoded: {decode(encode_text)}")
