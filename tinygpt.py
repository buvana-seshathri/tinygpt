"""
Tiny GPT - Phase 1: Tokenizer + Bigram Baseline
--------------------------------------------------
Goal of this phase: get the full pipeline working end to end
(load text -> encode -> train a dumb model -> generate text)
before we add any attention. This is our sanity check.

A bigram model just learns "given this character, what's the
next character likely to be?" - no context beyond 1 character.
It'll generate garbage, but if garbage looks vaguely Shakespeare-ish
(right character frequencies, occasional real words), the pipeline works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---- 1. Load and tokenize data ----
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab size: {vocab_size} characters")

# character <-> integer lookup
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# 90/10 train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---- 2. Batching ----
block_size = 8   # how many characters of context we look at
batch_size = 32  # how many sequences we train on at once

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ---- 3. Bigram model ----
class BigramModel(nn.Module):
    """Each character directly predicts the next character's
    probabilities via a lookup table. No attention, no context window
    beyond the current character. This is our baseline."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)  # (batch, time, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # only care about last time step
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# ---- 4. Train ----
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

steps = 3000
for step in range(steps):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# ---- 5. Generate sample text ----
context = torch.zeros((1, 1), dtype=torch.long)  # start with newline char
generated = model.generate(context, max_new_tokens=300)
print("\n--- sample output ---")
print(decode(generated[0].tolist()))
