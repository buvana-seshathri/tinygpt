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
block_size = 32  # how many characters of context we look at (bigger now - attention can use it)
batch_size = 32  # how many sequences we train on at once
n_embd = 64      # size of each character's embedding vector
n_layer = 4      # how many transformer blocks to stack
n_head = 4       # how many attention heads per block

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ---- 3. Self-attention head, from scratch ----
class Head(nn.Module):
    """One self-attention head. Turns each position's embedding into
    a new embedding that's a weighted average of ALL earlier positions
    (including itself) - where the weights are learned, not fixed."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask: position i can only look at positions <= i
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size) - "what I contain"
        q = self.query(x)  # (B, T, head_size) - "what I'm looking for"
        v = self.value(x)  # (B, T, head_size) - "what I'll share if picked"

        # how much does each position "want" to attend to each other position
        scores = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T), scaled so softmax isn't too peaky
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # block future positions
        weights = F.softmax(scores, dim=-1)               # turn scores into a probability distribution

        out = weights @ v   # (B, T, head_size) - weighted average of Values
        return out

# ---- 4. Multiple heads in parallel, then combine ----
class MultiHeadAttention(nn.Module):
    """Runs several attention heads side by side, each with a smaller
    head_size, then concatenates their outputs back to n_embd size.
    Each head is free to learn a different pattern - e.g. one might
    focus on the previous vowel, another on sentence starts, etc."""

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # mixes the heads' outputs back together

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.proj(out)

# ---- 5. Feed-forward: lets the model "think" per position ----
class FeedForward(nn.Module):
    """Attention gathers information from other positions.
    This is where the model processes that information - a plain
    2-layer MLP applied independently to each position."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

# ---- 6. Transformer block: attention + feedforward + residuals + norm ----
class Block(nn.Module):
    """One transformer block. The 'x = x +' pattern is a residual
    connection: the sublayer only has to learn what to ADD to x,
    which makes training much easier than learning a full replacement."""

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # attend, then add back to x
        x = x + self.ffwd(self.ln2(x))   # think, then add back to x
        return x

# ---- 7. Full model: embeddings + stacked transformer blocks + output layer ----
class AttentionModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                          # (B, T, n_embd) - "what character is this"
        pos_emb = self.position_embedding(torch.arange(T))           # (T, n_embd)    - "where in the sequence"
        x = tok_emb + pos_emb                                        # combine identity + position
        x = self.blocks(x)                                           # attend + think, n_layer times
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # attention only has room for block_size positions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# ---- 8. Estimate loss (averaged over several batches, no gradient tracking) ----
eval_iters = 100

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---- 9. Train ----
model = AttentionModel(vocab_size, n_embd, block_size, n_head, n_layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

steps = 5000
for step in range(steps):
    if step % 500 == 0 or step == steps - 1:
        losses = estimate_loss(model)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---- 10. Generate sample text ----
context = torch.zeros((1, 1), dtype=torch.long)  # start with newline char
generated = model.generate(context, max_new_tokens=500)
print("\n--- sample output ---")
print(decode(generated[0].tolist()))
