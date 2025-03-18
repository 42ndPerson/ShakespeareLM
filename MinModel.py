import mlx.core as mx
import mlx.nn as nn
import mlx.utils as xu
import mlx.optimizers as optim
import os

# Constants
# Data
val_size = 0.1
# Model
block_size = 64
embd_dims = 96
num_layers = 4
num_heads = 4
ffwd_exp = 4
# Training
batch_size = 16
max_iters = 10000
eval_interval = 250
learning_rate = 1e-3
eval_iters = 200
dropout = 0.2

# Random
mx.random.seed(1042)

# Data
text = open('input.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encode-Decode
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[int(i)] for i in l])
# Encode Text
data = mx.array(encode(text))

# Train-Val Split
n = int((1 - val_size)*len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(low=0, high=len(data) - block_size - 2, shape=(batch_size,)) # Is the -2 necessary
    x = mx.array([data[i.item():i.item()+block_size] for i in ix])
    y = mx.array([data[i.item()+1:i.item()+block_size+1] for i in ix])

    return x, y

# Model Structure
class EmbeddingTable(nn.Module):
    def __init__(self, vocab_size: int, embd_dim: int):
        super().__init__()

        self.table = mx.random.uniform((vocab_size, embd_dim))

    def __call__(self, x):
        return self.table[x]

class FeedForward(nn.Module):
    def __init__(self, embd_dims: int, dropout: float):
        super().__init__()

        self.lin1 = nn.Linear(
            input_dims=embd_dims, 
            output_dims=ffwd_exp*embd_dims
        )
        self.lin2 = nn.Linear(
            input_dims=ffwd_exp*embd_dims, 
            output_dims=embd_dims
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        y = self.lin1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.lin2(y)

        return y

class Block(nn.Module):
    def __init__(self, embd_dims: int, num_heads: int, dropout: float):
        super().__init__()

        head_size = (embd_dims // num_heads) * num_heads
        self.sa = nn.MultiHeadAttention(
            dims=embd_dims, 
            num_heads=num_heads,
            value_dims=head_size
        )
        self.ffwd = FeedForward(embd_dims=embd_dims, dropout=dropout)
        self.ln1 = nn.LayerNorm(dims=embd_dims)
        self.ln2 = nn.LayerNorm(dims=embd_dims)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y = self.sa(y, y, y, mask=mask)
        x = x + y

        x = self.dropout(x)
        x = x + self.ffwd(self.ln2(x))

        return x

class GPT_LM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embd_dims: int,
        vocab_size: int, 
        num_heads: int, 
        dropout: float
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            dims=embd_dims
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            dims=embd_dims
        )
        self.layers = [
            Block(
                embd_dims=embd_dims,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range (num_layers)
        ]
        self.norm = nn.LayerNorm(dims=embd_dims)
        self.out_proj = nn.Linear(
            input_dims=embd_dims,
            output_dims=vocab_size,
            bias=False
        )

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)

        tok_embd = self.token_embedding(x)
        pos_embd = self.position_embedding(mx.arange(0,x.shape[1],1,dtype=mx.int32))
        x = tok_embd + pos_embd
        for l in self.layers:
            x = l(x, mask)
        x = self.norm(x)
        logits = self.out_proj(x)

        return logits

    def generate(self, x, temp=1):
        ctx = x[None, -block_size:]

        while True:
            full_pred = self(ctx) # Predict
            next_pred_dist = full_pred[:,-1,:] # Get last token
            next_pred_char = mx.random.categorical(next_pred_dist[0] * (1/temp)) # Select
            ctx = mx.concatenate([ctx, next_pred_char[None,None]], axis=1) # Add to ctx
            ctx = ctx[:, -block_size:] # Reselect ctx

            yield next_pred_char.item()

        '''
        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[0])
        mask = mask.astype(self.token_embedding.weight.dtype)

        # First we process the prompt x the same way as in __call__ but
        # save the caches in cache
        tok_embd = self.token_embedding(x)
        pos_embd = self.position_embedding(mx.arange(0,x.shape[0],1,dtype=mx.int32))
        x = tok_embd + pos_embd
        x = x[None,:,:]
        for l in self.layers:
            x = l(x, mask=mask)
        x = self.norm(x)
        y = self.out_proj(x[:, -1])  # <--- we only care about the last logits
                                     #      that generate the next token
        y = mx.random.categorical(y * (1/temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]
            
            tok_embd = self.token_embedding(x)
            pos_embd = self.position_embedding(mx.arange(0,x.shape[0],1,dtype=mx.int32))
            x = tok_embd + pos_embd
            for i in range(len(self.layers)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x = self.layers[i](x, mask=None)
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1/temp))

            yield y
        '''

# Loss Calc
def calc_loss(model, X, targets):
    logits = model(X)

    B, L, V = logits.shape
    loss = nn.losses.cross_entropy(logits, targets)
    return loss.mean()

# Estimate Loss
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = mx.zeros((eval_iters))
        for k in range(eval_iters):
            X, Y = get_batch(split)
            loss = calc_loss(model, X, Y)
            losses[k] = loss.mean()
        out[split] = losses.mean()
    model.train()

    return out

# Model Init
model = GPT_LM(
    num_layers=num_layers,
    embd_dims=embd_dims,
    vocab_size=vocab_size,
    num_heads=num_heads,
    dropout=dropout
)
if os.path.exists('weights.npz'):
    model.load_weights('weights.npz', strict=True)
num_params = sum(v.size for k,v in xu.tree_flatten(model.parameters()))
print(f'{num_params / 1e6} M Parameters')

# Training
optimizer = optim.AdamW(learning_rate=learning_rate)
model.train()
vg_fn = mx.value_and_grad(calc_loss)
for iter in range(max_iters):
    # Progress Printout
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample
    xb, yb = get_batch('train')

    # Evaluate the loss
    loss, grads = vg_fn(model, xb, yb)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
model.eval()

# Save
model.save_weights('weights.npz')
print("Saved")

# Generate
context = mx.zeros((1), dtype=mx.int32)
gen = model.generate(context)
for _ in range(1000):
    print(decode([next(gen)]), end="")