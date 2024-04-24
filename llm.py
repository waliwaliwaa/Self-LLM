import torch
import torch.nn as nn
import os
import requests
import tiktoken
import pandas as pd
import math

token_length= 16
d_model = 64
batch_size = 4
num_heads = 4
num_blocks = 8
dropout = 0.1
learning_rate = 1e-3  # 0.001
max_iters = 5000  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
TORCH_SEED = 1234
torch.manual_seed(TORCH_SEED)

with open('dataset.txt', 'r') as f:
    text = f.read()

encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)


train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validation_data = tokenized_text[train_index:]


class FeedForward(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features= self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.ffn(x)
    
class Attention(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.token_length = token_length
        self.dropout = dropout

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones((self.token_length, self.token_length))))  # Lower triangular mask
        
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # batch size, time steps, channels
        B, T, C = x.shape
        assert T <= self.token_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v= self.value_layer(x)

        # Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.token_length = token_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_length = token_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.ffn = FeedForward()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        x = x + self.multi_head_attention_layer(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x
    
class LLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_length = token_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        self.token_embedding_table = nn.Embedding(num_embeddings=self.max_token_value+1, embedding_dim=self.d_model)

        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] + 
            [nn.LayerNorm(self.d_model)]
        ))
        self.llm_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_table = torch.zeros(self.token_length, self.d_model)
        postion = torch.arange(0, self.token_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # [:, 0::2], ':' = selecting all lines, '0::2' = starting from 0, step by step 2
        position_encoding_table[:, 0::2] = torch.sin(postion * div_term)
        position_encoding_table[:, 1::2] = torch.sin(postion * div_term)
        position_encoding_table = position_encoding_table.unsqueeze(0).expand(batch_size, -1, -1)
        # [token_length, d_model] -> [T, d_model]
        position_embedding = position_encoding_table[:T, :].to(device)

        x = self.token_embedding_table(idx) + position_embedding
        x = self.transformer_blocks(x)

        logits = self.llm_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.token_length:]
            logits, loss = self(idx_crop)
            logits_last_timestep = logits[:, -1, :]
            probs = torch.nn.functional.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

model = LLM()
model = model.to(device=device)

def get_batch(split: str):
    data = train_data if split == 'train' else validation_data
    idxs = torch.randint(low=0, high=len(data) - token_length, size=(batch_size, ))
    x = torch.stack([data[idx: idx + token_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1: idx + token_length + 1] for idx in idxs]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')