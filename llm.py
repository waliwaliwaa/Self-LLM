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
dropout = 0.1


with open('dataset.txt', 'r') as f:
    text = f.read()

encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
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

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_sizem, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        
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


