""" Architecture of the TransformerDecoder """

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.utils.time_exec_utils import log_execution_time


class TransformerDecoder(nn.Module):
    """ GPT-style decoder-only language model """

    def __init__(self, vocab_size, context_len, device, embedding_dim=32):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.context_len = context_len
        # lookup table of tokens is used so that each token reads the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        # pos embedding table adds information about the position of each token in the context
        self.pos_embedding_table = nn.Embedding(context_len, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    @log_execution_time
    def forward(self, idx):
        """ 
        The forward pass of the model returns the logits of shape (B,T,C)
        # where: B=batch_size T=context_len C=vocab_size
        """
        # idx is a (B,T) tensor of integers which are indices in the current context
        B, T = idx.shape 
        token_embd = self.token_embedding_table(idx)      # (batch_size, context_len, embedding_dim)
        positions = torch.arange(T).to(self.device)       # tensor([0, 1, 2, ..., T-1])
        pos_embd = self.pos_embedding_table(positions)    # (context_len, embedding_dim)
        x = token_embd + pos_embd                         # (batch_size, context_len, embedding_dim)
        logits = self.lm_head(x)                          # (batch_size, context_len, vocab_size)
        return logits

    @log_execution_time
    def generate(self, idx, max_new_tokens):
        """ Generate new tokens from the model """
        for _ in range(max_new_tokens):
            # crop idx to the last context_len tokens
            idx_context = idx[:, -self.context_len:]
            # get the predictions
            logits = self(idx_context) # (B,T,C) 
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution to get the next token index
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
