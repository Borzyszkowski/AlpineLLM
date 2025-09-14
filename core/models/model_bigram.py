""" Architectures of the neural network """

import torch
import torch.nn as nn

from core.utils.time_exec_utils import log_execution_time


class BigramLanguageModel(nn.Module):
    """ A simple bigram language model as a baseline """

    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # lookup table of tokens is used so that each token reads the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    @log_execution_time
    def forward(self, idx):
        """ 
        The forward pass of the model returns the logits of shape (B,T,C)
        # where: B=batch_size T=context_len C=vocab_size
        """
        # idx is a (B,T) tensor of integers which are indices in the current context
        logits = self.token_embedding_table(idx) # (B,T,C)
        return logits

    @log_execution_time
    def generate(self, idx, max_new_tokens):
        """ Generate new tokens from the model """
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx) # (B,T,C) 
            # focus only on the last time step since the Bigram model is stateless
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution to get the next token index
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
