""" Collection of tokenizers for text data. """

import string


class CharacterLevelTokenizer:
    """ A simple character-level tokenizer for text data. """

    def __init__(self):
        self.vocab = sorted(set(string.ascii_letters + string.digits + string.punctuation + " \n"))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, str_input):
        """ encoder: take a string, output a list of integers """
        return [self.token_to_id[token] for token in str_input]

    def decode(self, token_ids):
        """ decoder: take a list of integers, output a string """
        return ''.join([self.id_to_token[token_id] for token_id in token_ids])
