""" Lightweight inference wrapper for the demo application """

import logging
import torch

from core.preprocessors.tokenizers import CharacterLevelTokenizer
from core.models.transformer_decoder import TransformerDecoder
from core.models.bigram import BigramLanguageModel


class AlpineLLMInference:
    def __init__(self, cfg, hyperparam_cfg):
        self.cfg = cfg
        self.hyperparam_cfg = hyperparam_cfg
        self.device = torch.device(f"cuda:{self.cfg.cuda_id}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharacterLevelTokenizer()
        self.model = self.select_model()
        self.get_model(cfg.load_weights_path)

    def run_demo(self):
        """ Run a simple demo loop to generate text based on user input """
        while True:
            prompt = input("Enter a prompt (or 'exit' to quit): ")
            if prompt.lower() == 'exit':
                logging.info("Exiting the demo.")
                break
            generated_text = self.generate_text(prompt)
            logging.info(f"Generated Text:\n{generated_text}\n")

    @torch.no_grad()
    def generate_text(self, prompt):
        # tokenize input
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        # generate tokens
        output_ids = self.model.generate(input_ids, max_new_tokens=self.cfg.max_new_tokens)
        # decode to string
        return self.tokenizer.decode(output_ids[0].tolist())

    def select_model(self):
        """ Selects the neural network architecture based on the desired configuration """
        vocab_size = len(self.tokenizer.vocab)
        if self.cfg.model_type == 'transformer':
            model = TransformerDecoder(vocab_size=vocab_size, 
                                       hyperparam_cfg=self.hyperparam_cfg,
                                       device=self.device).to(self.device)
        elif self.cfg.model_type == 'bigram':
            model = BigramLanguageModel(vocab_size=vocab_size).to(self.device)
        else:
            raise ValueError(f"Model type '{self.cfg.model_type}' is not supported!")
        model_name = model.__class__.__name__
        logging.info(f'Selected model type: {self.cfg.model_type} with name: {model_name}')
        return model

    def get_model(self, model_path):
        """ Loads weights of the model from the specified path """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint[0], strict=False)
        logging.info(f'Restored model from: {model_path}')
