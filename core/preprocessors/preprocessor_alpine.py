"""" Preprocess raw Alpine data for deep learning experiments """

import glob
import json
import logging
import numpy as np
import os
import random
import torch
import unicodedata

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.utils.utils import makepath
from core.preprocessors.preprocessing_utils import np2torch, DotDict
from core.preprocessors.tokenizers import CharacterLevelTokenizer


class PreprocessorAlpine:
    def __init__(self, cfg):
        self.inp_path = cfg.inp_path
        self.out_path = cfg.out_path
        makepath(self.out_path)

        # read and preprocess the data file
        input_files = glob.glob(os.path.join(self.inp_path, '*.txt'))
        concat_file = self.concat_files(input_files)
        logging.info(f'Starting data preprocessing for the file: {concat_file}')
        data = self.data_preprocessing(concat_file)
        
        # split data into train, test, val subsets
        splits = self.split_data(data)

        # serialize the data splits into .pt files on disk
        self.export_data(splits)

        logging.info(f"Data preprocessed and exported to the location: {self.out_path}")

    def concat_files(self, input_files):
        """ Clean and concat all text files into a single file """
        concat_file = os.path.join(self.out_path, 'consolidated_data.txt')
        logging.info(f'Consolidating {len(input_files)} text files into a single file: {concat_file}')
        with open(concat_file, 'w', encoding='utf-8') as outfile:
            for fname in input_files:
                with open(fname, 'r', encoding='utf-8-sig') as infile:
                    text = infile.read()
                    text = self.normalize_text(text)
                    outfile.write(text + "\n")
        return concat_file

    def normalize_text(self, text):
        """ Normalizes raw text by removing non-ASCII characters """
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in text if not unicodedata.combining(c)])
        # Replace common “smart quotes” with ASCII equivalents
        text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
        # Encode to ASCII, ignore errors, then back to str
        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def data_preprocessing(self, file_path):
        """ Processes a single text file with data """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"Length of dataset in characters: {len(text)}")
        logging.debug(f"The first 1000 characters: \n {text[:1000]}")

        # very simple character-level tokenizer
        tokenizer = CharacterLevelTokenizer()
        logging.info(f"Vocabulary size: {len(tokenizer.vocab)}")
        logging.debug(f"All the unique characters: {''.join(tokenizer.vocab)}")

        # encode the entire text dataset and store it into a torch.Tensor
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        logging.info(f"Dataset shape: {data.shape} & Dataset type {data.dtype}")
        logging.debug(f"The first 1000 characters tokenized: \n {data[:1000]}")
        return data
            
    def split_data(self, data, train_size=0.8, test_size=0.1, val_size=0.1):
        """ Splits data into train/test/val sets """
        assert int(round(train_size + test_size + val_size)) == 1, 'Wrong train/test/val ratio!'
        logging.info(f'Number of all data samples (tokens): {len(data)}')
        logging.info(f'Splitting data into train/test/val with ratio {train_size}/{test_size}/{val_size}')

        # do not shuffle the data to keep consistency of text
        n = len(data)
        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)
        
        splits = {
            'train': data[:train_end],
            'val': data[train_end:val_end],
            'test': data[val_end:]
        }

        logging.info(f'Splitted: {len(splits["train"])} train - {len(splits["test"])} test - {len(splits["val"])} val')
        return splits

    def export_data(self, splits):
        """ Exports the processed data for a each split as .pt file """
        for split_name, split_data in splits.items():
            logging.info(f'Exporting data for the split: {split_name}')
            out_file = os.path.join(self.out_path, f'{split_name}', 'data_tensor.pt')
            makepath(os.path.dirname(out_file))
            torch.save(split_data, out_file)
            logging.info(f'Exported {split_name} samples to {out_file}')
