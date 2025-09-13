"""" Preprocess raw data for deep learning experiments """

import glob
import json
import logging
import numpy as np
import os
import random
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.utils.utils import makepath
from core.preprocessors.preprocessing_utils import np2torch, DotDict


class PreprocessorAlpine:
    def __init__(self, cfg):
        self.inp_path = cfg.inp_path
        self.out_path = cfg.out_path
        makepath(self.out_path)

        logging.info('Starting data preprocessing!')

        # split data into train, test, val subsets
        self.all_seqs = glob.glob(self.inp_path + '/*.json')
        splits = self.split_data(sequences=self.all_seqs)

        # preprocess the data
        for split_name, split_sequences in splits.items():
            logging.info(f'Processing data for the {split_name} split')
            self.data_preprocessing(cfg, split_name, split_sequences)
            logging.info(f'Processing for the {split_name} split finished!')

        logging.info(f'Splitted: {len(splits["train"])} train - {len(splits["test"])} test - {len(splits["val"])} val')
        logging.info(f"Data preprocessed and exported to the location: {self.out_path}")

    def data_preprocessing(self, cfg, split_name, split_sequences):
        """ Processes sequences of data for the given split """
        for idx, sequence in enumerate(tqdm(split_sequences)):
            logging.debug(f"Processing sequence {idx}: {sequence}")

            # read all lines of a sequence from a JSON file into a list
            with open(sequence, 'r') as file:
                content = file.readlines()

            # convert each line from JSON string to dictionary
            seq_data = [json.loads(line) for line in content if line.strip()]
            if len(seq_data) == 0:
                logging.warning(f"Skipped empty JSON file for the sequence ID {idx} with path {sequence}")
                continue

            # run the trace scheduler for a single sequence
            seq_data = self.extract_features(seq_data)

            # data has to be kept in np.float32 arrays as values
            try:
                for key, value in seq_data.items():
                    try:
                        seq_data[key] = np.array(value, dtype=np.float32)
                    except Exception as e:
                        logging.error(f"Error converting value for key {key}: {value}, Error: {e}")
                        raise e 
            except Exception as e:
                logging.error(f"Skipping sequence {idx} due to an error: {e}")
                logging.error(f"Full sequence name: {sequence}")
                continue

            # sequence has to be kept as a DotDict with np.float32 arrays as values
            seq_data = {key: np.array(value, dtype=np.float32) for key, value in seq_data.items()}
            seq_data = DotDict(seq_data)

            # make sure that all features have equal shape
            seq_feat_lens = [value.shape[0] for value in seq_data.values()]
            assert all(shape == seq_feat_lens[0] for shape in seq_feat_lens), f"Not all values have the same shape in seq {sequence}"

            # save the results
            out_dir = f"{idx:04d}_{os.path.basename(os.path.normpath(sequence)).split('.', 1)[0]}"
            seq_out_dir = os.path.join(self.out_path, split_name, out_dir)
            for data_name, data in seq_data.items():
                data = np2torch(item={data_name: data})
                outfname = makepath(os.path.join(seq_out_dir, '%s.pt' % data_name), isfile=True)
                torch.save(data, outfname)
            np.savez(os.path.join(seq_out_dir, 'frame_ids.npz'), frame_ids=list(seq_data.keys()))
            
            # save the label file
            # TODO: MOCK implement a way to extract labels
            label = random.randint(0, 1)  # Placeholder for label extraction logic
            np.savez(os.path.join(seq_out_dir, 'label.npz'), label=label)
            with open(os.path.join(seq_out_dir, 'label.txt'), "w") as f: f.write(str(label))

            # write auxiliary information
            np.savez(os.path.join(seq_out_dir, "seq_path.npz"), seq_path=sequence)
            with open(os.path.join(seq_out_dir, "seq_path.txt"), "w") as f: f.write(sequence)
            
    def split_data(self, sequences, train_size=0.8, test_size=0.1, val_size=0.1):
        """ Splits data into train/test/val sets """
        assert int(round(train_size + test_size + val_size)) == 1, 'Wrong train/test/val ratio!'
        logging.info(f'Number of all sequences: {len(sequences)}')
        logging.info(f'Splitting data into train/test/val with ratio {train_size}/{test_size}/{val_size}')

        splits = {'train': [], 'test': [], 'val': []}
        train, test = train_test_split(sequences, test_size=test_size)
        train, val = train_test_split(train, test_size=val_size)
        splits['train'].extend(train)
        splits['test'].extend(test)
        splits['val'].extend(val)

        logging.info(f'Splitted: {len(splits["train"])} train - {len(splits["test"])} test - {len(splits["val"])} val')
        return splits

    def extract_features(self, json_objects):
        """ 
        Extract desired features from the JSON objects and return them as a dictionary.
        """
        extracted_sequence = {}
        for json_object in json_objects:
            logging.debug(f"Extracting features from the JSON {json_object}")

            # Validate input type
            if not isinstance(json_object, dict):
                logging.error("Incoming frame is not a valid dictionary.")
                continue

            frame_id = json_object.get('frame_id')
            if not frame_id:
                logging.error("Frame ID is missing or invalid.")
                continue

            # TODO: MOCK add actual feature extraction logic here
            extracted_sequence[frame_id] = range(96)
        return extracted_sequence
