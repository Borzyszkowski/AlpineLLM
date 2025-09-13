"""" Load preprocessed data """

import argparse
import glob
import numpy as np
import os

import torch
from torch.utils import data


class DataloaderBase(data.Dataset):
    def __init__(self, dataset_dir, data_split='train'):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split_name = data_split
        self.sequence_paths = glob.glob(os.path.join(dataset_dir, data_split, '*/*.pt'))

    def load(self, datasets):
        """ Load an actual file """
        for d in datasets:
            k = os.path.basename(d)
            loaded = torch.load(d, weights_only=True)

            # TODO: MOCK This assumes data is not sequential and loads only one element
            loaded = list(loaded.values())[0]
            return loaded

    def load_sample(self, idx):
        """ Load a single data sequence """
        sample_path = glob.glob(self.sequence_paths[idx])
        input_tensor = self.load(sample_path)

        # Get label and path of a sequence
        label_file = glob.glob(os.path.dirname(self.sequence_paths[idx]) + '/label.npz')
        path_file = glob.glob(os.path.dirname(self.sequence_paths[idx]) + '/seq_path.txt')

        label = np.load(label_file[0])["label"].tolist()
        path = open(path_file[0], 'r').read() if path_file else "unknown_path"

        return input_tensor, label, path
    
    def __len__(self):
        """ Return number of samples in the entire dataset """
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        """ Get single data item from the dataset as a pytorch tensor and return it """
        data_item, label, path = self.load_sample(idx)
        return data_item, label, path


def test_dataloader(data_dir, split_name):
    """ 
    This is a test code to load example data. 
    Normally the dataloadr is called from the training / infrerence program.
    """
    from torch.utils.data import DataLoader

    print("Dataloder test function...")
    kwargs = {'num_workers': 10,
              'batch_size': 1,
              'shuffle': True,
              'drop_last': False
              }

    print(f'Loading data from: {data_dir} for split {split_name}')
    dataset = DataloaderBase(data_dir, split_name)
    dataset = DataLoader(dataset, **kwargs)

    for (it, batch) in enumerate(dataset, 0):
        if it == 0:
            print(f"Data for iteration {it}:")
            print(batch)

    print(f'Dataset size: {len(dataset.dataset)}')
    print(f'Data loading finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data-Loader-Test')
    parser.add_argument('--data-dir', required=True, type=str, 
                        help='Absolute path to the preprocessed dataset organized in train/test/val directories. \
                              Data has to be stored as tensors with sequences in PyTorch .pt format.')
    parser.add_argument('--split-name', required=True, choices=['train', 'test', 'val'], type=str,
                        help='Name of data split')
    args = parser.parse_args()
    data_dir = args.data_dir
    split_name = args.split_name
    test_dataloader(data_dir, split_name)
