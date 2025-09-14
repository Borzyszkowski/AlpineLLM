"""" Load preprocessed data """

import argparse
import os

import torch
from torch.utils import data


class DataloaderLLM(data.Dataset):
    def __init__(self, dataset_dir, data_split='train', context_len=8):
        super().__init__()
        self.context_len = context_len
        self.data = torch.load(os.path.join(dataset_dir, data_split, 'data_tensor.pt'), weights_only=True)
    
    def __len__(self):
        """ Return number of samples in the entire dataset """
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        """ Get single data item from the dataset as a pytorch tensor and return it """
        x = self.data[idx:idx + self.context_len]           # input sequence
        y = self.data[idx + 1:idx + self.context_len + 1]   # target sequence (input shifted by 1 for completion)
        return x, y


def test_dataloader(data_dir, split_name):
    """ 
    This is a test code to load example data. 
    Normally the dataloadr is called from the training / infrerence program.
    """
    from torch.utils.data import DataLoader

    print("Dataloder test function...")
    context_length = 8
    kwargs = {'num_workers': 10,
              'batch_size': 2,
              'shuffle': True,
              'drop_last': False
              }

    print(f'Loading data from: {data_dir} for split {split_name}')
    dataset = DataloaderLLM(data_dir, split_name, context_length)
    dataset = DataLoader(dataset, **kwargs)

    for (it, batch) in enumerate(dataset, 0):
        if it == 0:
            inputs, targets = batch
            print(f"Data for iteration {it}:")
            print('inputs:')
            print(inputs.shape)
            print(inputs)
            print('targets:')
            print(targets.shape)
            print(targets)
        else:
            break

    print(f'Dataset size: {len(dataset.dataset)}')
    print(f'Data loading finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data-Loader-Test')
    parser.add_argument('--data-dir', required=True, type=str, 
                        help='Absolute path to the preprocessed dataset organized in train/test/val directories. \
                              Data has to be stored as tensors with samples in PyTorch .pt format.')
    parser.add_argument('--split-name', required=True, choices=['train', 'test', 'val'], type=str,
                        help='Name of data split')
    args = parser.parse_args()
    data_dir = args.data_dir
    split_name = args.split_name
    test_dataloader(data_dir, split_name)
