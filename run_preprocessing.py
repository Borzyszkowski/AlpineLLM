"""
This code will process data and save the .pt files in the out_path folder.
Next, you can directly use the dataloader to load and use the data.    
"""
import argparse
import logging
import os

from core.utils.utils import makelogger, Config
from core.preprocessors.preprocessor_alpine import PreprocessorAlpine
from core.preprocessors.preprocessor_shakespeare import PreprocessorShakespeare


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
                        prog = "Preprocessing",
                        description = "Run preprocessing pipeline.")
    parser.add_argument('--inp-path', required=False, default='./raw_data', type=str,
                        help='The path to the raw data that should be pre-processed.')
    parser.add_argument('--out-path', required=False, default='./PREPROCESSED_DATA', type=str,
                        help='The path to the folder to save the processed data.')
    parser.add_argument('--data-type', required=False, default='alpine', type=str, choices=['alpine', 'shakespeare'],
                        help='Type of the data to be processed among supported options.')
    parser.add_argument('--process-id', required=False, default='P01', type=str,
                        help='The appropriate ID for the processed data (folder name).')
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/preprocessing_cfg.yml')
    default_config = {
        'data_type': args.data_type,
        'inp_path': os.path.join(args.inp_path, args.data_type),
        'out_path': os.path.join(args.out_path, args.process_id),
    }
    cfg = Config(default_config, user_cfg_path)
    cfg.write_cfg(write_path=os.path.join(cfg.out_path, 'preprocessing_cfg.yml'))

    makelogger(os.path.join(cfg.out_path, 'data_extraction.log'))
    
    # Run the desired data preprocessor
    logging.info(f"Running pre-processor for data_type {cfg.data_type}")
    if cfg.data_type == 'alpine':
        PreprocessorAlpine(cfg)
    elif cfg.data_type == 'shakespeare':
        PreprocessorShakespeare(cfg)
    else:
        raise NotImplementedError(f"Preprocessor for data type {cfg.data_type} is not implemented yet.")
