""" Run the demo application """

import argparse
import json
import logging
import os

from core.demo_program.demo_inference import AlpineLLMInference
from core.utils.utils import Config, makelogger


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
                        prog = "Demo",
                        description = "Run demo pipeline.")
    parser.add_argument("--training-path", required=False, default='./TRAINING_RESULTS/T01', type=str,
                        help = "Absolute path to the training experiment to load.")
    parser.add_argument('--work-dir', required=False, default='./DEMO_RESULTS', type=str,
                        help='The path to the working directory where the demo results will be saved.')
    parser.add_argument('--expr-ID', required=False, default='D01', type=str,
                        help='Demo ID')
    return parser.parse_args()


def run_demo(cfg):
    """ Runs the demo experiment with given configurations """
    logging.info(f"Running demo ecperiment for the model type: {cfg.model_type}")
    logging.info(f"Selected model: {cfg.load_weights_path}")
    os.chdir(cfg.project_root_path)

    # every training experiment has hyperparameters in the "params.json"
    params_file = os.path.join(os.path.dirname(os.path.dirname(cfg.load_weights_path)), "params.json")
    if not os.path.exists(params_file):
        logging.error(f"Could not find a params file for the model: {params_file}")
        exit(1)
    with open(params_file, 'r') as file:
        hyperparam_cfg = json.load(file)
        hyperparam_cfg["batch_size"] = 1
    hyperparam_cfg = Config(hyperparam_cfg)
    hyperparam_cfg.write_cfg(os.path.join(cfg.work_dir, f'hyperparam_cfg.yml'))
    logging.debug(f"hyperparam_cfg: {hyperparam_cfg}")
    logging.debug(f"cfg: {cfg}")

    # run the demo application using inference wrapper
    demo_inference = AlpineLLMInference(cfg=cfg, hyperparam_cfg=hyperparam_cfg)
    demo_inference.run_demo()

    logging.info(f"Completed demo app for the model: {cfg.load_weights_path}")
    logging.info(f"Results available in the output directory {cfg.work_dir}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()

    # Make sure that the working directory path is absolute
    if not os.path.isabs(args.work_dir):
        args.work_dir = os.path.abspath(os.path.join(cwd, args.work_dir)) 

    # Read the model's checkpoint path from a file
    checkpoint_path_file = os.path.join(args.training_path, "best_checkpoint_path.txt")
    if os.path.exists(checkpoint_path_file):
        with open(checkpoint_path_file, "r") as f:
            load_weights_path = f.read().strip()
    else:
        logging.error(f"File {checkpoint_path_file} does not exist")
        exit(1)

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/demo_cfg.yml')
    default_config = {
        'expr_ID': args.expr_ID,
        'work_dir': os.path.join(args.work_dir, args.expr_ID),
        'load_weights_path': load_weights_path,
        'cuda_id': 0,
        'model_type': 'transformer',
        'user_cfg_path': user_cfg_path,
        'project_root_path': cwd,
        'max_new_tokens': 500
    }
    config = Config(default_config, user_cfg_path)
    config.write_cfg(write_path=os.path.join(config.work_dir, 'demo_cfg.yml'))
    logger = makelogger(os.path.join(config.work_dir, "logs.txt"))
    
    # Run a single demo experiment
    run_demo(config)
