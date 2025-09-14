""" Run the Evaluation (Inference Only) procedure """

import argparse
import json
import logging
import os

from tqdm import tqdm

from core.utils.utils import Config, makelogger
from core.training.trainer import Trainer


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
                        prog = "Evaluation (Inference Only)",
                        description = "Run evaluation pipeline.")
    parser.add_argument("--data-path", required=False, default='./PREPROCESSED_DATA/P00', type=str,
                        help = "Absolute path to the directory that contains ready dataset for evaluation.")
    parser.add_argument("--training-path", required=False, default='./TRAINING_RESULTS', type=str,
                        help = "Absolute path to the training experiment for evaluation.")
    parser.add_argument('--work-dir', required=False, default='./EVAL_RESULTS', type=str,
                        help='The path to the working directory where the evaluation results will be saved.')
    parser.add_argument('--expr-ID', required=False, default='E00', type=str,
                        help='Evaluation ID')
    return parser.parse_args()


def run_evaluation(cfg):
    """ Runs the evaluation experiment with given configurations """
    logging.info("Running offline evaluation on test data")
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

    # run the evaluation using inference_only flag
    evaluator = Trainer(cfg=cfg, hyperparam_cfg=hyperparam_cfg, inference_only=True)
    evaluator.evaluate(epoch_num=1, ds_name="test")

    logging.info(f"Completed evaluation for the model: {cfg.load_weights_path}")
    logging.info(f"Results available in the output directory {cfg.work_dir}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()

    # Read the model's checkpoint path from a file
    checkpoint_path_file = os.path.join(args.training_path, "best_checkpoint_path.txt")
    if os.path.exists(checkpoint_path_file):
        with open(checkpoint_path_file, "r") as f:
            load_weights_path = f.read().strip()
    else:
        logging.error(f"File {checkpoint_path_file} does not exist")
        exit(1)

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/evaluation_cfg.yml')
    default_config = {
        'dataset_dir': args.data_path,
        'expr_ID': args.expr_ID,
        'work_dir': os.path.join(args.work_dir, args.expr_ID),
        'trial_dir': os.path.join(args.work_dir, args.expr_ID),
        'cuda_id': 0,
        'try_num': 0,
        'use_multigpu': False,
        'n_workers': 10,
        'log_every_iteration': 10,
        'model_name': None,
        'user_cfg_path': user_cfg_path,
        'project_root_path': cwd,
        'load_weights_path': load_weights_path,
    }
    config = Config(default_config, user_cfg_path)
    config.write_cfg(write_path=os.path.join(config.work_dir, 'evaluation_cfg.yml'))
    logger = makelogger(os.path.join(config.work_dir, "logs.txt"))

    # Run a single evaluation experiment
    run_evaluation(config)
