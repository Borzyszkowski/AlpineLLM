""" Run the training procedure """

import argparse
import os
import ray
import shutil

from functools import partial
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial
from typing import Dict, List

from core.utils.utils import makelogger, Config
from core.training.trainer import Trainer


logger = makelogger()


class TrialStatusReporter(CLIReporter):
    """ Class responsible for generating training reports """
    def __init__(self, cfg):
        super(CLIReporter, self).__init__()
        self.statuses = []
        self.cfg = cfg

    def should_report(self, trials, done=False):
        """ Generates the report once any new trial changes its status """
        old_statuses = self.statuses
        self.statuses = [t.status for t in trials]
        return old_statuses != self.statuses

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        progress_str = self._progress_str(trials, done, *sys_info)
        with open(os.path.join(self.cfg.ray_trials_path, "trial_logs.txt"), 'a') as f:
            f.write(progress_str + '\n')
            print(progress_str)


def training(cfg, hyperparam_cfg=None):
    """ Runs a single training with given configurations """
    os.chdir(cfg.project_root_path)
    ctx = train.get_context()
    cfg['trial_dir'] = os.path.join(cfg.work_dir, ctx.get_experiment_name(), os.path.basename(ctx.get_trial_dir())) 
    cfg['core_dir'] = os.path.join(cfg.work_dir, ctx.get_experiment_name())

    # save general config file and the trial hyperparam config file
    cfg.write_cfg(os.path.join(cfg.core_dir, 'TR%02d_%s' % (cfg.try_num, "training_cfg.yml")))
    hyperparam_cfg = Config(hyperparam_cfg)
    hyperparam_cfg.write_cfg(os.path.join(cfg.trial_dir, 'trial_hyperparam_cfg.yml'))

    # run the training
    trainer = Trainer(cfg=cfg, hyperparam_cfg=hyperparam_cfg)
    trainer.fit()


def run_hyperparameter_search(cfg, cpus_per_trial=1, gpus_per_trial=1):
    """ Runs a hyperparameter search using Ray, which starts parallel trainings """
    logger.info("Running the training experiment")
    logger.debug(f"cfg: {cfg}")
    ray.init(object_store_memory=10**9)
    assert ray.is_initialized() is True

    # define the hyperparameters that have to be explored
    hyperparam_cfg = {
        "lr": tune.grid_search([0.001]),
        "batch_size": tune.grid_search([512]),
        "context_len": tune.grid_search([8]),
    }
    logger.debug(f"hyperparam_cfg: {hyperparam_cfg}")

    # define a scheduler for the hyperparameter search
    scheduler = ASHAScheduler(
        time_attr='iter',
        metric="loss",
        mode="min",
        max_t=cfg.n_epochs,
        grace_period=cfg.n_epochs // 10 + 1,
        reduction_factor=2
    )

    # run a single training procedure
    result = tune.run(
        partial(training, cfg),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=hyperparam_cfg,
        num_samples=cfg.num_trials,
        scheduler=scheduler,
        progress_reporter=TrialStatusReporter(cfg),
        storage_path=cfg.ray_trials_path,
        log_to_file=("trial_stdout.log", "trial_stderr.log")
    )

    # save the results
    best_trial = result.get_best_trial("loss", "min", "all")
    logger.info(f"Best trial config: {best_trial.config}")

    # save the best checkpoint's path to the main experiment
    if os.path.exists(os.path.join(best_trial.local_path, "best_checkpoint_path.txt")):
        dst = os.path.join(cfg.work_dir, "best_checkpoint_path.txt")
        shutil.copy2(os.path.join(best_trial.local_path, "best_checkpoint_path.txt"), dst)
        logger.info(f"The best checkpoint's path saved at: {dst}")
    else:
        logger.error(f"The best checkpoint's path not found!")

    ray.shutdown()
    assert ray.is_initialized() is False


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
                        prog = "Training",
                        description = "Run training pipeline.")
    parser.add_argument("--data-path", required=False, default='./PREPROCESSED_DATA/P00', type=str,
                        help = "The path to the directory that contains ready dataset for training.")
    parser.add_argument('--work-dir', required=False, default='./TRAINING_RESULTS', type=str,
                        help='The path to the working directory where the training results will be saved.')
    parser.add_argument('--expr-ID', required=False, default='T00', type=str,
                        help='Training ID')
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()

    # Make sure that the working directory path is absolute
    if not os.path.isabs(args.work_dir):
        args.work_dir = os.path.abspath(os.path.join(cwd, args.work_dir)) 

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/training_cfg.yml')
    default_config = {
        'n_workers': 10,
        'use_multigpu': False,
        'dataset_dir': args.data_path,
        'expr_ID': args.expr_ID,
        'work_dir': os.path.join(args.work_dir, args.expr_ID),
        'load_weights_path': None,
        'try_num': 0,
        'n_epochs': 1,
        'log_every_iteration': 10,
        'cuda_id': 0,
        'model_name': None,
        'user_cfg_path': user_cfg_path,
        'project_root_path': cwd,
        'ray_trials_path': os.path.join(args.work_dir, args.expr_ID),
        'num_trials': 1,
        'early_stop_patience': 5,
        'export_onnx': True,
    }
    config = Config(default_config, user_cfg_path)

    # Run a single training or a hyperparameter search with Ray Tune
    run_hyperparameter_search(config)
