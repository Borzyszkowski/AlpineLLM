""" General purpose utility functions. """

import logging 
import os
import yaml


def makelogger(logfile_path=None, mode='w'):
    """ 
    Initializes and configures the logger. 
    Args:
        logfile_path (str): Desired path to a file where the logs are exported.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # if the logging directory is given, logs will be stored in a file
    if logfile_path:
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        fh = logging.FileHandler('%s' % logfile_path, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def makepath(desired_path, isfile=False):
    """
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    """
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):
            os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path):
            os.makedirs(desired_path)
    return desired_path


class Config(dict):
    """ Parser for the .yaml configuration files"""
    def __init__(self, config, user_cfg_path=None):
        user_config = self.load_cfg(user_cfg_path) if user_cfg_path else {}

        # Update default_cfg with user_config (overwriting defaults if needed)
        config.update(user_config)
        super().__init__(config)

    def load_cfg(self, load_path):
        with open(load_path, "r") as infile:
            cfg = yaml.safe_load(infile)
        return cfg if cfg is not None else {}

    def write_cfg(self, write_path):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        dump_dict = {k: v for k, v in self.items() if k != "default_cfg"}
        with open(write_path, "w") as outfile:
            yaml.safe_dump(dump_dict, outfile, default_flow_style=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
