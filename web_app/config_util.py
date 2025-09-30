""" Config utility """

import os
import yaml


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
