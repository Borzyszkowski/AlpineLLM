"""" Utility functions for the pre-processing scripts """

import json
import numpy as np
import torch

from copy import copy


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def np2torch(item):
    out = {}
    for k, v in item.items():
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v)
    return out


def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_ndjson(file_path, list_of_dicts):
    """ Saves a list of dictionaries to a file in NDJSON format. """
    # Ensure the input is a list of dictionaries
    if not isinstance(list_of_dicts, list) or not all(isinstance(d, dict) for d in list_of_dicts):
        raise ValueError("Input must be a list of dictionaries.")
    
    # Write each dictionary as a single JSON object on a new line
    with open(file_path, "w") as file:
        for dictionary in list_of_dicts:
            file.write(json.dumps(dictionary) + "\n")
