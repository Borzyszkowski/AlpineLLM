""" 
This file contains a collection of evaluation metrics with plots for the evaluator module.
"""

import pandas as pd


def create_dataframe(preds, targets):
    """ Create a dataframe of the model's results. """
    df = pd.DataFrame({
        "Predictions": preds,
        "Targets": targets,
    })
    return df
