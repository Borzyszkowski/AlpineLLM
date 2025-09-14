""" 
This file contains a collection of evaluation metrics with plots for the evaluator module.
"""

import pandas as pd


def create_dataframe(tokenizer, outputs, targets):
    """ Create a dataframe of the model's results. """
    df = pd.DataFrame({
        "Outputs": tokenizer.decode(outputs),
        "Targets": tokenizer.decode(targets),
    })
    return df
