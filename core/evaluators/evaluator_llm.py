""" Evaluator receives results of inference and evaluates them. """

import argparse
import logging
import numpy as np
import os
import torch

from tensorboardX import SummaryWriter

from core.evaluators.evaluation_metrics import create_dataframe
from core.preprocessors.tokenizers import CharacterLevelTokenizer


class EvaluatorLLM:
    """ 
    EvaluatorBase class that generates evaluation metrics.
    It receives a batch of data directly from inference, eg. output / target tensors.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cumulated_outputs = []
        self.cumulated_targets = []

    def run_evaluator(self, output_tensor, target_tensor):
        """ 
        Accumulates inference results for every batch of data. 
        Args:
            output_tensor  # Shape: [batch_size, context_len, vocab_size]
            target_tensor  # Shape: [batch_size, context_len, vocab_size]
        """
        self.cumulated_outputs.extend(output_tensor.cpu().tolist())
        self.cumulated_targets.extend(target_tensor.cpu().tolist())

    def gen_full_report(self, output_dir, swriter):
        """ 
        Generates the evaluation report for accumulated data.
        Analyzes samples jointly to get high-level statistics.
        """
        logging.info("Running Evaluation Report Generation!")

        # Ensures that all accumulated values have the same size
        assert len(self.cumulated_outputs) == len(self.cumulated_targets)

        # Creates a pandas data frame of the accumulated data
        # TODO: Plots just one element as an example, should be extended to more samples / statistics
        df = create_dataframe(self.tokenizer, self.cumulated_outputs[0][0], self.cumulated_targets[0][0])
        df.to_csv(os.path.join(output_dir, "evaluation_results_table.csv"), index=False)
        df_html = df.to_html(index=False, escape=False)
        swriter.add_text("Evaluation Results Table", df_html, 0)


def test_evaluator(out_path):
    """ 
    This is a test code to run evaulation on random, example data. 
    """
    print("Evaluator test function...")
    swriter = SummaryWriter(log_dir=out_path)
    tokenizer = CharacterLevelTokenizer()
    evaluator = EvaluatorLLM(tokenizer)

    output_tensor = torch.rand(2, 8, len(tokenizer.vocab)) 
    target_tensor = torch.rand(2, 8, len(tokenizer.vocab))

    # Run evaluator on the same sample 10 times as a check
    for _ in range(10):
        evaluator.run_evaluator(output_tensor, target_tensor)

    # Generate report at the end of the test set processing
    evaluator.gen_full_report(out_path, swriter)
    print(f"Evaluator test function completed. Results stored at: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator Unit Test')
    parser.add_argument('--out-path', required=False, default="output_dir_evaluator", type=str, 
                        help='Output path to store the results')
    args = parser.parse_args()
    out_path = args.out_path
    test_evaluator(out_path)
