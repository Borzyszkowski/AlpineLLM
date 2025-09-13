""" Evaluator receives results of inference and evaluates them. """

import argparse
import logging
import numpy as np
import os
import torch

from tensorboardX import SummaryWriter

from core.evaluators.evaluation_metrics import (create_dataframe, 
                                                create_classification_report, 
                                                create_confusion_matrix)

class EvaluatorBase:
    """ 
    EvaluatorBase class that generates evaluation metrics.
    It receives a batch of data directly from inference, eg. input / output tensors.
    Additionally, it has access to the ground_truth (if available) and video_path (if available)

    Please interact with the evaluator through public methods:
     - run_evaluator()
     - gen_full_report()
    Other methods should be used as private from inside of the class.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.cumulated_outputs = []
        self.cumulated_truths = []
        self.cumulated_paths = []

    def run_evaluator(self, output_tensor, ground_truths, sample_paths):
        """ 
        Accumulates inference results for every batch of data. 
        Args:
            output_tensor  # Size: [batch_size, classes_num]
            ground_truths  # Size: [batch_size, classes_num]
            sample_paths   # Size: [batch_size]
        """
        output_tensor = torch.argmax(output_tensor, dim=1)
        self.cumulated_outputs.extend(output_tensor.cpu().numpy())
        self.cumulated_truths.extend(ground_truths.cpu().numpy())
        self.cumulated_paths.extend(sample_paths)

    def gen_full_report(self, output_dir, swriter):
        """ 
        Generates the classification report for accumulated data, i.e. analyzes samples jointly to get high-level statistics.
        """
        logging.info("Running Classification Report Generation!")

        # Ensures that all accumulated values have the same size
        assert len(self.cumulated_outputs) == len(self.cumulated_truths) == len(self.cumulated_paths)

        # Creates a pandas data frame of the accumulated data
        df = create_dataframe(self.cumulated_outputs, self.cumulated_truths, self.cumulated_paths)
        df.to_csv(os.path.join(output_dir, "classification_results_table.csv"), index=False)
        df_html = df.to_html(index=False, escape=False)
        swriter.add_text("Classification Results Table", df_html, 0)

        # Creates a scikit-learn classification report
        cr_plot = create_classification_report(truth=self.cumulated_truths, 
                                               pred=self.cumulated_outputs)
        swriter.add_figure("classification_report", cr_plot)
        cr_plot.savefig(os.path.join(output_dir, "classification_report.png"), dpi=300)

        # Creates a scikit-learn confusion matrix
        cm_plot = create_confusion_matrix(truth=self.cumulated_truths, 
                                          pred=self.cumulated_outputs)
        swriter.add_figure("confusion_matrix", cm_plot)
        cm_plot.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)


def test_evaluator(out_path):
    """ 
    This is a test code to run evaulation on random, example data. 
    """
    print("Evaluator test function...")
    swriter = SummaryWriter(log_dir=out_path)
    evaluator = EvaluatorBase()

    input_tensor = torch.rand(2, 10, 7) 
    output_tensor = torch.rand(2, 10, 7)
    labels = [0, 1]
    seq_paths = ["example_path_1", "example_path_2"]

    # Run evaluator on the same sample 10 times as a check
    for _ in range(10):
        evaluator.run_evaluator(input_tensor, output_tensor, labels, seq_paths)

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
