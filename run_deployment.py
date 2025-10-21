""" Deploy the trained model to the Hugging Face Model Hub """

import argparse
import logging
import os

from datetime import datetime
from huggingface_hub import HfApi

from core.utils.utils import makelogger


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
                        prog = "Deployment to Hugging Face Model Hub")
    parser.add_argument("--model-path", required=False, default='./TRAINING_RESULTS/best_model.pt', type=str,
                        help = "Path to the resulting model artifact in the PyTorch format.")
    parser.add_argument("--repo-id", required=False, default='Borzyszkowski/AlpineLLM-model', type=str,
                        help = "ID of the repository on Hugging Face Hub where the model will be deployed.")
    return parser.parse_args()


def deploy_model(args):
    """ Deploy the model to Hugging Face, based on the configuration provided. """
    logging.info(f"Deployng model {args.model_path} to the Hugging Face repo_id: {args.repo_id}...")
    api = HfApi()
    
    # Upload model weights in PyTorch format
    api.upload_file(
        path_or_fileobj=args.model_path,
        path_in_repo="best_model.pt",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"{datetime.now().strftime('%Y%m%d_%H%M%S')} | Upload PyTorch model"
    )
    logging.info("PyTorch model uploaded successfully.")

    # Upload model weights in ONNX format (if available)
    onnx_path = args.model_path.replace('.pt', '.onnx')
    if onnx_path and os.path.exists(onnx_path):
        api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo="best_model.onnx",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"{datetime.now().strftime('%Y%m%d_%H%M%S')} | Upload ONNX model"
        )
        logging.info("ONNX model uploaded successfully.")
    else:
        logging.warning("No ONNX model found — skipping ONNX upload.")

    logging.info("Model deployment complete!")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    makelogger()
    deploy_model(args)
