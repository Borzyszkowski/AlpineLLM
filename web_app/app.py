""" A simple Gradio web app to interact with the AlpineLLM model """

import gradio as gr
import os
import torch

from demo_inference import AlpineLLMInference
from config_util import Config


def start_app():
    """ Start the web app via Gradio """
    app = gr.Interface(
        fn=inference.generate_text,
        inputs=[
            gr.Textbox(lines=3, placeholder="Type your alpinism prompt..."),
            gr.Slider(50, 1000, value=200, step=10, label="Max tokens"),
        ],
        outputs=gr.Textbox(),
        title="AlpineLLM Demo",
        description="A domain-specific language model for alpine storytelling.",
    )
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define the configuration
    cfg = {
        'load_weights_path': "model/best_model",
        'cuda_id': 0,
        'model_type': 'transformer',
    }
    cfg = Config(cfg)

    # Define the hyperparameters
    hyperparam_cfg={
        "embedding_dim": 384,
        "num_heads": 6,
        "num_layers": 6,
        "dropout": 0.2,
        "context_len": 256,
        "lr": 3e-4,
    }
    hyperparam_cfg = Config(hyperparam_cfg)

    # Start the application
    inference = AlpineLLMInference(cfg, hyperparam_cfg)
    start_app()
