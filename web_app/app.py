""" A simple Gradio web app to interact with the AlpineLLM model """

import gradio as gr
import os
import torch

from huggingface_hub import hf_hub_download

from demo_inference import AlpineLLMInference
from config_util import Config


def download_model(cfg):
    """ Download the model weights from Hugging Face Hub """
    model_path = hf_hub_download(
        repo_id=cfg.repo_id, 
        filename=cfg.model_name, 
        cache_dir="./model-cache"
    )
    return model_path


def start_app():
    """ Start the web app via Gradio with custom layout """
    with gr.Blocks(css="""#builtwithgradio, .footer, .svelte-1ipelgc {display: none !important;}""") as app:
        gr.Markdown("<h1 style='text-align: center;'> AlpineLLM App</h1>")
        gr.Markdown(
            "<p style='text-align: center;'>"
            "A domain-specific language model for alpine storytelling. <br>"
            "Generate climbing stories, mountain impressions, and expedition-style text."
            "</p>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    lines=8,
                    label="Your alpine prompt...",
                    placeholder="A dawn climb on the Matterhorn..."
                )
                max_tokens = gr.Slider(50, 1000, value=300, step=10, label="Max output tokens")
                generate_btn = gr.Button("🚀 Generate")

            with gr.Column(scale=2):
                output = gr.Textbox(lines=20, label="Generated Alpine Story", interactive=False)

        # Bind button click to inference
        generate_btn.click(
            fn=inference.generate_text,
            inputs=[prompt, max_tokens],
            outputs=output
        )

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define the configuration
    cfg = {
        'cuda_id': 0,
        'model_type': 'transformer',
        'repo_id': "Borzyszkowski/AlpineLLM-model",
        'model_name': "best_model",
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

    # Ensure model weights are available
    cfg.load_weights_path = download_model(cfg)

    # Start the application
    inference = AlpineLLMInference(cfg, hyperparam_cfg)
    start_app()
