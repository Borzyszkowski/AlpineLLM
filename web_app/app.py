""" A simple Gradio web app to interact with the AlpineLLM model. """

import gradio as gr
import torch

from demo_inference import AlpineLLMInference
from tokenizer import CharacterLevelTokenizer


def initialize_inference():
    """ Load tokenizer and model """

    # Load tokenizer
    tokenizer = CharacterLevelTokenizer()

    # Load model (re-instantiate + load weights)
    inference = AlpineLLMInference(
        cfg={"load_weights_path": "model/best_model"},
        hyperparam_cfg={
            "embedding_dim": 384,
            "num_heads": 6,
            "num_layers": 6,
            "dropout": 0.2,
            "context_len": 256,
            "lr": 3e-4,
        },
        tokenizer=tokenizer,
    )

def generate(prompt, max_tokens=200):
    return inference.generate_text(prompt, max_new_tokens=max_tokens)


def start_app():
    """ Start the web app via Gradio """
    app = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(lines=3, placeholder="Type your alpinism prompt..."),
            gr.Slider(50, 500, value=200, step=10, label="Max tokens"),
        ],
        outputs=gr.Textbox(),
        title="AlpineLLM Demo",
        description="A domain-specific language model for alpine storytelling.",
    )
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == '__main__':
    start_app()
