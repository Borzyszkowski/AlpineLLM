<p align="center">
    <h1 align="center">AlpineLLM</h1>
    <h4 align="center">
        A domain-specific language model for alpine storytelling.
        <br><br>
        Try asking about mountain adventures! 🏔️
    </h4>
    <h4 align="center">
        <strong>Author:</strong> 
        <a href="https://borzyszkowski.github.io/">Bartek Borzyszkowski</a> 
    </h4>  
</p>

<img src="assets/background_round.png" alt="AlpineLLM" style="border-radius: 15px;"/> 

## About AlpineLLM

AlpineLLM is a lightweight base language model pre-trained from scratch on raw text corpora drawn primarily from public-domain literature on alpinism.

This repository contains the source code for all stages of the model's development pipeline, including data preparation, training, evaluation, and deployment.

Currently supported version of the model: **AlpineLLM-Tiny-10M-Base**

Pre-trained model weights and an interactive live demo are available in the related repositories listed below.

### Related Repositories

- [**🤗 AlpineLLM Model Page @ HuggingFace**](https://huggingface.co/Borzyszkowski/AlpineLLM-Tiny-10M-Base)
- [**⛏️ AlpineLLM Live Demo @ HuggingFace Spaces**](https://huggingface.co/spaces/Borzyszkowski/AlpineLLM-App)

### How to install?

The software has been tested on Ubuntu 20.04 with CUDA 12.2 and Python3.10.

Please use a Python virtual environment to install the dependencies:
    
    python3.10 -m venv venv_AlpineLLM
    source venv_AlpineLLM/bin/activate
    pip install -r requirements.txt

### How to start?

- <b>Download dataset</b>

    The dataset consists of raw text corpora, including public-domain literature related to alpinism. It is based on several books available through Project Gutenberg and can be downloaded using the provided script:

    ```
    bash core/utils/download_dataset.sh
    ```

- <b>Pre-processing</b>

    The repository provides offline pre-processing stage that aims to serialize data to speed up training. Program parameters can be specified in the pre-processing config file.
    
    To start the pre-processing, please run:
    ```
    python3 run_preprocessing.py
    ```

- <b>Training</b>

    The repository provides a comprehensive training program allowing to schedule multiple training experiments with hyperparameter search. Program parameters can be specified in the training config file.
    
    To start the training, please run:
    ```
    python3 run_training.py
    ```

- <b>Evaluation (offline, inference-only)</b>

    The repository provides an evaluation program to quickly test the models. It allows to read the model's checkpoint and evaluate it on a desired test set. Program parameters can be specified in the evaluation config file.
    
    To start the evaluation, please run:
    ```
    python3 run_evaluation.py
    ```

- <b>Demo program (text generation)</b>

    The repository provides a demo program which allows to use the language model for text generation. It loads the model's checkpoint and allows the user to type the input prompt for completion by the model. Program parameters can be specified in the demo config file.

    To start the demo, please run:
    ```
    python3 run_demo.py
    ```

- <b>Deployment to Hugging Face</b>

    This project supports deployment to Hugging Face Model Hub for sharing the model weights.

    To start the deployment, please run:
    ```
    python3 run_deployment.py
    ```

    Live web demo of AlpineLLM at Hugging Face Spaces is developed as a separate repository:
    
    [https://huggingface.co/spaces/Borzyszkowski/AlpineLLM-App](https://huggingface.co/spaces/Borzyszkowski/AlpineLLM-App)  

### How to run unit tests?

Apart from the end-to-end data processing pipelines, it is possible to execute each component of the pipeline in a modular, indepentent way.

We provide several unit tests that allow to quickly start and test the key components:

- <b>Data loader unit test</b>

    ```
    python3 -m core.dataloaders.dataloader_llm
    ```

- <b>Evaluator unit test</b>

    ```
    python3 -m core.evaluators.evaluator_llm
    ```

### Contact and technical support
- <b>Bartek Borzyszkowski</b> <br>
    Web: <a href="https://borzyszkowski.github.io/">borzyszkowski.github.io</a>
