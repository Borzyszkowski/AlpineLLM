# AlpineLLM

A domain-specific language model for alpine storytelling.

### How to install?

The software has been tested on Ubuntu 20.04 with CUDA 12.2 and Python3.10.

Please use a Python virtual environment to install the dependencies:
    
    python3.10 -m venv venv_AlpineLLM
    source venv_AlpineLLM/bin/activate
    pip install -r requirements.txt

### How to start?

- Download dataset

    The dataset consists of raw text corpora, including public-domain literature related to alpinism. It is based on several books available through Project Gutenberg and can be downloaded using the provided script:

    ```
    bash core/utils/download_dataset.sh
    ```

- Pre-processing

    The repository provides offline pre-processing stage that aims to serialize data to speed up training. Program parameters can be specified in the pre-processing config file.
    
    To start the pre-processing, please run:
    ```
    python3 run_preprocessing.py
    ```

- Training

    The repository provides a comprehensive training program allowing to schedule multiple training experiments with hyperparameter search. Program parameters can be specified in the training config file.
    
    To start the training, please run:
    ```
    python3 run_training.py
    ```

- Evaluation (offline, inference-only)

    The repository provides an evaluation program to quickly test the models. It allows to read the model's checkpoint and evaluate it on a desired test set. Program parameters can be specified in the evaluation config file.
    
    To start the evaluation, please run:
    ```
    python3 run_evaluation.py
    ```

- Demo application (text generation)

    The repository provides a demo program which allows to use the language model for text generation. It loads the model's checkpoint and allows the user to type the input prompt for completion by the model. Program parameters can be specified in the demo config file.

    To start the demo, please run:
    ```
    python3 run_demo.py
    ```

### How to run unit tests?

Apart from the end-to-end data processing pipelines, it is possible to execute each component of the pipeline in a modular, indepentent way.

We provide several unit tests that allow to quickly start and test the key components:

- Data loader unit test

    ```
    python3 -m core.dataloaders.dataloader_llm
    ```

- Evaluator unit test

    ```
    python3 -m core.evaluators.evaluator_llm
    ```

### Contact and technical support
- <b>Bartek Borzyszkowski</b> <br>
    Web: <a href="https://borzyszkowski.github.io/">borzyszkowski.github.io</a>
