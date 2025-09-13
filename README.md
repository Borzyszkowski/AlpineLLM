# AlpineLLM

A domain-specific language model for alpine storytelling.

### How to install?

The software has been tested on Ubuntu 20.04 with CUDA 12.2 and Python3.10.

Please use a Python virtual environment to install the dependencies:
    
    python3.10 -m venv venv_AlpineLLM
    source venv_AlpineLLM/bin/activate
    pip install -r requirements.txt

### How to start?

- Pre-processing

    The repository provides offline pre-processing stage that aims to serialize data to speed up training. Please refer to the pre-processing config file for more details.
    
    To start the pre-processing, please run:
    ```
    python3 run_preprocessing.py
    ```

- Training

    The repository provides a comprehensive training program allowing to schedule multiple training experiments with hyperparameter search. Please refer to the training config file for more details.
    
    To start the training, please run:
    ```
    python3 run_training.py
    ```

- Evaluation (offline, inference-only)

    The repository provides an evaluation program to quickly test the models. It allows to read the model's checkpoint and evaluate it on a desired test set. Please refer to the evaluation config file for more details.
    
    To start the evaluation, please run:
    ```
    python3 run_evaluation.py
    ```

### How to run unit tests?

Apart from the end-to-end data processing pipelines, it is possible to execute each component of the pipeline in a modular, indepentent way.

We provide several unit tests that allow to quickly start and test the key components:

- Data loader unit test

    ```
    python3 -m core.dataloaders.dataloader
    ```

### Contact and technical support
- <b>Bartek Borzyszkowski</b> <br>
    Web: <a href="https://borzyszkowski.github.io/">borzyszkowski.github.io</a>
