""" Main class for the training """

import json
import numpy as np
import logging
import os
import torch

from datetime import datetime
from ray import train
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core.evaluators.evaluator_llm import EvaluatorLLM
from core.dataloaders.dataloader_llm import DataloaderLLM
from core.models.bigram import BigramLanguageModel
from core.models.transformer_decoder import TransformerDecoder
from core.preprocessors.tokenizers import CharacterLevelTokenizer
from core.training.early_stopping import EarlyStopping
from core.utils.utils import makelogger, makepath


class Trainer:
    def __init__(self, cfg, hyperparam_cfg, inference_only=False):
        # define the basic configuration
        self.cfg = cfg
        self.hyperparam_cfg = hyperparam_cfg
        self.inference_only=inference_only
        self.trial_dir = cfg.trial_dir if not self.inference_only else cfg.work_dir
        makelogger(makepath(os.path.join(self.trial_dir, f'{cfg.expr_ID}.log'), isfile=True)).info if not inference_only else None

        # set the destination directory for TensorBoard
        self.swriter = self.set_summary_writer()

        # set the hardware type (use GPU with CUDA if available)
        self.device = self.set_hardware_type()
        
        # select the tokenizer (same as used in data pre-processing)
        self.tokenizer_name, self.tokenizer = self.select_tokenizer()

        # select the neural network model as a backbone
        self.model_name, self.model = self.select_model()

        # check if multiple GPUs can be used for training
        if cfg.use_multigpu:
            self.model = nn.DataParallel(self.model)
            logging.info("Training on Multiple GPUs")

        # define an optimizer and initialize the loss
        if not self.inference_only:
            self.optimizer = self.set_optimizer()
            self.best_loss = np.inf

        # load weights from the file if specified
        if cfg.load_weights_path is not None:
            self.get_model(self.cfg.load_weights_path)

        # load data for each subset (assuming it has been preprocessed)
        self.ds_train, self.ds_test = self.load_data()
        self.train_iterator, self.test_iterator = iter(self.ds_train), iter(self.ds_test)

    def fit(self):
        """ Main logic, in which the model is trained and evaluated """
        start_time = datetime.now().replace(microsecond=0)
        logging.info(f'Started training at {start_time} for {self.cfg.train_iters} iterations\n')

        # Schedule learning rate optimization and early stopping
        prev_lr = self.hyperparam_cfg['lr']
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(patience=self.cfg.early_stop_patience, trace_func=logging.info)

        # Run the main train / val logic
        global_iter = 0
        while global_iter < self.cfg.train_iters:
            logging.info('--- Current Train/Val Iter # %03d' % global_iter)

            # Run main training and evaluation logic
            train_loss, global_iter = self.train(global_iter)
            val_loss = self.evaluate(global_iter)

            # Update the learning rate if required by the optimizer
            lr_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                logging.info('--- Model learning rate changed from %.2e to %.2e ---' % (prev_lr, current_lr))
                prev_lr = current_lr

            # Save loss and the best model
            with torch.no_grad():
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(global_iter)
                self.swriter.add_scalars('total_loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_iter)
                with open(os.path.join(self.trial_dir, "best_val_loss.json"), 'w') as file:
                    json.dump({"best_val_loss": self.best_loss}, file)

            # Check the evaluation loss for the early stopping condition
            if early_stopping(val_loss):
                logging.info('Early stopping training of the model!\n')
                break

        # Finish the training and evaluate the model on the test set
        end_time = datetime.now().replace(microsecond=0)
        logging.info(f'Finished Training at {end_time}')
        logging.info(f'Training done in {(end_time - start_time)}!')
        logging.info(f'Best val total loss achieved: {self.best_loss}')
        logging.info(f'Best model path: {self.model_path}')

        # Use the best model's checkpoint and run the evaluation on the test set
        logging.info(f'Running evaluation on the test set!\n')
        self.get_model(self.model_path) 
        _ = self.evaluate(global_iter=1, ds_name="test")
        self.swriter.close() 

        # Export the model to ONNX format if required
        if self.cfg.export_onnx:
            self.export_to_onnx(onnx_path=f'{self.model_path}.onnx')

    def train(self, global_iter, ds_name='train'):
        """ Main training logic """
        self.model.train()
        interval_loss = []

        # Generate some example text to verify the model is working
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        example = self.tokenizer.decode(self.model.generate(context, max_new_tokens=500)[0].tolist())
        logging.info(f"Example text generated by the model:\n{example}\n")

        # LLM is trained for a certain number of iterations (batches)
        for curr_iter in range(self.cfg.val_every_iteration):

            # Get the next batch of data (or restart if iterator is exhausted)
            input_tensor, target_tensor = self.get_next_data_batch()

            # Create inputs for the neural network
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # Generate model visualization and information about the input tensor
            if curr_iter == 0 and global_iter == 0:
                self.visualize_model(input_tensor)

            # Clear gradients
            self.optimizer.zero_grad()
            output_tensor = self.model(input_tensor)

            # Calculate the loss function
            current_loss = self.compute_loss(output_tensor, target_tensor)

            # Calculating gradients
            current_loss.backward()
            interval_loss.append(current_loss.item())

            # Update parameters
            self.optimizer.step()

            # Print information about the loss
            if global_iter % self.cfg.log_every_iteration == 0:
                self.create_loss_message(current_loss, global_iter, ds_name)
            global_iter += 1

        return self.compute_interval_summary(ds_name, interval_loss, global_iter), global_iter

    @torch.no_grad()
    def evaluate(self, global_iter, ds_name='val'):
        """ Main evaluation logic for validation and testing of the model """
        iter_steps = self.cfg.val_iters if ds_name == 'val' else len(self.ds_test)
        self.model.eval()
        interval_loss = []

        # Initialize the evaluator if a test set is used
        self.evaluator = EvaluatorLLM(self.tokenizer) if 'test' in ds_name else None

        # Generate some example text to verify the model is working
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        example = self.tokenizer.decode(self.model.generate(context, max_new_tokens=500)[0].tolist())
        logging.info(f"Example text generated by the model:\n{example}\n")
        
        with torch.no_grad():
            for curr_iter in range(iter_steps):

                # Get the next batch of data (or restart if iterator is exhausted)
                if ds_name == 'val':
                    input_tensor, target_tensor = self.get_next_data_batch()
                elif 'test' in ds_name:
                    input_tensor, target_tensor = next(self.test_iterator)

                # Create inputs for the neural network
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                # Forward propagation
                output_tensor = self.model(input_tensor)

                # Calculate the loss function
                current_loss = self.compute_loss(output_tensor, target_tensor)
                interval_loss.append(current_loss.item())

                # Print information about the loss
                if curr_iter % self.cfg.log_every_iteration == 0:
                    self.create_loss_message(current_loss, curr_iter, ds_name)

                # Run evaluator on the test set samples
                if 'test' in ds_name:
                    self.evaluator.run_evaluator(output_tensor, target_tensor)

            # Generate evaluation report at the end of the test set processing
            self.evaluator.gen_full_report(self.trial_dir, self.swriter) if 'test' in ds_name else None

            return self.compute_interval_summary(ds_name, interval_loss, global_iter)

    def compute_interval_summary(self, ds_name, interval_loss, global_iter):
        """ Compute the interval summary and save the report """
        interval_loss = sum(interval_loss) / len(interval_loss)
        train.report({"mode": ds_name, "loss": interval_loss, "global_iter": global_iter})
        return interval_loss

    def set_summary_writer(self):
        """ Set the summary writer for tensorboard """
        summary_logdir = os.path.join(self.trial_dir, 'summaries')
        swriter = SummaryWriter(log_dir=summary_logdir)
        logging.info(f'[{self.cfg.expr_ID}] - Experiment has started!')
        logging.info(f'tensorboard --logdir={summary_logdir}')
        logging.info(f'Torch Version: {torch.__version__}')
        logging.info(f'Base dataset directory: {self.cfg.dataset_dir}')
        return swriter

    def set_hardware_type(self):
        """ Set the hardware type such as CPU/GPU """
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{self.cfg.cuda_id}" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.empty_cache()
            gpu_brand = torch.cuda.get_device_name(self.cfg.cuda_id)
            gpu_count = torch.cuda.device_count() if self.cfg.use_multigpu else 1
            logging.info(f'Using {gpu_count} CUDA core(s) [{gpu_brand}]!')
        else:
            logging.info(f'Using CPU!')
        return device

    def set_optimizer(self):
        """ Set the optimizer and its parameters """
        params = [var[1] for var in self.model.named_parameters()]
        params_number = sum(p.numel() for p in params if p.requires_grad)
        logging.info('Total trainable parameters of the model: %2.3f M.' % (params_number * 1e-6))
        optimizer = optim.Adam(params, lr=self.hyperparam_cfg['lr'])
        return optimizer

    def select_model(self):
        """ Selects the neural network architecture based on the desired configuration """
        vocab_size = len(self.tokenizer.vocab)
        if self.cfg.model_type == 'transformer':
            model = TransformerDecoder(vocab_size=vocab_size, 
                                       hyperparam_cfg=self.hyperparam_cfg,
                                       device=self.device).to(self.device)
        elif self.cfg.model_type == 'bigram':
            model = BigramLanguageModel(vocab_size=vocab_size).to(self.device)
        else:
            raise ValueError(f"Model type '{self.cfg.model_type}' is not supported!")
        model_name = model.__class__.__name__
        logging.info(f'Selected model type: {self.cfg.model_type} with name: {model_name}')
        return model_name, model

    def select_tokenizer(self):
        """ Selects the tokenizer based on the desired configuration """
        tokenizer = CharacterLevelTokenizer()
        tokenizer_name = tokenizer.__class__.__name__
        logging.info(f'Selected tokenizer name: {tokenizer_name}')
        return tokenizer_name, tokenizer

    def get_model(self, model_path):
        """ Loads weights of the model from the specified path """
        restored = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Metadata Format in PyTorch: Recognizes model_state_dict and optionally optimizer_state_dict.
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:  
                restored.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                restored.load_state_dict(checkpoint, strict=False)

        # Legacy Tuple Format in PyTorch: Extracts model and optimizer state dictionaries from a tuple
        elif isinstance(checkpoint, tuple):
            restored.load_state_dict(checkpoint[0], strict=False)
            if len(checkpoint) > 1 and not self.inference_only:
                self.optimizer.load_state_dict(checkpoint[1])

        logging.info(f'Restored model from: {model_path}')

    def save_model(self, global_iter):
        """ Saves a checkpoint with the model """
        checkpoint_dir = os.path.join(self.trial_dir, f"checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{global_iter}.pt")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.model_path)
        with open(os.path.join(train.get_context().get_trial_dir(), "best_checkpoint_path.txt"), "w") as f:
            f.write(self.model_path + "\n")
        logging.info(f'Checkpoint with the model saved at: {self.model_path}')

    def export_to_onnx(self, onnx_path):
        """ Export the trained model to ONNX format """
        logging.info(f"Exporting model to ONNX format...")
        example_input, example_target = next(iter(self.ds_test))
        example_input = example_input.to(self.device)
        torch.onnx.export(
            self.model, example_input, onnx_path,
            input_names=['input_ids'], output_names=['logits'],
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'context_len'},
                          'logits': {0: 'batch_size', 1: 'context_len'}},
            opset_version=12
        )
        logging.info(f"Model exported to ONNX saved at: {onnx_path}")

    def visualize_model(self, input_tensor):
        """ Save model definition and input tensor """
        with open(os.path.join(self.trial_dir, f'model.txt'), 'w') as f:
            f.write(str(self.model))

        with open(os.path.join(self.trial_dir, f'input_tensor.txt'), 'w') as f:
            logging.debug(f"Input tensor (example): {input_tensor}")
            f.write(str(input_tensor))

        with open(os.path.join(self.trial_dir, f'input_tensor_shape.txt'), 'w') as f:
            logging.debug(f"Input tensor shape: {str(input_tensor.shape)}")
            f.write(f"Input tensor shape: {str(input_tensor.shape)}")

    def compute_loss(self, output_logits, targets):
        """ Define and compute the loss function """
        B, T, C = output_logits.shape
        output_logits = output_logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(output_logits, targets)
        return loss

    def load_data(self):
        """ Loads train/test data using the pre-defined dataloader """
        test_kwargs = {'num_workers': self.cfg.n_workers,
                       'batch_size': self.hyperparam_cfg['batch_size'],
                       'shuffle': False, 
                       'drop_last': False}

        # Load the test dataset
        ds_test = DataloaderLLM(dataset_dir=self.cfg.dataset_dir, 
                                data_split="test", 
                                context_len=self.hyperparam_cfg['context_len'])    
        ds_test = DataLoader(ds_test, **test_kwargs)
        logging.info(f'Dataset Test size: {len(ds_test.dataset)}')

        # For inference mode only, run evaluation on the test data
        if self.inference_only:
            return None, ds_test
        
        train_kwargs = {'num_workers': self.cfg.n_workers,
                        'batch_size': self.hyperparam_cfg['batch_size'],
                        'shuffle': True, 
                        'drop_last': False}

        # Load train data
        ds_train = DataloaderLLM(dataset_dir=self.cfg.dataset_dir, 
                                 data_split='train',
                                 context_len=self.hyperparam_cfg['context_len'])    
        ds_train = DataLoader(ds_train, **train_kwargs)
        logging.info(f'Dataset Train size: {len(ds_train.dataset)}')
        return ds_train, ds_test

    def get_next_data_batch(self):
        """ Retrieves the next batch from the infinite dataset iterator """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.ds_train)
            return next(self.train_iterator)
            
    def create_loss_message(self, loss, it, ds_name):
        """ Generates and logs the loss message with given input parameters """
        exp = self.cfg.expr_ID + str(self.cfg.try_num)
        msg = f'Exp: {exp} - Split: {ds_name} - Iter: {it} - Model: {self.model_name} - Loss: {loss}'
        logging.info(msg)
