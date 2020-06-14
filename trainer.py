import json
import math
import os
import shutil
import sys
from typing import Type, Dict

import torch
import transformers

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data
from torch.optim.optimizer import Optimizer
from tqdm import trange, tqdm

from dateutil.relativedelta import relativedelta

import random
import numpy as np
import logging
from model import EmptyHeads

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from datetime import datetime

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def set_seed(seed, n_gpu):
    logger.info(f"   see seed for random, numpy and torch {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def print_model_state_dict(model):
    for param_tensor in model.state_dict():
        logger.info(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")


def print_optimizer_state_dict(optimizer):
    for var_name in optimizer.state_dict():
        logger.info(f"{var_name}\t{optimizer.state_dict()[var_name]}")


def count_params(model: torch.nn.Module, print_details: bool = False):
    trainable_count = 0
    total_count = 0
    if isinstance(model, torch.nn.Sequential):
        for index in model._modules:
            if print_details:
                print_model_state_dict(model._modules[index])
                logger.info(model._modules[index])
            trainable_count += sum(p.numel() for p in model._modules[index].parameters() if p.requires_grad)
            total_count += sum(p.numel() for p in model._modules[index].parameters())
    else:
        if print_details:
            print_model_state_dict(model)
            logger.info(model)
        total_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'  Total params: {total_count}')
    logger.info(f'  Trainable params: {trainable_count}')
    logger.info(f'  Non-trainable params: {total_count - trainable_count}')


def batch_to_device(batch, device, keep_label=False):
    features = batch['features']
    if isinstance(features, dict):
        for feature_name in features:
            features[feature_name] = features[feature_name].to(device)
    else:
        for inx in range(len(features)):
            for feature_name in features[inx]:
                features[inx][feature_name] = features[inx][feature_name].to(device)

    label_space = batch['labels']
    if label_space == None:  # for tasks like lm, labels are none.
        return features, None
    if not keep_label:
        labels = {"label_space_" + str(inx): label_space[inx].to(device) if torch.is_tensor(label_space[inx]) else
        label_space[inx] for inx in range(len(label_space))}
    else:
        labels = label_space
    return features, labels


def is_wandb_available():
    return _has_wandb


class CollateFunction():
    def __init__(self, up_model):
        self.up_model = up_model

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            padded_features = self.up_model.padding_features(batch)
            return {'features': padded_features,
                    "labels": None}  # label_ids are in features, this task does not need labels, we set


class ModelTrainer():
    def __init__(self, up_model: nn.Module, down_layer: nn.Module = None, train_dataset=None,
                 dev_dataset=None, dev_evaluator=None,
                 epochs: int = 1,
                 visiable_device: str = "0",
                 scheduler: str = 'warmuplinear',
                 warmup_ratio: float = 0.1,
                 optimizer_class: Type[Optimizer] = transformers.AdamW,
                 optimizer_params: Dict[str, object] = {'lr': 5e-5, 'eps': 1e-6, 'correct_bias': False},
                 weight_decay: float = 0.01,
                 early_stop: int = 20,
                 # 20 evaluation steps without improving on the early_stop_on metric as specified in dev_evaluator
                 evaluation_steps: int = 500,
                 output_path: str = None,
                 save_best_model: bool = True,
                 max_grad_norm: float = 1,
                 fp16: bool = False,
                 accumulation_steps=1,
                 fp16_opt_level: str = 'O1',
                 seed: int = 122,
                 data_loader_shuffle=True,
                 device: str = None,
                 dev_batch_size: int = -1,  # the same as train_batch_size
                 n_gpu: int = None,
                 report_model: bool = True,
                 per_gpu_train_batch_size: int = 8,
                 restore_training: bool = False,
                 local_rank: int = -1,
                 wandb_config=None):
        """
        this trainer is written for training a sequential model that contains an upstream_layer (usually transformers)
        and a downstream_layer (usually task-specific heads like FF, RNN, CNN for encoding the output of upstram_layer)

        :param up_model: transformers like transformers.GPT2LMHeadModel or transformers.BERTModel
        :param down_layer: None if up_model already wraps up with an output encoder such as LMHead in GPT2LMHeadModel, else nn.Module for encoding the output of up_model
        :param train_dataset: train_dataset, it can be either instance of torch.data.Dataset or IterableDataset (defined in data.py)
        :param dev_dataset: dev_dataset, it can be either instance of torch.data.Dataset or IterableDataset
        :param dev_evaluator: dev_evaluator, evaluator on dev_dataset for early stop and performance tracking during training (defined in evaluate.py)
        :param epochs: number of epoches for training
        :param visiable_device: devices chosen to perform training
        :param scheduler: scheduler specially from transformers: see options in self._get_scheduler
        :param warmup_ratio: warmup_ratio ratio for learning rate over total training steps
        :param optimizer_class: transformers.AdamW de byfault
        :param optimizer_params: optimizer params
        :param weight_decay:weight decay
        :param early_stop:early stop steps
        :param evaluation_steps:logging steps
        :param output_path: path to save the checkpoint with the best performance as specified in early_stop_on in dev_evaluator instance
        :param save_best_model:save best checkpoint or the latest checkpoint
        :param max_grad_norm:max grad norm
        :param fp16: fp16 training
        :param accumulation_steps:accumulation steps
        :param fp16_opt_level:fp16 opt level
        :param seed:random seed for reproducibility
        :param data_loader_shuffle:Whether to shuffle data_loader of training dataset and dev dataset after epoch ends
        :param device: device for training, None or gpu for gpu training, cpu for gpu training
        :param dev_batch_size: development batch size, usually larger than training batch size due to no grads calculation and hence less burden on memory
        :param n_gpu: number of gpus for training
        :param report_model:if report model's structure and number of trainable params in logging
        :param per_gpu_train_batch_size: what it means literally
        :param restore_training: if restore training if the training process is interupped due to some accidents
        :param local_rank:for distributed training
        :param wandb_config: wandb logging if not none, else without wandb logging
        """

        self.up_model = up_model
        if down_layer == None:
            # In this example, the upstream_layer already integrate the downstream head (namely, simple LM head as in transformers.GPT2LMHeadModel)
            # EmptyHeads is created here only for placeholder purpose
            down_layer = EmptyHeads()

        self.down_layer = down_layer
        assert output_path != None
        output_path = os.path.join("tmp", output_path)
        # os.makedirs(output_path,exist_ok=True)
        if restore_training:
            if not os.listdir(output_path):
                raise ValueError(f"no checkpoint found in {output_path}")
            else:
                logger.info("   loading embedding weights from saved checkpoint")
                self.up_model = self.up_model.reload(
                    output_path)  # for other transformers (apart from bert), the load_saved function has not been added

                logger.info("   loading downstream weights from saved checkpoint")
                self.down_layer.load_saved(output_path)
                with open(output_path + "/ck_report.json") as f:
                    self.ck_report = json.load(f)

        self.model = torch.nn.Sequential(self.up_model, self.down_layer)

        if is_wandb_available() and wandb_config != None:
            # keep track of model topology and gradients if is_wandb_available and args!=None
            wandb.init(project=wandb_config.wandb_project_name, config=wandb_config, name=wandb_config.wandb_run_name)
            wandb.watch(
                (self.up_model, self.down_layer), log_freq=max(100, evaluation_steps)
            )
        self.wandb_config = wandb_config

        self._restore_training = restore_training
        self.early_stop = early_stop

        self._dev_evaluator = dev_evaluator

        self._evaluation_steps = evaluation_steps
        self._save_best_model = save_best_model
        self._max_grad_norm = max_grad_norm

        os.makedirs(output_path, exist_ok=True)
        if os.listdir(output_path) and not restore_training:
            out = input(
                "Output directory ({}) already exists and is not empty, you wanna remove it before start? (y/n)".format(
                    output_path))
            if out == "y":
                shutil.rmtree(output_path)
                os.makedirs(output_path, exist_ok=True)
            else:
                raise ValueError("Output directory ({}) already exists and is not empty".format(
                    output_path))

        logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fileHandler = logging.FileHandler(os.path.join(output_path, "log.out"), mode="a")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        self._dev_evaluator.reset_logger(output_path)

        self.output_path = output_path

        if device is None or device == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                n_gpu = 1 if n_gpu == 1 else torch.cuda.device_count()
            else:
                logger.warning("no cuda is found in your machine, now use cpu")
                device = torch.device("cpu")
                n_gpu = 0
        elif device == "cpu":
            device = torch.device("cpu")
            n_gpu = 0
        else:
            raise ValueError("set device to be None, cuda or cpu")
        assert n_gpu <= torch.cuda.device_count()

        logger.info("Use pytorch device: {}, with gpu_number={}".format(device, n_gpu))

        self._train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        self._dev_batch_size = dev_batch_size if dev_batch_size != -1 else self._train_batch_size

        if isinstance(train_dataset, data.IterableDataset):
            self._train_dataloader = DataLoader(train_dataset, batch_size=None)
            self._steps_per_epoch = len(self._train_dataloader.dataset)
        else:
            self._train_dataloader = DataLoader(train_dataset, shuffle=data_loader_shuffle,
                                                batch_size=self._train_batch_size)
            self._steps_per_epoch = len(self._train_dataloader)

        if isinstance(dev_dataset, data.IterableDataset):
            dev_dataloader = DataLoader(dev_dataset, batch_size=None)
        else:
            dev_dataloader = DataLoader(dev_dataset, shuffle=data_loader_shuffle, batch_size=self._dev_batch_size)

        if accumulation_steps > 1:
            self._steps_per_epoch = self._steps_per_epoch // accumulation_steps

        self._dev_data = dev_dataset
        self._dev_evaluator.reset_dataloader(dev_dataloader)

        self.collate_fn = CollateFunction(self.up_model)
        # Use customize batching
        self._train_dataloader.collate_fn = self.collate_fn

        self._train_data = train_dataset
        self._per_gpu_train_batch_size = per_gpu_train_batch_size

        set_seed(seed, n_gpu)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[int(i) for i in visiable_device.split(',')])
            self.model = self.model.to(f'cuda:{self.model.device_ids[0]}')

        elif n_gpu == 1:
            self.model = self.model.to(device)

        self._device = device
        self._n_gpu = n_gpu

        self._total_train_steps = int(self._steps_per_epoch * epochs)
        self._epochs = epochs

        if report_model:
            count_params(self.model, print_details=True)

        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if local_rank != -1:
            self._total_train_steps = self._total_train_steps // torch.distributed.get_world_size()

        self._optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        warmup_steps = math.ceil(self._total_train_steps * warmup_ratio)  # by default 20% of train data for warm-up
        logger.info(f"   Warmup-steps: {warmup_steps}")

        self._scheduler = self._get_scheduler(self._optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                              num_total=self._total_train_steps)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(self.model, self._optimizer, opt_level=fp16_opt_level)
            self.model = model
            self._optimizer = optimizer

        self._fp16 = fp16
        tb_writer = None
        if local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        self._tb_writer = tb_writer
        self._local_rank = local_rank
        self._best_score = -float("inf")
        self._early_stop_count = 0
        self.last_time = datetime.now()
        self.accumulation_steps = accumulation_steps
        # assert evaluation_steps % accumulation_steps == 0, "evaluation_steps should be divisable by accumulation_steps"

    def _train_epoch(self, epoch: int, global_steps: int):
        epoch_steps = 0
        epoch_loss = 0.0

        self.model.zero_grad()
        for step, data in enumerate(
                tqdm(self._train_dataloader, desc="training", total=self._steps_per_epoch * self.accumulation_steps)):

            self.model.train()
            if data["labels"] != "skip-device":
                input, labels = batch_to_device(data, self._device)
                # add labels to input for training where this step is ignored when inference
                if isinstance(labels, dict):
                    for idx in range(len(input)):
                        input[idx].update(labels)
            else:
                input = data["features"]
            loss_value, _ = self.model(input)

            if self._n_gpu > 1:
                loss_value = loss_value.mean()
            if self.accumulation_steps > 1:
                loss_value = loss_value / self.accumulation_steps

            if self._fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss_value, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(self._optimizer), self._max_grad_norm)
            else:
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
            epoch_loss += loss_value

            if (step + 1) % self.accumulation_steps == 0:

                self._optimizer.step()
                self._scheduler.step()
                self.model.zero_grad()

                epoch_steps += 1
                total_global = epoch_steps + global_steps

                if self._evaluation_steps > 0 and (total_global) % self._evaluation_steps == 0:
                    dev_loss, eval_scores = self._dev_eval_in_training(epoch, epoch_steps)
                    logger.info("   ***** Evaluation report *****")
                    logger.info(f"  Output path (short): {self.output_path}")
                    logger.info(f"  Early stop on: {self._dev_evaluator.early_stop_on}")
                    logger.info(f"  Early stop count = {self._early_stop_count}/{self.early_stop}")
                    logger.info(
                        f"  Eval steps = {self._evaluation_steps} or (iterations = {self._evaluation_steps * self.accumulation_steps})")
                    logger.info(f"  Best score ({self._dev_evaluator.early_stop_on}) = {self._best_score}")
                    logger.info(f"  Gradient Accumulation steps = {self.accumulation_steps}")

                    logger.info(
                        f"  Num of training examples (actually no. of iterations per epoch for Iterable Dataset)  = {len(self._train_data)}")
                    logger.info(
                        f"  Num of development examples (actually no. of iterations per epoch for Iterable Dataset) = {len(self._dev_data)}")
                    now_time = datetime.now()
                    logger.info(f"  Time spent since last evaluation = {self.time_diff(self.last_time, now_time)}")
                    self.last_time = now_time

                    logger.info(f"  Epoch = {epoch + 1}/{self._epochs}")
                    logger.info(f"  Steps = {total_global}/{self._total_train_steps}")
                    logger.info(
                        f"  Instantaneous batch size per GPU = {self._per_gpu_train_batch_size} and n_gpu = {self._n_gpu} so the input batch size = {self._train_batch_size}")
                    if dev_loss != None:
                        logger.info(f"  dev_loss = {dev_loss:.6f}\t||\t dev_eval_scores = {eval_scores}")
                    else:
                        logger.info(f"  dev_eval_scores = {eval_scores}")

                    train_loss = epoch_loss / epoch_steps
                    logger.info(f"  train_loss = {train_loss}")
                    logger.info("\n********************************************")

                    if is_wandb_available() and self.wandb_config != None:
                        if dev_loss != None:
                            wandb.log(
                                {"loss_dev": dev_loss,
                                 f"best_score_for_{self._dev_evaluator.early_stop_on}": self._best_score,
                                 "loss_train": train_loss, "lr": self._scheduler.get_lr()[0]},
                                step=total_global)
                        else:
                            wandb.log({"loss_train": train_loss,
                                       f"best_score_for_{self._dev_evaluator.early_stop_on}": self._best_score,
                                       "lr": self._scheduler.get_lr()[0]},
                                      step=total_global)

                    for key, value in eval_scores.items():
                        if is_wandb_available() and self.wandb_config != None:
                            wandb.log({f"eval_{key}_dev": value}, step=total_global)
                        self._tb_writer.add_scalar(f"eval_{key}_dev", value, total_global)

                    self._tb_writer.add_scalar("lr", self._scheduler.get_lr()[0], total_global)
                    if dev_loss != None:
                        self._tb_writer.add_scalar("loss_dev", dev_loss, total_global)

                    self._tb_writer.add_scalar("loss_train", train_loss, total_global)

                    if self._early_stop_count >= self.early_stop:
                        logger.info(
                            f"  Continuous {self.early_stop} evaluation steps without loss reduction, so early stopped...")
                        sys.exit(0)

        return epoch_loss, epoch_steps

    def train(self):
        if self._restore_training:
            logger.info(f"***** restoring training from the previous checkpoint: {self.ck_report}*****")
        else:
            logger.info("***** Running training *****")
        logger.info(
            f"  Num of training examples (actually iterations per epoch for Iterable Dataset) = {len(self._train_data)}")
        logger.info(f"  Output path (short): {self.output_path}")
        logger.info(
            f"  Steps per Epoch = {self._steps_per_epoch} or iterations per epoch = {self._steps_per_epoch * self.accumulation_steps}")
        logger.info(f"  Num of Epochs = {self._epochs}")
        logger.info(f"  Best score ({self._dev_evaluator.early_stop_on}) = {self._best_score}")
        logger.info(
            f"  Eval every {self._evaluation_steps} steps or every {self._evaluation_steps * self.accumulation_steps} iterations")
        logger.info(f"  Early stop = {self.early_stop}")
        logger.info(f"  Gradient Accumulation steps = {self.accumulation_steps}")

        logger.info(f"  Total optimization steps = {self._total_train_steps}")
        logger.info(
            f"  Instantaneous batch size per GPU = {self._per_gpu_train_batch_size} and n_gpu = {self._n_gpu} so the input batch size = {self._train_batch_size}")
        global_loss = 0.0
        global_steps = 0
        self.last_time = datetime.now()
        for epoch in trange(self._epochs, desc="Epoch"):
            epoch_loss, epoch_steps = self._train_epoch(epoch, global_steps)
            global_loss += epoch_loss
            global_steps += epoch_steps
            logger.info(f"epoch {epoch + 1} ends, {self._epochs - epoch - 1} epoches left")
            logger.info(
                f"\nglobal_average_loss={global_loss / global_steps},global_steps={global_steps} on training set")

        if self._local_rank in [-1, 0]:
            self._tb_writer.close()

    def _dev_eval_in_training(self, epoch, steps):
        return_scores = {}
        if self._dev_evaluator is not None:

            return_scores = self._dev_evaluator(self.model, self.collate_fn,
                                                output_path=self.output_path, epoch=epoch, steps=steps)

            early_stop_on = self._dev_evaluator.early_stop_on

            check_score = -return_scores[early_stop_on] if early_stop_on == "loss" or early_stop_on == "perplexity" else \
                return_scores[early_stop_on]
            if check_score >= self._best_score and self._save_best_model:
                eval_scores_transformed = {key:
                                               return_scores[key].item() if torch.is_tensor(return_scores[key]) else
                                               return_scores[key]
                                           for key in return_scores.keys()}
                self.save(self.output_path,
                          {"training_examples (when pos_num=1 for ranking)": len(self._train_data),
                           "evaluation_steps": self._evaluation_steps,
                           "train_batch_size": self._train_batch_size, "epoch": epoch + 1, "total_epochs": self._epochs,
                           "steps": steps,
                           "saved_at_total_steps": steps + epoch * self._steps_per_epoch,
                           "steps_per_epoch": self._steps_per_epoch, "eval_scores_on_dev": eval_scores_transformed})

                self._best_score = check_score

                logger.info(f"  Save check-point at epoch={epoch} step={steps}")
                self._early_stop_count = 0
            else:
                self._early_stop_count += 1

        return return_scores.pop("loss").item() if "loss" in return_scores else None, return_scores

    def save(self, path, eval_details):
        if path is None:
            return
        logger.info(f"   Save model to {path}")
        contained_modules = []

        to_iterate = self.model.module._modules if self._n_gpu > 1 else self.model._modules

        for idx, name in enumerate(to_iterate):
            module = to_iterate[str(name)]

            model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append(
                {'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        if self.wandb_config != None:
            with open(os.path.join(path, 'hyperparams.json'), 'w') as f:
                json.dump(self.wandb_config.__dict__, f, indent=2)

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)
        with open(os.path.join(path, 'ck_report.json'), 'w') as fOut:
            json.dump(eval_details, fOut, indent=2)

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, num_total: int):
        assert scheduler in ["constantlr", "warmuplinear", "warmupconstant", "warmupcosine",
                             "warmupcosinewithhardrestarts"], (
            'scheduler should be one of ["constantlr","warmupconstant","warmupcosine","warmupcosinewithhardrestarts"]')
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=num_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=num_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                   num_warmup_steps=warmup_steps,
                                                                                   num_training_steps=num_total)

    def time_diff(self, t_a, t_b):
        t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
        return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
