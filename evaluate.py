from typing import List, Dict
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SingleCLMEvaluator():
    def __init__(self, dataloader: DataLoader = None,
                 data_tag: str = "dev",
                 device: int = None, tokenizer=None, early_stop_on: str = "perplexity"):

        if data_tag not in ["dev", "train", "test"]:
            raise ValueError("data_tag has to be one of dev, train or test")
        assert early_stop_on in ["loss", "perplexity"]
        self.early_stop_on = early_stop_on
        self.dataloader = dataloader
        self.data_tag = data_tag
        self.tokenizer = tokenizer

        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == -1:
            self.n_gpu = 0
            self.device = torch.device("cpu")

    def reset_dataloader(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def reset_logger(self, output_path):
        pass

    def __call__(self, model, collate_fn, output_path: str = None, epoch: int = -1, steps: int = -1,
                 target_names: List[str] = None, do_predict: bool = False) -> Dict[
        str, float]:

        if do_predict and self.tokenizer == None:
            raise ValueError("you are doing predict so need a tokenizer")
        if self.dataloader is None:
            raise ValueError(" need to set dataloader for this evaluator, call reset_dataloader()")

        model.eval()
        if epoch == -1 and steps == -1:
            logger.info(
                f"\nEvaluation the model on {self.data_tag} dataset")
        else:
            logger.info(
                "\nEvaluation the model on " + self.data_tag + " dataset" + f" in epoch {epoch} after {steps} steps:")

        self.dataloader.collate_fn = collate_fn
        total_loss = 0.0
        total_steps = 0

        for step, batch in enumerate(tqdm(self.dataloader, desc="evaluating")):
            input = batch["features"]
            # batch to device
            for feature_name, ids in input.items():
                input[feature_name] = ids.to(self.device)

            with torch.no_grad():
                loss, logits = model(input)
                loss = loss.mean()
                total_loss += loss

            total_steps += 1
        eval_loss = total_loss / total_steps
        eval_results = {"loss": eval_loss}

        perplexity = torch.exp(torch.tensor(eval_loss)).clone().detach()
        eval_results["perplexity"] = perplexity.mean().item()
        return eval_results
