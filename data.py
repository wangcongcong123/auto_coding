from torch.utils.data import Dataset
import os, pickle, json
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm

class SrcCodeDataset(Dataset):
    def __init__(self, file_path, model, cache_path=None):
        """
        this dataset class is used to load source code dataset in batch for fine-tuning with GPT2LMModel
        :param model: the model that the dataset will be fed to
        """
        self.inputs = []
        load_cache = False
        if cache_path != None:
            load_cache = self._load_cache(cache_path)
        if not load_cache:
            self._build(file_path, model)
        if cache_path != None:
            self._cache(cache_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"]
        # input_mask = self.inputs[index]["attention_mask"] we don't need attention_mask for this task
        # return {"input_ids": input_ids, "input_mask": input_mask}
        return {"input_ids": input_ids}

    def _load_cache(self, cache_path):
        load_cache = False
        if os.path.isdir(cache_path):
            if os.path.isfile(os.path.join(cache_path, "inputs.pk")):
                with open(os.path.join(cache_path, "inputs.pk"), "rb") as f:
                    logger.info(
                        f"  load cached token ids of model from {cache_path}")
                    self.inputs = pickle.load(f)
                    load_cache = True
        return load_cache

    def _cache(self, cache_path):
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        with open(os.path.join(cache_path, "inputs.pk"), "wb") as f:
            pickle.dump(self.inputs, f)
            logger.info(
                f"  save tokenized ids of samples to: {cache_path}/inputs.pk")

    def _build(self, file_path, model):
        with open(file_path) as f:
            for line in tqdm(f):
                example = json.loads(line.strip())
                if example["label"].lower() == "python":
                    encoded_plus = model.tokenizer.encode_plus(
                        model.tokenize("<python>") + example["token_ids"] + [model.eos_token_id],
                        max_length=model.max_seq_length)
                elif example["label"].lower() == "java":
                    encoded_plus = model.tokenizer.encode_plus(
                        model.tokenize("<java>") + example["token_ids"] + [model.eos_token_id],
                        max_length=model.max_seq_length)
                self.inputs.append(encoded_plus.data)
