# https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset


class LowercaseAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.ep = 1e-5

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.lowercase_nlloss(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / (x['lowercase_nlloss'] + self.ep)})
        return dataset

    def lowercase_nlloss(self, batch):
        texts = [x.lower() for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts, 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask,
                                shift_logits=self.config['model_shift_logits'],
                                mask_id=self.config['model_mask_id'])
        return {'lowercase_nlloss': losses}

