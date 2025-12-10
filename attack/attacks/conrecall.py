# https://arxiv.org/pdf/2409.03363
import copy
import random
import logging
import numpy as np

import datasets
from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset, load_dataset


logging.basicConfig(level=logging.WARNING)

def make_conrecall_prefix(dataset, n_shots, perplexity_bucket=None, target_index=None):
    prefixes = []
    if target_index is not None:
        indices_to_keep = [i for i in range(len(dataset)) if i != target_index]
        dataset = dataset.select(indices_to_keep)

    if perplexity_bucket is not None:
        datasets.disable_progress_bar()
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
        datasets.enable_progress_bar()

    all_indices = list(range(len(dataset)))
    
    # Ensure n_shots is not larger than the available population
    n_shots = min(n_shots, len(all_indices))
    if n_shots > len(all_indices):
        logging.warning(f"Requested n_shots ({n_shots}) is larger than available population ({len(all_indices)}). Reducing n_shots to {len(all_indices)}.")

    indices = random.sample(all_indices, n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)


class ConrecallAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])

    def build_non_member_prefix(self, perplexity_bucket=None):
        return make_conrecall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def build_member_prefix(self, target_index, dataset, perplexity_bucket=None):
        return make_conrecall_prefix(
            dataset=dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket,
            target_index=target_index
        )

    def build_one_prefix(self, perplexity_bucket=None):
        return make_conrecall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def run(self, dataset: Dataset) -> Dataset:
        ds_clone = copy.deepcopy(dataset)
        dataset = dataset.map(
            lambda x: self.conrecall_nlloss(x, ds_clone),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v7",
        )
        dataset = dataset.map(lambda x: {self.name: (
            x[f'{self.name}_nm_nlloss'] - x[f'{self.name}_m_nlloss']) / x['nlloss']})
        return dataset

    def conrecall_nlloss(self, batch, dataset):
        if self.config["match_perplexity"]:
            it = enumerate(zip(batch["perplexity_bucket"], batch["text"]))
            non_member_texts = [
                self.build_non_member_prefix(ppl_bucket) + " " + text
                for i, (ppl_bucket, text) in it
            ]

            it = enumerate(zip(batch["perplexity_bucket"], batch["text"]))
            ds_members_only = dataset.filter(lambda x: x["label"] == 1)
            member_texts = [
                self.build_member_prefix(
                    perplexity_bucket=ppl_bucket,
                    target_index=i,
                    dataset=ds_members_only
                ) + " " + text
                for i, (ppl_bucket, text) in it
            ]
        else:
            non_member_texts = [self.build_non_member_prefix() + " " + text for text in batch["text"]]

            ds_members_only = dataset.filter(lambda x: x["label"] == 1)
            member_texts = [
                self.build_member_prefix(
                    target_index=i,
                    dataset=ds_members_only
                ) + " " + text
                for i, text in enumerate(batch["text"])
            ]

        ret = {}
        for texts, label in [(non_member_texts, "nm"), (member_texts, "m")]:
            tokenized = self.tokenizer.batch_encode_plus(
                            texts,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.get('max_length', 512),
                        )
            token_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            losses = compute_nlloss(self.model, token_ids, attention_mask,
                                    shift_logits=self.config['model_shift_logits'],
                                    mask_id=self.config['model_mask_id'])
            ret[f"{self.name}_{label}_nlloss"] = losses
        return ret

    def extract_features(self, batch):
        # 1. Basic recall feature with original prefix
        texts = [self.build_one_prefix() + " " + text for text in batch["text"]]
        recall_tokenized = self.tokenizer.batch_encode_plus(
                                texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.config.get('max_length', 512),
                            )
        recall_losses = compute_nlloss(
            self.model, 
            recall_tokenized['input_ids'].to(self.device),
            recall_tokenized['attention_mask'].to(self.device),
            shift_logits=self.config['model_shift_logits'],
            mask_id=self.config['model_mask_id']
        )
        basic_recall_feature = [-loss for loss in recall_losses]

        # 2. Contrastive recall features with different prefixes
        contrast_features = []
        for _ in range(self.config.get('n_contrasts', 3)):
            texts = [self.build_one_prefix() + " " + text for text in batch["text"]]
            tokenized = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.get('max_length', 512),
            )
            losses = compute_nlloss(
                self.model, 
                tokenized['input_ids'].to(self.device),
                tokenized['attention_mask'].to(self.device)
            )
            contrast_features.append([-loss for loss in losses])

        # Combine all features
        features = np.column_stack([basic_recall_feature] + contrast_features)
        return features
