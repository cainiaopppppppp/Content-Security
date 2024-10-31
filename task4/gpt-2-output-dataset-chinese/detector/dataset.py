import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False, max_texts=None):
        self.name = name
        self.train = self.load_texts(f'{data_dir}/{name}.train.json', max_texts) if not skip_train else None
        self.test = self.load_texts(f'{data_dir}/{name}.test.json', max_texts)
        self.valid = self.load_texts(f'{data_dir}/{name}.valid.json', max_texts)

    # 读取文件
    def load_texts(self, data_file, max_texts=None, expected_size=None):
        texts = []
        with open(data_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in tqdm(data[:max_texts], total=(expected_size if expected_size is not None else max_texts), desc=f'Loading {data_file}'):
                texts.append(item['content'])
        return texts


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: BertTokenizer,
                 max_sequence_length: int = 512, epoch_size: int = None, token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        label = self.random.randint(2) if self.epoch_size is not None else (1 if index < len(self.real_texts) else 0)
        text = self.real_texts[index] if label == 1 else self.fake_texts[index - len(self.real_texts)]

        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_sequence_length, padding='max_length')
        mask = [float(token != self.tokenizer.pad_token_id) for token in tokens]

        return torch.tensor(tokens), torch.tensor(mask), label
