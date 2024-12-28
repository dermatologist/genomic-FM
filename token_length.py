import os
import sys

import pandas as pd
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, MatthewsCorrCoef
import torch
from genomic_tokenizer import GenomicTokenizer
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (AutoTokenizer, BertConfig,
                          BertForSequenceClassification)

from embedding.hg38_char_tokenizer import CharacterTokenizer
from embedding.tokenization_dna import DNATokenizer
from src.dataloader.bell_wrapper import ClinVarDataWrapper


def get_df(seq_length=512):
    file_path = f'output/lung_cancer_{seq_length}.pkl'
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
    else:
        # Thow error if the file does not exist
        raise FileNotFoundError(f'File {file_path} does not exist')
    df = df.sample(frac=1).reset_index(drop=True)
    return df

seq_max_length = int(sys.argv[2])
df = get_df(seq_max_length)
# DNABert2
model_name = "zhihan1996/DNABERT-2-117M"

if len(sys.argv) < 2:
    print("Please provide a tokenizer to use")
    exit(0)

if sys.argv[1] == 'gt':
    tokenizer = GenomicTokenizer(model_max_length=seq_max_length, introns=False)
elif sys.argv[1] == 'dnab':
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                model_max_length=seq_max_length,
                                                trust_remote_code=True)
elif sys.argv[1] == 'hyena':
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
        model_max_length=seq_max_length,
    )
elif sys.argv[1] == '3mer':
    vocab_file = 'embedding/vocab3.txt'
    tokenizer = DNATokenizer(vocab_file, max_len=seq_max_length)
elif sys.argv[1] == '6mer':
    vocab_file = 'embedding/vocab6.txt'
    tokenizer = DNATokenizer(vocab_file, max_len=seq_max_length)
else:
    print("Invalid tokenizer, please choose between 'gt' or 'dnab'")
    exit(0)

sentences = df["alt"].tolist()

# find avenrage length of input_ids for all sentences
lengths = []
for sentence in sentences:
    encoded = tokenizer(sentence)
    lengths.append(len(encoded["input_ids"]))
print(f"Average token size: {int(sum(lengths) / len(lengths))}")
