# Description: This script compares the performance of different tokenizers on the lung cancer dataset using a BERT architecture, training from scratch.
import os
import random
import sys

import pandas as pd
from genomic_benchmarks.dataset_getters.pytorch_datasets import \
    HumanEnhancersCohn
from genomic_tokenizer import GenomicTokenizer
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from transformers import (AutoTokenizer)

from embedding.hg38_char_tokenizer import CharacterTokenizer
from embedding.tokenization_dna import DNATokenizer





def process_data(data):
    processed_data = []
    len_data = len(data)
    for record in tqdm(data, desc="Appending sequences"):
        # get 0 or 1 randomly
        label = random.choice([0, 1])
        if label == 0:
            processed_data.append([record, record, label])
        else:
            processed_data.append([mutate(record, 100), mutate(record, 100), label])
    return processed_data


def mutate(sequence, num_mutations):
    """
    Mutate the sequence by changing num_mutations bases
    Mutation can be substitution, insertion or deletion
    """
    bases = ['A', 'C', 'G', 'T']
    seq = list(sequence)
    for _ in range(num_mutations):
        mutation_type = random.choice(['substitution', 'insertion', 'deletion'])
        if mutation_type == 'substitution':
            # substitution
            idx = random.randint(0, len(seq) - 1)
            seq[idx] = random.choice([base for base in bases if base != seq[idx]])
        elif mutation_type == 'insertion':
            # insertion
            idx = random.randint(0, len(seq) - 1)
            seq.insert(idx, random.choice(bases))
        else:
            # deletion
            idx = random.randint(0, len(seq) - 1)
            seq.pop(idx)
    return ''.join(seq)

def get_data():
    dset = HumanEnhancersCohn(split='train', version=0)
    data = []
    for x, y in dset:
        data.append(x)
    return data

def get_df(seq_length=512):
    data = get_data()
    processed_data = process_data(data)
    # create a pandas dataframe from the processed data
    df = pd.DataFrame(processed_data, columns=['ref', 'alt', 'label'])
    # Save the DataFrame to a pickle file
    return df

if __name__ == '__main__' :
    # System
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gpus = 3
    # create chkpoint directory in /tmp if it does not exist
    if not os.path.exists('/tmp/checkpoints'):
        os.makedirs('/tmp/checkpoints')
    tmpdir = '/tmp/checkpoints/'
    # Delete all files in the directory
    for file in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, file))
    # Sequence length
    seq_max_length = int(sys.argv[2])
    max_model_length = 512
    df = get_df(seq_max_length)
    # DNABert2
    model_name = "zhihan1996/DNABERT-2-117M"
    # Parameters
    epochs = 1

    if len(sys.argv) < 2:
        print("Please provide a tokenizer to use")
        exit(0)

    if sys.argv[1] == 'gt':
        tokenizer = GenomicTokenizer(model_max_length=seq_max_length)
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

    run_name = sys.argv[1]
    wandb_logger = WandbLogger(name=run_name, project=f"Tokenizer comparison")

    trainer_args = {
        'max_epochs': epochs,
        'logger': wandb_logger
    }
    # Using `DistributedSampler` with the dataloaders. During `trainer.test()`,
    # it is recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once.
    # Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.
    if gpus >= 1:
        trainer_args['accelerator'] = 'gpu'
        trainer_args['devices'] = 1 # gpus
        trainer_args['num_nodes'] = 1
        trainer_args['strategy'] = 'ddp'
    else:
        trainer_args['accelerator'] = 'cpu'

    sentences = df['alt'].tolist()
    labels = df['label'].tolist()

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        print(f"Sentence: {sentence}")
        print(f"Tokens: {tokens}")
        exit(0)