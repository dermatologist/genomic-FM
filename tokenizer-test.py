# Description: This script compares the performance of different tokenizers
# on the lung cancer dataset using a BERT architecture, training from scratch.
import os
import sys

import pandas as pd
import pytorch_lightning as pl
from torchmetrics.classification import AUROC
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
from src.dataloader.data_wrapper import ClinVarDataWrapper

DISEASE_SUBSET = ['Lung_cancer',
                  'EGFR-related_lung_cancer',
                  'Lung_carcinoma',
                  'Autoimmune_interstitial_lung_disease-arthritis_syndrome',
                  'Global_developmental_delay_-_lung_cysts_-_overgrowth_-_Wilms_tumor_syndrome',
                  'Small_cell_lung_carcinoma',
                  'Chronic_lung_disease',
                  'Lung_adenocarcinoma',
                  'Lung_disease',
                  'Non-small_cell_lung_carcinoma', 'LUNG_CANCER', 'Squamous_cell_lung_carcinoma']


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        # self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model = model_name
        self.lr = lr
        self.auroc = AUROC(num_classes=num_labels, task='multiclass')

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(outputs, batch['labels'])
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(outputs, batch['labels'])
        # Calculate AUROC
        logits = torch.as_tensor(outputs)
        labels = torch.as_tensor(batch['labels'])
        self.auroc(logits, labels)
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'preds': outputs.argmax(dim=1), 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_accuracy', accuracy, sync_dist=True)
        self.log('val_f1', f1, sync_dist=True)
        # Log the validation AUROC
        auroc = self.auroc.compute()
        self.log('val_auroc', auroc)
        self.auroc.reset()

    def test_step(self, batch, batch_idx, **kwargs):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(outputs, batch['labels'])
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'preds': outputs.argmax(dim=1), 'labels': batch['labels']}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_accuracy', accuracy, sync_dist=True)
        self.log('val_f1', f1, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def process_data(data):
    new_data = []
    for x, y in tqdm(data, desc="Appending sequences"):
        new_data.append([x[0], x[1], x[2], y])
    return new_data


def get_df(seq_length=512):
    file_path = f'output/lung_cancer_{seq_length}.pkl'
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
    else:
        DATA = ClinVarDataWrapper()
        data = DATA.get_data(Seq_length=seq_length, target='CLNDN', disease_subset=True)
        processed_data = process_data(data)
        # create a pandas dataframe from the processed data
        df = pd.DataFrame(processed_data, columns=['ref', 'alt', 'annotation', 'label'])
        # Save the DataFrame to a pickle file
        df.to_pickle(file_path)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df


if __name__ == '__main__':
    # System
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # GPU
    accel = 'unknown'
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        device = torch.cuda.get_device_name(0)
        accel = 'cuda'
    # if mps is available
    elif torch.backends.mps.is_available():
        gpus = 1
        device = torch.device("mps")
        accel = 'mps'
    else:
        gpus = 0
        device = torch.device("cpu")
    num_workers = 4
    batch_size = 12  # 8
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

    run_name = f"{sys.argv[1]} - {seq_max_length}"
    wandb_logger = WandbLogger(name=run_name, project="Tokenizer comparison")

    trainer_args = {
        'max_epochs': epochs,
        'logger': wandb_logger
    }
    # Using `DistributedSampler` with the dataloaders. During `trainer.test()`,
    # it is recommended to use `Trainer(devices=1, num_nodes=1)`
    # to ensure each sample/batch gets evaluated exactly once.
    # Otherwise, multi-device settings use `DistributedSampler`
    # that replicates some samples to make sure all devices have same batch size in case of uneven inputs.
    if gpus >= 1:
        trainer_args['accelerator'] = 'gpu'
        trainer_args['devices'] = gpus
        trainer_args['num_nodes'] = 1
        if accel != 'mps':
            trainer_args['strategy'] = 'ddp'
    else:
        trainer_args['accelerator'] = 'cpu'

    sentences = df['alt'].tolist()
    labels = df['label'].tolist()

    # convert 'Lung_cancer' to 1 and Other_disease to 0
    labels = [1 if label in DISEASE_SUBSET else 0 for label in labels]

    dataset = TextDataset(sentences, labels, tokenizer, max_length=max_model_length)  # max_model_length


    for index, sentence in enumerate(sentences):
        print(sentence)
        tokenized = tokenizer(sentence, truncation=True, padding='max_length', max_length=max_model_length)
        print(tokenized)
        if index > 2:
            break
