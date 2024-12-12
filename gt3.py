import torch
import os
import sys
from src.dataloader.data_wrapper import (
    ClinVarDataWrapper
)
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from genomic_tokenizer import GenomicTokenizer
import pandas as pd
from tqdm import tqdm


DISEASE_SUBSET = ['Lung_cancer','EGFR-related_lung_cancer','Lung_carcinoma','Autoimmune_interstitial_lung_disease-arthritis_syndrome','Global_developmental_delay_-_lung_cysts_-_overgrowth_-_Wilms_tumor_syndrome','Small_cell_lung_carcinoma','Chronic_lung_disease','Lung_adenocarcinoma','Lung_disease','Non-small_cell_lung_carcinoma','LUNG_CANCER','Squamous_cell_lung_carcinoma']

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
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
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
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'preds': outputs.argmax(dim=1), 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_accuracy', accuracy, sync_dist=True)
        self.log('val_f1', f1, sync_dist=True)

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

# Example usage
# texts = ["This is a positive sentence.", "This is a negative sentence."]
# labels = [1, 0]
def get_df():
    file_path = 'output/lung_cancer.pkl'
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
    else:
        DATA = ClinVarDataWrapper()
        data = DATA.get_data(Seq_length=512, target='CLNDN', disease_subset=True)
        processed_data = process_data(data)
        # create a pandas dataframe from the processed data
        df = pd.DataFrame(processed_data, columns=['ref', 'alt', 'annotation', 'label'])
        # Save the DataFrame to a pickle file
        df.to_pickle(file_path)
    return df


if __name__ == '__main__' :
    df = get_df()
    # Genomic tokenizer
    model_max_length = 512
    # DNABert2
    model_name = "zhihan1996/DNABERT-2-117M"

    if len(sys.argv) < 2:
        print("Please provide a tokenizer to use")
        exit(0)

    if sys.argv[1] == 'gt':
        tokenizer = GenomicTokenizer(model_max_length)
    elif sys.argv[1] == 'dnab':
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        print("Invalid tokenizer, please choose between 'gt' or 'dnab'")
        exit(0)

    sentences = df['alt'].tolist()
    labels = df['label'].tolist()

    # convert 'Lung_cancer' to 1 and Other_disease to 0
    labels = [1 if label in DISEASE_SUBSET else 0 for label in labels]


    dataset = TextDataset(sentences, labels, tokenizer, max_length=128)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)


    #  Initiate model from scratch
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        hidden_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=2,
    )

    _model = BertForSequenceClassification(config)

    model = BertClassifier(_model, num_labels=2)
    model = model.cuda()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=3
        )
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(ckpt_path='output/best.ckpt', dataloaders=val_loader)
    trainer.test(ckpt_path='best', dataloaders=test_loader)