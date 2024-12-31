# Description: This script compares the performance of different tokenizers
# on the lung cancer dataset using a BERT architecture, training from scratch.
import os
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from genomic_tokenizer import GenomicTokenizer
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import AUROC, MatthewsCorrCoef
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from embedding.hg38_char_tokenizer import CharacterTokenizer
from embedding.tokenization_dna import DNATokenizer
from src.dataloader.bell_wrapper import ClinVarDataWrapper

DISEASE_SUBSET = [
    "Lung_cancer",
    "EGFR-related_lung_cancer",
    "Lung_carcinoma",
    "Autoimmune_interstitial_lung_disease-arthritis_syndrome",
    "Global_developmental_delay_-_lung_cysts_-_overgrowth_-_Wilms_tumor_syndrome",
    "Small_cell_lung_carcinoma",
    "Chronic_lung_disease",
    "Lung_adenocarcinoma",
    "Non-small_cell_lung_carcinoma",
    "LUNG_CANCER",
    "Squamous_cell_lung_carcinoma",
]

# convert the classes to integers
DISEASE_KEYS = {
    "Lung_cancer": 10,
    "EGFR-related_lung_cancer": 1,
    "Lung_carcinoma": 2,
    "Autoimmune_interstitial_lung_disease-arthritis_syndrome": 3,
    "Global_developmental_delay_-_lung_cysts_-_overgrowth_-_Wilms_tumor_syndrome": 4,
    "Small_cell_lung_carcinoma": 5,
    "Chronic_lung_disease": 6,
    "Lung_adenocarcinoma": 7,
    "Non-small_cell_lung_carcinoma": 8,
    "LUNG_CANCER": 9,
    "Squamous_cell_lung_carcinoma": 11,
}

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
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr=2e-5):
        super().__init__()
        # self.model = BertForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
        self.model = model_name
        self.lr = lr
        self.auroc = AUROC(num_classes=num_classes, task="multiclass")
        self.matthews_corr = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])
        # Calculate AUROC
        logits = torch.as_tensor(outputs)
        labels = torch.as_tensor(batch["labels"])
        self.auroc(logits, labels)
        self.matthews_corr(logits.argmax(dim=1), labels)
        self.log("val_loss", loss, sync_dist=True)
        return {
            "val_loss": loss,
            "preds": outputs.argmax(dim=1),
            "labels": batch["labels"],
        }

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted")
        self.log("val_accuracy", accuracy, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)
        # Log the validation AUROC
        auroc = self.auroc.compute()
        matthews_corr = self.matthews_corr.compute()
        self.log("val_matthews_corr", matthews_corr, sync_dist=True)
        self.log("val_auroc", auroc, sync_dist=True)
        self.auroc.reset()

    def test_step(self, batch, batch_idx, **kwargs):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])
        self.log("val_loss", loss, sync_dist=True)
        return {
            "val_loss": loss,
            "preds": outputs.argmax(dim=1),
            "labels": batch["labels"],
        }

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted")
        self.log("val_accuracy", accuracy, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def process_data(data):
    new_data = []
    for x, y in tqdm(data, desc="Appending sequences"):
        new_data.append([x[0], x[1], x[2], y])
    return new_data


def get_df(seq_length=512):
    file_path = f"output/lung_cancer_mc_{seq_length}.pkl"
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
    else:
        DATA = ClinVarDataWrapper(
            my_subset=DISEASE_SUBSET,
            percent=50,
        )
        data = DATA.get_data(Seq_length=seq_length, target="CLNDN", disease_subset=True, multi_class=True)
        processed_data = process_data(data)
        # create a pandas dataframe from the processed data
        df = pd.DataFrame(processed_data, columns=["ref", "alt", "annotation", "label"])
        # Save the DataFrame to a pickle file
        df.to_pickle(file_path)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def labels_to_int(labels):
    for i, label in enumerate(labels):
        if label in DISEASE_KEYS:
            labels[i] = DISEASE_KEYS[label]
        else:
            labels[i] = 0
    return labels


if __name__ == "__main__":
    # System
    torch.set_float32_matmul_precision("medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # GPU
    accel = "unknown"
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        accel = "cuda"
    # if mps is available
    elif torch.backends.mps.is_available():
        gpus = 1
        device = torch.device("mps")
        accel = "mps"
    else:
        gpus = 0
        device = torch.device("cpu")
    num_workers = 4
    batch_size = 12  # 8
    # create chkpoint directory in /tmp if it does not exist
    if not os.path.exists("/tmp/checkpoints"):
        os.makedirs("/tmp/checkpoints")
    tmpdir = "/tmp/checkpoints/"
    # Delete all files in the directory
    for file in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, file))
    # Sequence length
    seq_max_length = int(sys.argv[2])
    max_model_length = 512
    # DNABert2
    model_name = "zhihan1996/DNABERT-2-117M"
    # Parameters
    epochs = 1
    num_classes = 12

    if len(sys.argv) < 2:
        print("Please provide a tokenizer to use")
        exit(0)

    if sys.argv[1] == "gt":
        tokenizer = GenomicTokenizer(model_max_length=seq_max_length, introns=False)
    elif sys.argv[1] == "dnab":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=seq_max_length, trust_remote_code=True
        )
    elif sys.argv[1] == "hyena":
        tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add DNA characters
            model_max_length=seq_max_length,
        )
    elif sys.argv[1] == "3mer":
        vocab_file = "embedding/vocab3.txt"
        tokenizer = DNATokenizer(vocab_file, max_len=seq_max_length)
    elif sys.argv[1] == "6mer":
        vocab_file = "embedding/vocab6.txt"
        tokenizer = DNATokenizer(vocab_file, max_len=seq_max_length)
    else:
        print("Invalid tokenizer, please choose between 'gt' or 'dnab'")
        exit(0)

    run_name = f"{sys.argv[1]} - {seq_max_length}"
    wandb_logger = WandbLogger(name=run_name, project="Comparison - multiclass")

    trainer_args = {"max_epochs": epochs, "logger": wandb_logger}
    # Using `DistributedSampler` with the dataloaders. During `trainer.test()`,
    # it is recommended to use `Trainer(devices=1, num_nodes=1)`
    # to ensure each sample/batch gets evaluated exactly once.
    # Otherwise, multi-device settings use `DistributedSampler`
    # that replicates some samples to make sure all devices have same batch size in case of uneven inputs.
    if gpus >= 1:
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = gpus
        trainer_args["num_nodes"] = 1
        if accel != "mps":
            trainer_args["strategy"] = "ddp"
    else:
        trainer_args["accelerator"] = "cpu"

    # Load the data, if exists, else create it
    try:
        train_df = pd.read_pickle(f"output/train_{seq_max_length}.pkl")
        val_df = pd.read_pickle(f"output/val_{seq_max_length}.pkl")
        test_df = pd.read_pickle(f"output/test_{seq_max_length}.pkl")
        print("Data loaded from pickle files")
    except:
        print("Data not found, creating new dataframes")
        df = get_df(seq_max_length)
        # Split the dataframe into training, validation and test sets
        train_df, val_df, test_df = (
            df[: int(0.6 * len(df))],
            df[int(0.6 * len(df)) : int(0.7 * len(df))],
            df[int(0.7 * len(df)) :],
        )

        # save the dataframes to pickle files
        train_df.to_pickle(f"output/train_{seq_max_length}.pkl")
        val_df.to_pickle(f"output/val_{seq_max_length}.pkl")
        test_df.to_pickle(f"output/test_{seq_max_length}.pkl")

    train_dataset = TextDataset(
        train_df["alt"].tolist(),
        labels_to_int(train_df["label"].tolist()),
        tokenizer,
        max_length=max_model_length,
    )
    val_dataset = TextDataset(
        val_df["alt"].tolist(),
        labels_to_int(val_df["label"].tolist()),
        tokenizer,
        max_length=max_model_length,
    )
    test_dataset = TextDataset(
        test_df["alt"].tolist(),
        labels_to_int(test_df["label"].tolist()),
        tokenizer,
        max_length=max_model_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    #  Initiate model from scratch
    # max_model_length = 512 for pre-trained BERT, but we can change it as we are training from scratch
    # config = BertConfig(
    #     vocab_size=tokenizer.vocab_size,
    #     max_position_embeddings=max_model_length,
    #     hidden_size=128,
    #     num_attention_heads=2,
    #     num_hidden_layers=2,
    #     hidden_dropout_prob=0.1,
    #     attention_probs_dropout_prob=0.1,
    #     num_classes=2,
    #     seed=42
    # )
    # config = BertConfig.from_pretrained('bert-base-uncased')

    #     BertConfig {
    #   "architectures": [
    #     "BertForMaskedLM"
    #   ],
    #   "attention_probs_dropout_prob": 0.1,
    #   "classifier_dropout": null,
    #   "gradient_checkpointing": false,
    #   "hidden_act": "gelu",
    #   "hidden_dropout_prob": 0.1,
    #   "hidden_size": 768,
    #   "initializer_range": 0.02,
    #   "intermediate_size": 3072,
    #   "layer_norm_eps": 1e-12,
    #   "max_position_embeddings": 512,
    #   "model_type": "bert",
    #   "num_attention_heads": 12,
    #   "num_hidden_layers": 12,
    #   "pad_token_id": 0,
    #   "position_embedding_type": "absolute",
    #   "transformers_version": "4.47.1",
    #   "type_vocab_size": 2,
    #   "use_cache": true,
    #   "vocab_size": 30522
    # }

    config = BertConfig(
        architecture=["BertForSequenceClassification"],
        attention_probs_dropout_prob=0,
        classifier_dropout=0,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        hidden_size=192,  # 768  # 15.2 M -> 9 M
        intermediate_size=768,  # 3072  9M -> 4.8 M
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=3,  #  12  # 86.1 M -> 15.2 M
        num_hidden_layers=3,  #  12
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type="absolute",
        transformers_version="4.47.1",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=tokenizer.vocab_size,
        seed=42,
        # "architectures": [
        #     "BertForSequenceClassification"
        # ],
        # "attention_probs_dropout_prob": 0.1,
        # "classifier_dropout": 0.1,
        # "gradient_checkpointing": false,
        # "hidden_act": "gelu",
        # "hidden_dropout_prob": 0.1,
        # "hidden_size": 128,
        # "initializer_range": 0.02,
        # "intermediate_size": 512,
        # "layer_norm_eps": 1e-12,
        # "max_position_embeddings": 512,
        # "model_type": "bert",
        # "num_attention_heads": 4,
        # "num_hidden_layers": 4,
        # "pad_token_id": 0,
        # "position_embedding_type": "absolute",
        # "transformers_version": "4.47.1",
        # "type_vocab_size": 2,
        # "use_cache": true,
        # "vocab_size": tokenizer.vocab_size,
    )
    # print(config)
    # exit(0)
    _model = BertForSequenceClassification(config)

    # Modify learning rate
    # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
    model = BertClassifier(_model, num_classes=num_classes, lr=3e-4)

    if gpus >= 1:
        model = model.to(device)

    chpt = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=tmpdir,
        filename=sys.argv[1],  # extension is added automatically
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer_args["callbacks"] = [chpt]
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(ckpt_path='output/best.ckpt', dataloaders=val_loader)

    ckpt_path = os.path.join(tmpdir, sys.argv[1] + ".ckpt")
    trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)
