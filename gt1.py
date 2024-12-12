"""
Real ClinVar" essentially refers to the most up-to-date, reliable information within the ClinVar database,
often implying a focus on clinically validated and well-established variant interpretations,
while "ClinVar" encompasses the entire database, which may include submissions with varying levels of evidence
and potential for conflicting interpretations depending on the source.


GV Record: Following this construction process, the minimum unit of GV-Rep dataset is a record,
which is an (x, y) pair. Here, x = (ref, alt, annotation), and y is the corresponding label indicating
the class of GV or a real value quantifying the effects of the GV.

"""
import os
from sklearn.model_selection import train_test_split
from src.dataloader.data_wrapper import (
    ClinVarDataWrapper
)
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments
from genomic_tokenizer import GenomicTokenizer
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import evaluate
import numpy as np

model_max_length = 512
g_tokenizer = GenomicTokenizer(model_max_length)

def tokenize_function(examples):
    return g_tokenizer(examples['alt'])

def process_data(data):
    new_data = []
    for x, y in tqdm(data, desc="Appending sequences"):
        new_data.append([x[0], x[1], x[2], y])
    return new_data

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model_name = "zhihan1996/DNABERT-2-117M"
dnabert2_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

DISEASE_SUBSET = ['Lung_cancer','EGFR-related_lung_cancer','Lung_carcinoma','Autoimmune_interstitial_lung_disease-arthritis_syndrome','Global_developmental_delay_-_lung_cysts_-_overgrowth_-_Wilms_tumor_syndrome','Small_cell_lung_carcinoma','Chronic_lung_disease','Lung_adenocarcinoma','Lung_disease','Non-small_cell_lung_carcinoma','LUNG_CANCER','Squamous_cell_lung_carcinoma']
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


# Create a train and eval dataset from the dataframe
train_df, eval_df = train_test_split(df, test_size=0.2)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

#  instantiate yourself
config = BertConfig(
    vocab_size=g_tokenizer.vocab_size,
    max_position_embeddings=512,
    hidden_size=128,
    num_attention_heads=2,
    num_hidden_layers=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_labels=2,
)

model = BertForSequenceClassification(config)

data_collator = DataCollatorWithPadding(tokenizer=g_tokenizer)

accuracy = evaluate.load("accuracy")

id2label = {0: "Lung_cancer", 1: "Other_disease"}
label2id = {"Other_disease": 0, "Lung_cancer": 1}

training_args = TrainingArguments(
    output_dir="output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()