import os
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from genomic_tokenizer import GenomicTokenizer
from src.dataloader.data_wrapper import (
    ClinVarDataWrapper
)
import pandas as pd

def process_data(data):
    new_data = []
    for x, y in tqdm(data, desc="Appending sequences"):
        new_data.append([x[0], x[1], x[2], y])
    return new_data

model_max_length = 512
tokenizer = GenomicTokenizer(model_max_length)

#  instantiate yourself
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

model = BertForSequenceClassification(config)


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

# Load the BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Example sentences and labels
# sentences = ["This is a positive sentence.", "This is a negative sentence."]
# labels = [1, 0]

# Read the dataframe and convert it to a list of sentences and labels
sentences = df['alt'].tolist()
labels = df['label'].tolist()

# convert 'Lung_cancer' to 1 and Other_disease to 0
labels = [1 if label in DISEASE_SUBSET else 0 for label in labels]

# Tokenize the sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")


# Create a DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    print(predictions)