# Tokenize the data


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_tokenized, y, test_size=0.2)

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

# Train the model
# Convert the data to torch tensors
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

# Create a TensorDataset and DataLoader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/tmp/clinvar',            # directory for storing logs
    logging_steps=10,
)

# Create Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
# Save the model
model.save_pretrained('./output')