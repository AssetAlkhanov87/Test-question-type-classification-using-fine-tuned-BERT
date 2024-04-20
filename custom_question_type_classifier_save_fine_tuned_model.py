import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import logging
logging.set_verbosity_error()

# Questions dataset
questions = []
f = open("questions_dataset/questions_dataset.txt", "r")

count = 0
for line in f:
    line_lst = line.split("###")
    line_lst[1] = line_lst[1][1:len(line_lst[1])-1]
    tuple_line = tuple(line_lst)
    
    questions.append(tuple_line)
    
# Convert labels to numerical values
label_to_id = {"Open-ended": 0, "Yes/No": 1, "Fill-in-the-blank": 2, "Matching": 3, "Multiple-choice": 4}
questions = [(text, label_to_id[label]) for text, label in questions]

# Split the dataset
train_data, val_data = train_test_split(questions, test_size=0.2, random_state=42)

# Define a custom dataset class
class QuestionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": tokenized_text["input_ids"].squeeze(), "label": torch.tensor(label)}

# Tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_to_id))

# Train the model
train_dataset = QuestionDataset(train_data, tokenizer)
val_dataset = QuestionDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["label"]
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_distilbert")


