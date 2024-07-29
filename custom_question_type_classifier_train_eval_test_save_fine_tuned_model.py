import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import logging
import random
logging.set_verbosity_error()

# Questions dataset
questions = []
test_data = []

# Open the files with explicit encoding
with open("questions_dataset/questions_dataset_train.txt", "r", encoding="utf-8") as f_train, \
     open("questions_dataset/questions_dataset_test.txt", "r", encoding="utf-8") as f_test:

    for line in f_train:
        line_lst = line.split("###")
        line_lst[1] = line_lst[1][1:len(line_lst[1])-1]
        tuple_line = tuple(line_lst)
        questions.append(tuple_line)

    for line_test in f_test:
        line_lst_test = line_test.split("###")
        line_lst_test[1] = line_lst_test[1][1:len(line_lst_test[1])-1]
        tuple_line_test = tuple(line_lst_test)
        test_data.append(tuple_line_test)

# Convert labels to numerical values
label_to_id = {"Open-ended": 0, "True-False": 1, "Fill-in-the-blank": 2, "Matching": 3, "Multiple-choice": 4, "Short-answer-question": 5}

# Convert labels to numerical values, using -1 for missing labels
questions_train_data = [(text, label_to_id.get(label, -1)) for text, label in questions]
test_data = [(text, label_to_id.get(label, -1)) for text, label in test_data]

# Filter out data points with invalid labels (-1)
questions_train_data = [item for item in questions_train_data if item[1] != -1]
test_data = [item for item in test_data if item[1] != -1]

# Split the test dataset into validation and test datasets
if len(test_data) > 1:
    test_dataset, val_dataset = train_test_split(test_data, test_size=0.5, random_state=42)
else:
    test_dataset, val_dataset = test_data, test_data

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
train_dataset = QuestionDataset(questions_train_data, tokenizer)
val_dataset = QuestionDataset(val_dataset, tokenizer)
test_dataset = QuestionDataset(test_dataset, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) if len(val_dataset) > 0 else None
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["label"]
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation step
    if val_loader:
        model.eval()
        val_labels = []
        val_preds = []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                labels = batch["label"]
                outputs = model(input_ids, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_labels.extend(labels.tolist())
                val_preds.extend(preds.tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print("No validation data available.")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_distilbert")

# Evaluation on test data
model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["label"]
        outputs = model(input_ids)
        preds = torch.argmax(outputs.logits, dim=1)
        test_labels.extend(labels.tolist())
        test_preds.extend(preds.tolist())

# Calculate test metrics
test_accuracy = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

print("Test Accuracy: {:.4f}".format(test_accuracy))
print("Test Precision: {:.4f}".format(test_precision))
print("Test Recall: {:.4f}".format(test_recall))
print("Test F1 Score: {:.4f}".format(test_f1))


