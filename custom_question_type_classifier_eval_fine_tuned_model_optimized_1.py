import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import logging
from sklearn.model_selection import train_test_split  # Add import statement
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the warning

# Questions dataset
questions = []
f = open("questions_dataset/questions_dataset.txt", "r")

for line in f:
    line_lst = line.split("###")
    line_lst[1] = line_lst[1][1:len(line_lst[1])-1]
    tuple_line = tuple(line_lst)
    questions.append(tuple_line)

# Convert labels to numerical values
label_to_id = {"Open-ended": 0, "Yes/No": 1, "Fill-in-the-blank": 2, "Matching": 3, "Multiple-choice": 4}
questions = [(text, label_to_id[label]) for text, label in questions]

# Tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_to_id))

# Dataset class
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
            return_attention_mask=True
        )
        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }

# Split the dataset
train_data, val_data = train_test_split(questions, test_size=0.2, random_state=42)

# Create DataLoader
val_dataset = QuestionDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# Calculate precision, recall, and F1 scores
precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division='warn')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Support: {support}")

