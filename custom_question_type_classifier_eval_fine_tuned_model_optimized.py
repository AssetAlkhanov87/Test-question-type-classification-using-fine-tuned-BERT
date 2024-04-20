import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Load the fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Questions dataset
questions = []
with open("questions_dataset/questions_dataset.txt", "r") as f:
    for line in f:
        line_lst = line.split("###")
        line_lst[1] = line_lst[1][1:len(line_lst[1])-1]
        tuple_line = tuple(line_lst)
        questions.append(tuple_line)

# Convert labels to numerical values
label_to_id = {"Open-ended": 0, "Yes/No": 1, "Fill-in-the-blank": 2, "Matching": 3, "Multiple-choice": 4}
questions = [(text, label_to_id[label]) for text, label in questions]

# Split the dataset
_, val_data = train_test_split(questions, test_size=0.2, random_state=42)

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
            return_attention_mask=True  # Ensure that attention mask is returned
        )
        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "attention_mask": tokenized_text["attention_mask"].squeeze(),  # Include attention mask in the batch
            "label": torch.tensor(label)
        }


# Tokenizer and model
val_dataset = QuestionDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Evaluation on the validation set
model.eval()

with torch.no_grad():
    all_preds = []
    all_labels = []

    for batch in val_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]  # Add this line to retrieve attention mask
        labels = batch["label"]
        outputs = model(input_ids, attention_mask=attention_mask)  # Pass attention mask to the model
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())


# Calculate accuracy
val_accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {val_accuracy}")
