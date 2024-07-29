import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import logging

logging.set_verbosity_error()


# Load the fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Example usage
user_input = input("Enter your question: ")
tokenized_input = tokenizer(user_input, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
model.eval()

with torch.no_grad():
    output = model(**tokenized_input)
    predicted_class = torch.argmax(output.logits, dim=1).item()

# Convert numerical label back to original label
id_to_label = {0: "Open-ended", 1: "True-False", 2: "Fill-in-the-blank", 3: "Matching", 4: "Multiple-choice", 5: "Short-answer-question"}
predicted_question_type = id_to_label[predicted_class]

print(f"The predicted question type is: {predicted_question_type}")
