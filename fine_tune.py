import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Load the dataset
df = pd.read_excel('fine_tune_data.xlsx')

# Remove trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names and the first few rows to debug
print("Column names:", df.columns)
print(df.head())

# Drop rows with NaN values in 'message' or 'label'
df.dropna(subset=['message', 'label'], inplace=True)

# Check the distribution of labels
print("Label distribution before mapping:")
print(df['label'].value_counts())

# Convert labels from -1, 0, 1 to 0, 1, 2
label_mapping = {-1: 0, 0: 1, 1: 2}
df['label'] = df['label'].map(label_mapping)

# Check the distribution of labels after mapping
print("Label distribution after mapping:")
print(df['label'].value_counts())

# Split the data into training and evaluation sets
texts = df['message'].tolist()
labels = df['label'].tolist()

# Ensure that texts are strings
texts = [str(text) for text in texts]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize the tokenizer
model_name = "cahya/bert-base-indonesian-522M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create custom Datasets
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long tensors
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: 0, 1, 2

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Reduce the number of epochs for testing
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    logging_steps=10,  # Log training metrics every 10 steps
)

# Define compute metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
fined_model_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_model'
tokenizer_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_tokenizer'

trainer.save_model(fined_model_path)
tokenizer.save_model(tokenizer_path)

print(f"Model saved to {fined_model_path}")
print(f"Tokenizer saved to {tokenizer_path}")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Compute confusion matrix
preds = trainer.predict(val_dataset).predictions.argmax(-1)
cm = confusion_matrix(val_labels, preds)
print("Confusion Matrix:")
print(cm)