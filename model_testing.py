import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# # Load Dataset
# df = pd.read_excel('sigma.xlsx')

# # Remove trailing spaces from column names
# df.columns = df.columns.str.strip()

# # Print column names and the first few rows to debug
# print("Column names:", df.columns)
# print(df.head())

# # Drop rows with NaN values in 'message' or 'label'
# df.dropna(subset=['message', 'label'], inplace=True)

# # Check the distribution of labels
# print("Label distribution before mapping:")
# print(df['label'].value_counts())

# # Convert labels from -1, 0, 1 to 0, 1, 2
# label_mapping = {-1: 0, 0: 1, 1: 2}
# df['label'] = df['label'].map(label_mapping)

# # Check the distribution of labels after mapping
# print("Label distribution after mapping:")
# print(df['label'].value_counts())

# # Split the data into training and evaluation sets
# texts = df['message'].tolist()
# labels = df['label'].tolist()

# # Ensure that texts are strings
# texts = [str(text) for text in texts]

# train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Initialize the tokenizer
# model_name = "cahya/bert-base-indonesian-522M"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Tokenize the texts
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# # Create custom Datasets
# class SentimentDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long tensors
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = SentimentDataset(train_encodings, train_labels)
# val_dataset = SentimentDataset(val_encodings, val_labels)

# Get Saved Model from Path
fined_model_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_model'
tokenizer_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(fined_model_path)

# # Compute confusion matrix
# preds = model.predict(val_dataset).predictions.argmax(-1)
# cm = confusion_matrix(val_labels, preds)
# print("Confusion Matrix:")
# print(cm)

# Example prediction pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Reverse label mapping
reverse_label_mapping = {0: -1, 1: 0, 2: 1}

# Continuous input loop for sentiment analysis
while True:
    # Get input from the user
    input_text = input("Enter a review (or type 'exit' to stop): ")
    if input_text.lower() == 'exit':
        break

    # Perform sentiment analysis on the input text
    result = sentiment_analysis(input_text)[0]
    original_label = reverse_label_mapping[int(result['label'].split('_')[1])]
    
    # Print the sentiment analysis result
    print(f"Text: {input_text} | Sentiment: {original_label} | Score: {result['score']:.2f}")