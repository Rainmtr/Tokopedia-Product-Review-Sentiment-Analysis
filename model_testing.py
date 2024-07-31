from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Get Saved Model from Path
fined_model_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_model'
tokenizer_path = 'C:/Code/SentimentAnalysisProject/product_sentiment_analysis/fined_tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(fined_model_path)

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