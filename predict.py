import torch
from models.sentiment_model import MultilingualSentimentClassifier
from transformers import XLMRobertaTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultilingualSentimentClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load('models/socialsift_model.pth'))
model.eval()

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def predict_sentiment(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=1).item()
        return "Negative (Help Needed)" if pred == 1 else "Neutral / Positive"

# Example
example_text = "We are surrounded by flood waters, please help!"
result = predict_sentiment(example_text)
print(f"Predicted sentiment: {result}")
