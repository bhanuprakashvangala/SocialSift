import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
from models.sentiment_model import MultilingualSentimentClassifier
from utils.preprocess import load_and_clean
from utils.download import download_file
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Download example CSV automatically
csv_url = "https://raw.githubusercontent.com/bhanuprakash-vangala/public-datasets/main/example_posts.csv"
csv_path = "data/example_posts.csv"
download_file(csv_url, csv_path)


class SocialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label


df = load_and_clean(csv_path)
texts = df['text'].tolist()
labels = df['label'].tolist()

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
dataset = SocialDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultilingualSentimentClassifier(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    loop = tqdm(loader)
    for input_ids, attention_mask, labels in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/3]")
        loop.set_postfix(loss=loss.item())


torch.save(model.state_dict(), 'models/socialsift_model.pth')
