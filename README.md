# SocialSift: Crisis-aware Sentiment Analysis

## Overview
Analyzes multilingual social media posts in real time during natural disasters. Identifies crisis-related sentiments to help aid organizations.

## Features
- Uses multilingual transformers (XLM-RoBERTa).
- Automatically downloads example social media dataset.
- Detects whether a post indicates help needed or is neutral/positive.

## Setup
```bash
pip install -r requirements.txt
Train
bash
Copy
Edit
python train.py
Predict
bash
Copy
Edit
python predict.py
```
Data
No need to provide any CSV. The example dataset will be downloaded automatically.

Finally, after generating all files, run:
pip install -r requirements.txt
python train.py
python predict.py
