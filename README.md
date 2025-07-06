# SocialSift: Crisis-aware Sentiment Analysis

## Overview
SocialSift analyzes multilingual social media posts in real time during natural disasters. The model identifies whether a post is neutral/positive or if it indicates that help is needed.

## Features
- Utilizes XLM-RoBERTa for multilingual support.
- Automatically downloads an example dataset on first run.
- Simple training and inference scripts.

## Usage
Install dependencies and train the model:

```bash
pip install -r requirements.txt
python train.py
```

After training, run predictions:

```bash
python predict.py
```

The example dataset is downloaded automatically; no additional data setup is required.
