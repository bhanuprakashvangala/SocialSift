import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class MultilingualSentimentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MultilingualSentimentClassifier, self).__init__()
        self.transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
