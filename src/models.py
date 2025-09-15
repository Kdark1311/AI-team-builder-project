# src/models.py
import torch
import torch.nn as nn
from transformers import BertModel

class MBTIModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.3, use_hidden_layer=True):
        super(MBTIModel, self).__init__()
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # thường là 768

        self.dropout = nn.Dropout(dropout)

        if use_hidden_layer:
            # Classifier "sâu" hơn để tăng khả năng học
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 4)
            )
        else:
            # Classifier đơn giản
            self.classifier = nn.Linear(hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Lấy CLS embedding trực tiếp (ổn định hơn pooler_output)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        x = self.dropout(cls_embedding)
        logits = self.classifier(x)

        return logits
