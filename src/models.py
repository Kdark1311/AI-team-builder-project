import torch
import torch.nn as nn
from transformers import BertModel

class MBTIModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.4,
                 use_hidden_layer=True, pooling="cls+mean"):
        super(MBTIModel, self).__init__()

        # Load full BERT (không freeze)
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        input_dim = hidden_size * 2 if pooling == "cls+mean" else hidden_size

        if use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 4)
            )
        else:
            self.classifier = nn.Linear(input_dim, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            pooled = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)
        elif self.pooling == "max":
            masked = outputs.last_hidden_state.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )
            pooled = masked.max(1).values
        elif self.pooling == "cls+mean":
            cls_emb = outputs.last_hidden_state[:, 0, :]
            mean_emb = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            mean_emb = mean_emb / attention_mask.sum(1, keepdim=True)
            pooled = torch.cat([cls_emb, mean_emb], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        x = self.dropout(pooled)
        logits = self.classifier(x)

        return logits
