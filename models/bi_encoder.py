# models/bi_encoder.py
import torch
from transformers import CamembertTokenizer, CamembertModel

class BiEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super(BiEncoder, self).__init__()
        self.bert = CamembertModel.from_pretrained(model_name)
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]  # Avant-dernière couche
        # Pooling - moyenne des embeddings de tokens
        pooled_output = torch.mean(hidden_states, dim=1)
        return pooled_output

    def encode(self, texts, max_length):
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']

