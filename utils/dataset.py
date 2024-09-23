# utils/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

class RetrieverDataset(Dataset):
    def __init__(self, train_csv, articles_df, tokenizer, max_length):
        self.data = pd.read_csv(train_csv)
        self.articles_df = articles_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extraction des informations de train.csv
        row = self.data.iloc[idx]
        question_id = str(row['id'])  # ID de la question
        category = row['category']
        subcategory = row['subcategory']
        question = row['question']

        # Concaténation des champs pour la question
        full_question = f"{category} {subcategory} {question}"

        # Tokenisation de la question
        question_enc = self.tokenizer.encode_plus(
            full_question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        # Extraction des articles pertinents (choix aléatoire d'un article pertinent)
        positive_article_ids = eval(row['article_ids'])
        if isinstance(positive_article_ids, list):
            positive_article_id = random.choice(positive_article_ids)
        else:
            positive_article_id = positive_article_ids

        # Vérification si l'article existe dans articles_df
        if positive_article_id in self.articles_df['id'].values:
            positive_article = self.articles_df[self.articles_df['id'] == positive_article_id].iloc[0]
            positive_text = f"{positive_article['section']} {positive_article['subsection']} {positive_article['article']}"
        else:
            positive_text = ""  # Gérer le cas où l'article n'est pas trouvé

        # Tokenisation de l'article pertinent choisi
        positive_enc = self.tokenizer.encode_plus(
            positive_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids_q': question_enc['input_ids'].squeeze(0),
            'attention_mask_q': question_enc['attention_mask'].squeeze(0),
            'input_ids_pos': positive_enc['input_ids'].squeeze(0),
            'attention_mask_pos': positive_enc['attention_mask'].squeeze(0),
            'article_id_pos': positive_article_id  # Utilisé pour construire les négatifs
        }

def collate_fn(batch):
    """
    Custom collate function to create in-batch negatives.
    """
    input_ids_q = torch.stack([item['input_ids_q'] for item in batch])
    attention_mask_q = torch.stack([item['attention_mask_q'] for item in batch])
    input_ids_pos = torch.stack([item['input_ids_pos'] for item in batch])
    attention_mask_pos = torch.stack([item['attention_mask_pos'] for item in batch])

    # Construire les articles négatifs (in-batch negatives)
    article_ids_pos = [item['article_id_pos'] for item in batch]
    input_ids_neg = []
    attention_mask_neg = []

    for i, item in enumerate(batch):
        # Exclure l'article positif de l'exemple actuel et utiliser les articles des autres questions comme négatifs
        neg_items = [input_ids_pos[j] for j in range(len(batch)) if j != i]
        neg_attention_masks = [attention_mask_pos[j] for j in range(len(batch)) if j != i]
        
        input_ids_neg.append(torch.stack(neg_items))  # N-1 négatifs
        attention_mask_neg.append(torch.stack(neg_attention_masks))

    input_ids_neg = torch.stack(input_ids_neg)
    attention_mask_neg = torch.stack(attention_mask_neg)

    return {
        'input_ids_q': input_ids_q,
        'attention_mask_q': attention_mask_q,
        'input_ids_pos': input_ids_pos,
        'attention_mask_pos': attention_mask_pos,
        'input_ids_neg': input_ids_neg,  # N-1 articles négatifs
        'attention_mask_neg': attention_mask_neg
    }

