# utils/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import random

class RetrieverDataset(Dataset):
    def __init__(self, train_csv, hard_neg_json, articles_df, tokenizer, max_length):
        self.data = pd.read_csv(train_csv)
        with open(hard_neg_json, 'r') as f:
            self.hard_neg = json.load(f)  # Chargement des négatifs durs depuis le fichier JSON
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

        # Extraction des articles négatifs durs (limite à 10 articles négatifs)
        negative_article_ids = self.hard_neg.get(question_id, [])[:10]
        negative_encs = []
        for neg_id in negative_article_ids:
            if neg_id in self.articles_df['id'].values:
                negative_article = self.articles_df[self.articles_df['id'] == neg_id].iloc[0]
                negative_text = f"{negative_article['section']} {negative_article['subsection']} {negative_article['article']}"
                # Tokenisation de chaque article négatif
                negative_enc = self.tokenizer.encode_plus(
                    negative_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
                )
                negative_encs.append(negative_enc)

        # Conversion des tokenizations en tensors
        input_ids_neg = torch.stack([enc['input_ids'].squeeze(0) for enc in negative_encs])
        attention_mask_neg = torch.stack([enc['attention_mask'].squeeze(0) for enc in negative_encs])

        return {
            'input_ids_q': question_enc['input_ids'].squeeze(0),
            'attention_mask_q': question_enc['attention_mask'].squeeze(0),
            'input_ids_pos': positive_enc['input_ids'].squeeze(0),
            'attention_mask_pos': positive_enc['attention_mask'].squeeze(0),
            'input_ids_neg': input_ids_neg,  # 10 articles négatifs
            'attention_mask_neg': attention_mask_neg
        }
