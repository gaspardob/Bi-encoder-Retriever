import pandas as pd
import torch
import json
from transformers import CamembertTokenizer, CamembertModel

# Charger les données
train_df = pd.read_csv('train.csv')
articles_df = pd.read_csv('articles.csv')

# Charger les négatifs durs
with open('hard_neg_bm25_train.json', 'r') as f:
    hard_negatives = json.load(f)


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

    def encode(self, texts, max_length=128):
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']


import torch.nn as nn
import torch.optim as optim

# Initialiser le modèle
model = BiEncoder('camembert-base')
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


# Fonction pour récupérer un article à partir de son ID
def get_article_text(article_id, articles_df):
    article_row = articles_df[articles_df['id'] == article_id]
    # Concaténer les colonnes pertinentes pour former l'article
    article_text = ' '.join(article_row[['section', 'subsection', 'article']].fillna('').values[0])
    return article_text


# Boucle d'entraînement simplifiée
for idx, row in train_df.iterrows():
    question_id = row['id']
    positive_article_ids = eval(row['article_ids'])  # Liste d'articles pertinents (au moins 1)

    if not positive_article_ids:
        continue  # Passer si pas d'article positif

    # Gestion de l'article positif
    positive_article_ids = eval(row['article_ids'])  # Peut être une liste ou un entier

    # Vérifier si positive_article_ids est un entier ou une liste
    if isinstance(positive_article_ids, int):
        positive_article_id = positive_article_ids  # Si c'est un entier, on l'utilise directement
    else:
        positive_article_id = positive_article_ids[0]  # Si c'est une liste, on prend le premier élément


    negative_article_ids = hard_negatives.get(str(question_id), [])

    if not negative_article_ids:
        continue  # Passer si pas d'articles négatifs

    negative_article_id = negative_article_ids[0]  # On prend le premier article négatif

    # Récupérer les textes
    question_text = row['question']
    positive_article_text = get_article_text(positive_article_id, articles_df)
    negative_article_text = get_article_text(negative_article_id, articles_df)

    # Encoder les textes
    question_input_ids, question_attention_mask = model.encode([question_text])
    pos_input_ids, pos_attention_mask = model.encode([positive_article_text])
    neg_input_ids, neg_attention_mask = model.encode([negative_article_text])

    # Passer par le modèle
    question_embedding = model(question_input_ids, question_attention_mask)
    pos_embedding = model(pos_input_ids, pos_attention_mask)
    neg_embedding = model(neg_input_ids, neg_attention_mask)

    # Similitudes cosinus
    cos_sim = nn.CosineSimilarity(dim=1)
    positive_similarity = cos_sim(question_embedding,
                                  pos_embedding)  # Similitude entre la question et l'article positif
    negative_similarity = cos_sim(question_embedding,
                                  neg_embedding)  # Similitude entre la question et l'article négatif

    # Logits: empiler les similarités et reformater en [batch_size, num_classes]
    logits = torch.stack([positive_similarity, negative_similarity], dim=0)  # [2]
    logits = logits.view(1, -1)  # Reformater en [1, 2], batch_size=1, num_classes=2

    # Labels: Tensor de taille [batch_size] indiquant la classe correcte
    labels = torch.tensor([0])  # [1], car la première classe (positive) est correcte

    # Assurons-nous que logits et labels ont les bonnes formes
    print(f'Logits shape: {logits.shape}')  # Devrait être [1, 2]
    print(f'Labels shape: {labels.shape}')  # Devrait être [1]

    # Calcul de la perte
    loss = criterion(logits, labels)

    # Backpropagation et optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Iteration {idx}, Loss: {loss.item()}')






