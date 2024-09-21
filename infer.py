# test.py
import numpy as np
import torch
import pandas as pd
from models.bi_encoder import BiEncoder
from config import Config
import torch.nn.functional as F


def load_model(model_path, model_name):
    model = BiEncoder(model_name)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model


def infer(model, question, articles_embs, max_length):
    # Encode la question
    input_ids_q, attention_mask_q = model.encode([question], max_length)
    input_ids_q = input_ids_q.to(Config.DEVICE)
    attention_mask_q = attention_mask_q.to(Config.DEVICE)

    # Calcul des embeddings de la question
    with torch.no_grad():
        q_emb = model(input_ids_q, attention_mask_q)

    q_emb = q_emb.repeat(articles_embs.shape[0], 1)
    articles_embs = torch.tensor(articles_embs).to(Config.DEVICE)

    # Calculer la similarité cosinus entre la question et chaque article
    print(articles_embs.shape)
    print(q_emb.shape)
    similarities = F.cosine_similarity(q_emb, articles_embs)
    print(similarities.shape)
    return similarities


if __name__ == "__main__":
    # Charger le modèle
    model = load_model('bi_encoder.pth', Config.MODEL_NAME)

    # Charger les embeddings des articles
    articles_embs = np.load("data/articles_embeddings.npy")

    # Exemple de question pour l'inférence
    question = "Quel bail est considéré comme un bail 9 ans à Bruxelles ?"

    # Effectuer l'inférence
    similarities = infer(model, question, articles_embs, Config.MAX_SEQ_LENGTH)

    # Trier les articles par pertinence
    sorted_indices = torch.argsort(similarities, descending=True)
    top_k = 10  # Nombre d'articles les plus pertinents à retourner

    # Charger les données des articles
    articles_df = pd.read_csv(Config.DATA_DIR + 'articles.csv')
    print(articles_df.iloc[9441])
    # Afficher les indices et les articles les plus pertinents
    print(f"Top {top_k} articles les plus pertinents pour la question : {question}")
    for idx in sorted_indices[:top_k]:
        idx_int = idx.item()  # Convertir le tensor en entier
        print(
            f"Article {idx_int}: {articles_df.iloc[idx_int]['article']})")
