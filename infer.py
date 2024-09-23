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


def infer(model, question, articles_embs, max_lengthn top_k):
    input_ids_q, attention_mask_q = model.encode([question], max_length)
    input_ids_q = input_ids_q.to(Config.DEVICE)
    attention_mask_q = attention_mask_q.to(Config.DEVICE)

    with torch.no_grad():
        q_emb = model(input_ids_q, attention_mask_q)

    q_emb = q_emb.repeat(articles_embs.shape[0], 1)
    articles_embs = torch.tensor(articles_embs).to(Config.DEVICE)

    print(articles_embs.shape)
    print(q_emb.shape)
    similarities = F.cosine_similarity(q_emb, articles_embs)

    sorted_indices = torch.argsort(similarities, descending=True)+1 #+1 car les ids des articles correspondent à 1 contrairement à la ligne 0
    return sorted_indices[:top_k]

ef infer_all_questions(model, questions, articles_embs, max_length, top_k):
    """
    Fonction qui prend toutes les questions et retourne les top_k articles pour chaque question
    en utilisant une multiplication matricielle pour optimiser le calcul des similarités.
    
    Args:
    - model: le modèle bi-encodeur utilisé pour encoder les questions et les articles.
    - questions: liste de questions à évaluer.
    - articles_embs: embeddings pré-calculés des articles.
    - max_length: longueur maximale des tokens pour les questions.
    - top_k: le nombre d'articles à retourner pour chaque question.
    
    Retourne:
    - Une liste des IDs des articles les plus pertinents pour chaque question.
    """
    input_ids_q, attention_mask_q = model.encode(questions, max_length)
    input_ids_q = input_ids_q.to(Config.DEVICE)
    attention_mask_q = attention_mask_q.to(Config.DEVICE)
    with torch.no_grad():
        q_embs = model(input_ids_q, attention_mask_q)  # q_embs a la forme (num_questions, embed_dim)
    articles_embs = torch.tensor(articles_embs).to(Config.DEVICE)  # (num_articles, embed_dim)

    # Calculer la similarité cosinus entre chaque question et tous les articles revient à un produit matriciel entre les embeddings des questions et ceux des articles
    similarities = torch.mm(q_embs, articles_embs.T)  # (num_questions, num_articles)

    top_k_indices = torch.argsort(similarities, dim=1, descending=True)[:, :top_k] + 1  # +1 si tes IDs commencent à 1

    return top_k_indices  # (num_questions, top_k)


if __name__ == "__main__":
    model = load_model('bi_encoder.pth', Config.MODEL_NAME)
    articles_embs = np.load("data/articles_embeddings.npy")
    question = "Quel bail est considéré comme un bail 9 ans à Bruxelles ?"
    rel_ids = infer(model, question, articles_embs, Config.MAX_SEQ_LENGTH, top_k=10)

    articles_df = pd.read_csv(Config.DATA_DIR + 'articles.csv')
    print(f"Top {top_k} articles les plus pertinents pour la question : {question}")
    for idx in rel_ids:
        idx_int = idx.item() 
        print(
            f"Article {idx_int}: {articles_df.iloc[idx_int-1]['article']})")
