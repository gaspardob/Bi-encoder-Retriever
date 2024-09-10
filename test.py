# test.py
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

def infer(model, question, articles, max_length):
    # Encode la question
    input_ids_q, attention_mask_q = model.encode([question], max_length)
    input_ids_q = input_ids_q.to(Config.DEVICE)
    attention_mask_q = attention_mask_q.to(Config.DEVICE)

    # Encode les articles
    input_ids_a, attention_mask_a = model.encode(articles, max_length)
    input_ids_a = input_ids_a.to(Config.DEVICE)
    attention_mask_a = attention_mask_a.to(Config.DEVICE)

    # Calcul des embeddings
    with torch.no_grad():
        q_emb = model(input_ids_q, attention_mask_q)
        a_embs = model(input_ids_a, attention_mask_a)

    # Similarité cosinus entre la question et chaque article
    similarities = F.cosine_similarity(q_emb, a_embs)
    return similarities

if __name__ == "__main__":
    # Charge le modèle
    model = load_model('bi_encoder.pth', Config.MODEL_NAME)

    # Charger les données d'inférence
    articles_df = pd.read_csv(Config.DATA_DIR + 'articles.csv')
    articles = articles_df['content'].tolist()

    # Exemple de question pour l'inférence
    question = "Quelle est l'impact du changement climatique sur les ressources en eau ?"

    # Effectuer l'inférence
    similarities = infer(model, question, articles, Config.MAX_SEQ_LENGTH)

    # Trier les articles par pertinence
    sorted_indices = torch.argsort(similarities, descending=True)
    top_k = 5  # Nombre d'articles les plus pertinents à retourner

    print(f"Top {top_k} articles les plus pertinents pour la question : {question}")
    for idx in sorted_indices[:top_k]:
        print(f"Article {idx.item()}: {articles_df.iloc[idx]['content'][:200]}... (similarité : {similarities[idx].item():.4f})")
