import numpy as np
import torch
import pandas as pd
from models.bi_encoder import BiEncoder
from config import Config
import torch.nn.functional as F
from infer import load_model, infer_all_questions

def evaluate(model, test_dataset, articles_embs, max_length=128, top_k=5):
    """
    Évalue le modèle bi-encodeur sur le dataset de test en calculant la précision moyenne et le rappel moyen.
    
    Args:
    - model: le modèle bi-encodeur chargé.
    - test_dataset: DataFrame contenant les questions et les articles pertinents (ground truth).
    - articles_embs: embeddings pré-calculés des articles.
    - max_length: longueur maximale pour le tokenization des questions.
    - top_k: nombre d'articles à retourner pour chaque question.
    
    Retourne:
    - La précision moyenne et le rappel moyen sur toutes les questions.
    """
    
    questions = test_dataset['question'].tolist()
    ground_truth_articles = test_dataset['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    top_k_retrieved_articles = infer_all_questions(model, questions, articles_embs, max_length, top_k)
    
    total_recall = 0.0
    total_precision = 0.0
    for i, retrieved_articles in enumerate(top_k_retrieved_articles):
        relevant_articles = set(ground_truth_articles[i])
        retrieved_articles_set = set(retrieved_articles.cpu().numpy())  # Convertir en set pour comparaison
        true_positives = len(relevant_articles.intersection(retrieved_articles_set))
        
        precision = true_positives / top_k
        total_precision += precision
        
        recall = true_positives / len(relevant_articles) if len(relevant_articles) > 0 else 0.0
        total_recall += recall
    
    avg_precision = total_precision / len(questions)
    avg_recall = total_recall / len(questions)
    
    return avg_precision, avg_recall

if __name__ == "__main__":
    model = load_model('bi_encoder.pth', Config.MODEL_NAME)
    test_dataset = pd.read_csv("data/test.csv")
    articles_embs = np.load("data/articles_embeddings.npy") 

    avg_precision, avg_recall = evaluate(model, test_dataset, articles_embs, max_length=128, top_k=5)
    
    print(f"Précision moyenne sur le dataset de test : {avg_precision:.4f}")
    print(f"Rappel moyen sur le dataset de test : {avg_recall:.4f}")
