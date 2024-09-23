import torch
import torch.nn.functional as F

def contrastive_loss_with_cross_entropy(qi_embedding, ai_positive_embedding, ai_negative_embeddings, temperature=0.07):
    """
    Calcule la loss contrastive pour un seul exemple avec F.cross_entropy.
    
    Arguments:
    qi_embedding -- embedding de la question qi
    ai_positive_embedding -- embedding de l'article positif a_i+
    ai_negative_embeddings -- liste des embeddings des articles négatifs A_i-
    temperature -- paramètre de température pour l'échelle de la similarité (tau)

    Retourne:
    Loss contrastive pour l'exemple donné.
    """
    
    positive_similarity = F.cosine_similarity(qi_embedding, ai_positive_embedding, dim=-1) / temperature
    negative_similarities = F.cosine_similarity(qi_embedding.unsqueeze(0), ai_negative_embeddings, dim=-1) / temperature
    
    similarities = torch.cat([positive_similarity.unsqueeze(0), negative_similarities], dim=0)  # [N+1]
    
    # Création des labels (le positif est à l'indice 0)
    labels = torch.zeros(1, dtype=torch.long)  # Le premier élément est l'article pertinent (label 0)
    loss = F.cross_entropy(similarities.unsqueeze(0), labels)
    
    return loss

