# utils/loss.py
import torch
import torch.nn.functional as F

def contrastive_cross_entropy_loss(q_emb, pos_emb, neg_embs, temperature=0.07):
    """
    Calcule la loss en prenant en compte la similarité cosinus entre la question et l'article pertinent,
    ainsi que la similarité avec tous les articles négatifs durs (hard negatives).
    """
    print(q_emb.shape, pos_emb.shape, len(neg_embs))
    pos_sim = F.cosine_similarity(q_emb, pos_emb)
    #neg_embs=torch.tensor(neg_embs).transpose(0,1)

    # Empiler les tenseurs transposés le long d'une nouvelle dimension
    neg_embs = torch.stack(neg_embs)
    print(neg_embs.shape)
    # Pour chaque article négatif, on répète l'embedding de la question autant de fois qu'il y a de négatifs
    neg_sims = torch.stack([F.cosine_similarity(q_emb, neg_embs[:,i,:]) for i in range(neg_embs.size(1))], dim=1)

    # Concaténer la similarité positive avec les similarités négatives
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)

    # Création des labels : 0 pour l'article pertinent
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    #print(logits.size())
    #print(labels.size())
    # Calcul de la cross entropy loss
    loss = F.cross_entropy(logits / temperature, labels)
    return loss
