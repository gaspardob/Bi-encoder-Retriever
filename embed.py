import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.bi_encoder import BiEncoder
from config import Config
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


# Fonction pour charger le modèle
def load_model(model_path, model_name):
    model = BiEncoder(model_name)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model


model = load_model("bi_encoder.pth", Config.MODEL_NAME)


# Fonction pour générer les embeddings avec précision mixte
def embed_articles(model, articles, max_length):
    """
    Encode et génère des embeddings pour une liste d'articles avec précision mixte.
    Sauvegarde les embeddings dans un tableau numpy.
    """
    all_embeddings = []

    batch_size = 8  # Ajuster selon ta configuration GPU

    # Boucle sur les articles par batch
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_articles = articles[i:i + batch_size]

        # Tokenisation des articles
        input_ids, attention_mask = model.encode(batch_articles, max_length=max_length)
        input_ids = input_ids.to(Config.DEVICE)
        attention_mask = attention_mask.to(Config.DEVICE)

        # Utilisation de la précision mixte avec autocast
        with torch.no_grad():
            with autocast():
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

        # Convertir les embeddings en FP32 avant de les ajouter à la liste
        all_embeddings.append(embeddings.float().cpu().numpy())

    # Concatène tous les embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings


# Chargement des articles
articles_df = pd.read_csv(Config.DATA_DIR + 'articles.csv')
articles = articles_df['article'].tolist()

# Génération des embeddings avec précision mixte
#embeddings = embed_articles(model, articles, max_length=512)

# Sauvegarde des embeddings dans un fichier numpy
#np.save(Config.DATA_DIR + 'articles_embeddings.npy', embeddings)


