import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from config import Config
from models.bi_encoder import BiEncoder
from utils.dataset import RetrieverDataset, collate_fn
from utils.loss import contrastive_cross_entropy_loss
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
config = Config()
device = config.DEVICE
articles_df = pd.read_csv(config.DATA_DIR + 'articles.csv')

# Chargement des données
train_dataset = RetrieverDataset(
    config.DATA_DIR + 'train.csv',
    articles_df,
    BiEncoder(config.MODEL_NAME).tokenizer,
    config.MAX_SEQ_LENGTH
)

# Création manuelle du DataLoader sans create_dataloader
train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=collate_fn
)

# Initialisation du modèle Bi-Encoder
model = BiEncoder(config.MODEL_NAME).to(device)

# Optimizer AdamW avec weight decay de 0.01
optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)

# Planificateur de taux d'apprentissage (warm-up et décroissance linéaire)
#num_training_steps = 14000
#num_warmup_steps = 300
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Pour le mixed precision training
scaler = GradScaler()

# Liste pour suivre les pertes pendant l'entraînement
losses = []

# Accumulation des gradients sur 3 étapes
gradient_accumulation_steps = 3

# Mode entraînement
model.train()

# Boucle d'entraînement
for epoch in range(10):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids_q = batch['input_ids_q'].to(device)
        attention_mask_q = batch['attention_mask_q'].to(device)
        input_ids_pos = batch['input_ids_pos'].to(device)
        attention_mask_pos = batch['attention_mask_pos'].to(device)
        input_ids_neg = batch['input_ids_neg'].to(device)  # Batch of negative examples
        attention_mask_neg = batch['attention_mask_neg'].to(device)

        # Utilisation de l'autocast pour le mixed precision
        with autocast():
            # Passage des inputs à travers le modèle
            q_emb = model(input_ids_q, attention_mask_q)
            pos_emb = model(input_ids_pos, attention_mask_pos)
            
            # On passe les articles négatifs
            neg_embs = [model(input_ids_neg[i], attention_mask_neg[i]) for i in range(input_ids_neg.size(0))]

            # Calcul de la perte avec les articles négatifs
            loss = contrastive_cross_entropy_loss(q_emb, pos_emb, neg_embs, temperature=0.05)

        # Backpropagation avec accumulation des gradients
        scaler.scale(loss).backward()

        # Accumuler les gradients toutes les `gradient_accumulation_steps`
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            #scheduler.step()  # Mise à jour du taux d'apprentissage

        # Suivi de la perte
        losses.append(loss.item())
        
        # Mise à jour de la barre de progression avec la perte actuelle
        progress_bar.set_postfix({"Loss": loss.item()})

    # Impression de la perte après chaque epoch
    print(f"Epoch {epoch + 1}/10, Loss: {loss.item()}")

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), 'bi_encoder.pth')

# Tracer la courbe de perte
plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('training_loss_curve.png')  # Sauvegarde de la courbe de perte sous forme d'image
plt.close()
