# train.py
import torch
from torch.utils.data import DataLoader
from config import Config
from models.bi_encoder import BiEncoder
from utils.dataset import RetrieverDataset
from utils.loss import contrastive_cross_entropy_loss
import pandas as pd

# Chargement des données
config = Config()
device = config.DEVICE
articles_df = pd.read_csv(config.DATA_DIR + 'articles.csv')

train_dataset = RetrieverDataset(
    config.DATA_DIR + 'train.csv',
    config.DATA_DIR + 'hard_neg_bm25_train.json',
    articles_df,
    BiEncoder(config.MODEL_NAME).tokenizer,
    config.MAX_SEQ_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Modèle
model = BiEncoder(config.MODEL_NAME).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

# Entraînement
model.train()
for epoch in range(config.EPOCHS):
    for batch in train_loader:
        input_ids_q = batch['input_ids_q'].to(device)
        attention_mask_q = batch['attention_mask_q'].to(device)
        input_ids_pos = batch['input_ids_pos'].to(device)
        attention_mask_pos = batch['attention_mask_pos'].to(device)
        input_ids_neg = batch['input_ids_neg'].to(device)
        attention_mask_neg = batch['attention_mask_neg'].to(device)
        print(input_ids_q.shape, input_ids_pos.shape, input_ids_neg.shape)
        # Encodage des questions, articles pertinents et négatifs
        q_emb = model(input_ids_q, attention_mask_q)
        pos_emb = model(input_ids_pos, attention_mask_pos)
        #input_ids_neg = input_ids_neg.transpose(0, 1)
        print(input_ids_neg.shape)

        neg_embs = [model(input_ids_neg[i], attention_mask_neg[i]) for i in range(input_ids_neg.size(0))]
        #neg_embs = [model(input_ids_neg[i].unsqueeze(0), attention_mask_neg[i].unsqueeze(0)) for i in range(input_ids_neg.size(0))]
        #print(len(neg_embs))
        print(len(neg_embs))
        # Calcul de la loss avec les hard negatives
        loss = contrastive_cross_entropy_loss(q_emb, pos_emb, neg_embs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {loss.item()}")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'bi_encoder.pth')
