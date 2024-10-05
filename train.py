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

config = Config()
device = config.DEVICE
articles_df = pd.read_csv(config.DATA_DIR + 'articles.csv')

train_dataset = RetrieverDataset(
    config.DATA_DIR + 'train.csv',
    articles_df,
    BiEncoder(config.MODEL_NAME).tokenizer,
    config.MAX_SEQ_LENGTH
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=collate_fn
)

model = BiEncoder(config.MODEL_NAME).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)

# Planificateur de taux d'apprentissage (warm-up et décroissance linéaire)
#num_training_steps = 14000
#num_warmup_steps = 300
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Pour le mixed precision training
scaler = GradScaler()

losses = []

gradient_accumulation_steps = 3

model.train()


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
            q_emb = model(input_ids_q, attention_mask_q)
            pos_emb = model(input_ids_pos, attention_mask_pos)
            
            neg_embs = [model(input_ids_neg[i], attention_mask_neg[i]) for i in range(input_ids_neg.size(0))]

            loss = contrastive_cross_entropy_loss(q_emb, pos_emb, neg_embs, temperature=0.05)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()  # Mise à jour du taux d'apprentissage

        losses.append(loss.item())
        
        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch + 1}/10, Loss: {loss.item()}")

torch.save(model.state_dict(), 'bi_encoder.pth')

plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('training_loss_curve.png') 
plt.close()
