# config.py

import torch

class Config:
    DATA_DIR = './data/'
    BATCH_SIZE = 16
    EPOCHS = 1
    LR = 1e-5
    MAX_SEQ_LENGTH = 128
    MODEL_NAME = 'camembert-base'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
