# train.py
import torch
from torch.utils.data import DataLoader
from config import Config
from models.bi_encoder import BiEncoder
from utils.dataset import RetrieverDataset
from utils.loss import contrastive_cross_entropy_loss
import pandas as pd

print("ok")
