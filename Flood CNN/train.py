import torch
from torch.utils.data import DataLoader
from dataset import FloodDataset
from model import FloodCNN
from utils import show_prediction

# CONFIG
DATA_PATH = "data/ecti2021/train/train"
BATCH_SIZE = 4
EPOCHS = 5
LIMIT = 1000   # keep small for fast training

# Load dataset
dataset = FloodDataset(DATA_PATH, limit=LIMIT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = FloodCNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TRAINING LOOP
for epoch in range(EPOCHS):
    total_loss = 0
    
    for images, masks in loader:
        preds = model(images)
        loss = criterion(preds, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    

# TEST + VISUALIZATION

show_prediction(image, mask, pred)