import os
import torch
import torch.nn as nn
import tqdm
from timeit import default_timer as timer
from Modules._01_config import *
from Modules._02_patchEmbedding import PatchEmbedding
from Modules._03_ViT import ViT
from Modules._04_dataset import train_dataloader, test_dataloader, class_names


model_path = f"ViT_{EPOCHS}_epochs.pth"
if os.path.exists(model_path):
    model = torch.load(model_path).to(device)
    print("Loaded existing model.")
else:
    model = ViT().to(device=device)
    print("Initialized new model.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)

startTime = timer()

for epoch in tqdm.tqdm(range(EPOCHS), position=0, leave=True):
    
    model.train()
    train_labels = []
    train_preds = []
    
    train_loss = 0
    
    for batch, (X, y) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True)):
        X = X.to(device)
        y = y.to(device)
        
        y_pred = model(X)
        y_pred_label = y_pred.argmax(dim=1)
        
        train_labels.extend(y.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())
        
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss = train_loss / len(train_dataloader)
    
    model.eval()
    val_labels = []
    val_preds = []
    
    val_loss = 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm.tqdm(test_dataloader, position=0, leave=True)):
            X = X.to(device)
            y = y.to(device)
            
            y_pred = model(X)
            y_pred_label = y_pred.argmax(dim=1)
            
            val_labels.extend(y.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())
            
            loss = loss_fn(y_pred, y)
            
            val_loss += loss.item()
            
        val_loss = val_loss / len(test_dataloader)
    
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Test  Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(f"Test Accuracy EPOCH {epoch+1}: {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-"*30)

    # Save the model after each epoch
    torch.save(model, f"ViT_{epoch+1}_epochs.pth")
