import torch
import torch.nn as nn
import tqdm as tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Modules._01_config import *
from Modules._02_patchEmbedding import PatchEmbedding
from Modules._03_ViT import ViT
from Modules._04_dataset import train_dataloader, test_dataloader, class_names
from Modules._05_training import model

torch.cuda.empty_cache()

labels = []
ids = []
imgs = []
model.eval()
with torch.inference_mode():
    for batch, (X,y) in enumerate(tqdm.tqdm(test_dataloader, position=0, leave=True)):
        X=X.to(device)
        y=y.to(device)

        # ids.extend([int(i)+1 for i in y])
        
        y_pred_test = model(X)
        y_pred_test_label = y_pred_test.argmax(dim=1)
        
        imgs.extend(X.detach().cpu())
        labels.extend([int(i) for i in y_pred_test_label])
plt.figure()
f, axarr = plt.subplots(3, 4)
counter = 0
for i in range(3):
    for j in range(4):
        if counter < len(imgs):  # Check if the counter is within the range of available images

            axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")  # Use imgs[counter] to access individual images
            axarr[i][j].set_title(f"Predicted : {class_names[labels[counter]]}")
            axarr[i][j].axis("off")

            counter += 1
