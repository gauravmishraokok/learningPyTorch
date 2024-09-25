import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Define the path and read the data
data_path = Path("data")
df = pd.read_csv(data_path / "A_Z Handwritten Data.csv").astype("float32")

# Extract features and labels
X_values = df.iloc[:, 1:].values
y_values = df.iloc[:, 0].values

# Convert to appropriate numpy types
X_values = X_values.astype(np.float32)
y_values = y_values.astype(np.float32)

# Convert to tensors
X_tensor = torch.tensor(X_values, dtype=torch.float32)
y_tensor = torch.tensor(y_values, dtype=torch.int64)

# Define the transformation
transformation = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((28, 28)), 
    transforms.ToTensor()
])

# Custom dataset class
class HandwrittenDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Reshape image to (1, 28, 28) from (784,)
        image = image.reshape(1, 28, 28)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create dataset objects
train_dataset = HandwrittenDataset(X_train, y_train, transform=transformation)
test_dataset = HandwrittenDataset(X_test, y_test, transform=transformation)

# Define batch size
BATCH_SIZE = 64  # Adjust as necessary

# Create dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
