"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
import data_setup, engine, model, utils
from timeit import Timer as timer
import multiprocessing


# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
HIDDEN_UNITS = 16
LEARNING_RATE = 0.001

# Setup directories
train_dir = r"F:\AI\PyTorch from FreeCodeCamp\05_0_PyTorch_Custom_Datasets\data\pizza_steak_sushi\train"
test_dir =  r"F:\AI\PyTorch from FreeCodeCamp\05_0_PyTorch_Custom_Datasets\data\pizza_steak_sushi\test"




# Setup target device
# device = "cuda"
# print(f"[INFO] Using device: {device}")

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])


# Create DataLoaders

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)


# Create model

model = model.TVGGModel(
    input_features=3,
    hidden_features=HIDDEN_UNITS,
    output_features=len(class_names)
)


# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Start training

results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS)


# Save the model
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")

