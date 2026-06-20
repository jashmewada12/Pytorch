
"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
import data_setup, engine, model_builder, utils

import pandas as pd
import matplotlib.pyplot as plt
import sys
import importlib
import pred_plot as pred_plot_module
if "pred_plot" in sys.modules:
    importlib.reload(sys.modules["pred_plot"])
from pred_plot import pred_plot


from torchvision import transforms

TEST_TRANSFORM = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

TRAIN_TRANSFORM = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.TrivialAugmentWide(num_magnitude_bins=20),
  transforms.ToTensor()
])

PLOT_TRANSFORM = transforms.Compose([
  transforms.Resize((64,64),antialias= True)
  
])

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="Train a TinyVGG model with custom hyperparameters.")

# 2. Add the arguments we want to be able to change
parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to train for")
parser.add_argument("--batch_size", default=32, type=int, help="Number of samples per batch")
parser.add_argument("--hidden_units", default=10, type=int, help="Number of hidden units in the model")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for the optimizer")
parser.add_argument("--train_transform",default=TRAIN_TRANSFORM,type=transforms.Compose,help="Type of transforms to apply on training images")
parser.add_argument("--pred_img",default = r"C:\Free code camp\fcc ml\going_modular_05\data\images.jpeg",
type = str,help = "path of the image you want to make predictions on" )
# 3. Parse the arguments
args = parser.parse_args()

# Setup hyperparameters using the parsed arguments
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
TRAIN_TRANSFORM = args.train_transform
IMG_PATH =  r"C:\Free code camp\fcc ml\going_modular_05\data\images.jpeg"

print(f"[INFO]\nepochs : {NUM_EPOCHS}\nbatch size : {BATCH_SIZE}\nhidden units : {HIDDEN_UNITS}\nlearning rate : {LEARNING_RATE}\nTransforms : {TRAIN_TRANSFORM}")

# Setup directories 
train_dir = "../data/pizza_steak_sushi/train"
test_dir = "../data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"



# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names,class_dict = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=TRAIN_TRANSFORM,
    test_transform = TEST_TRANSFORM,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG_v1(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

#making predictions on a random image
pred_plot(model = model,test_data = test_dataloader.dataset,class_names = class_names,device =device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model_v1.pth")

#Plotting 


#Setup a plot
Tinyvgg_v1_results = pd.DataFrame(results)
plt.figure(figsize=(15,10))
#Get number of epochs
epochs = range(len(Tinyvgg_v1_results))


#Plotting the train loss
plt.subplot(2,2,1)
plt.plot(epochs,Tinyvgg_v1_results["train_loss"],label = "Train loss")
plt.plot(epochs,Tinyvgg_v1_results["test_loss"],label = "Test loss")
plt.title("Train & Test loss")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(2,2,2)
plt.plot(epochs,Tinyvgg_v1_results["test_acc"],label = "Test accuracy")
plt.plot(epochs,Tinyvgg_v1_results["train_acc"],label = "Train accuracy")
plt.title("Test loss")
plt.xlabel("Epochs")
plt.legend()
