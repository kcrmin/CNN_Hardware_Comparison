# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# For image pixel
import numpy as np

# Get dataset and trained model path
import os


# Find a absolute directory path of a "current file"
def get_path(file=""):
    file_path = os.path.abspath(__file__)
    file_name = os.path.basename(__file__)
    current_path = file_path.replace(file_name, file)
    return current_path


class MnistCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Set & Configure Neural Network
        self.network = nn.Sequential(
            # Feature Extraction
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Classification
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def run(self, idx, batch):
        # Split images and labels
        images, labels = batch
        # Initiate self.forward()
        output = self(images)
        # Convert values to probability
        output = F.softmax(output, dim=1)
        # Get the max value
        probs, preds = torch.max(output, dim=1)
        # If prediction was wrong, negate
        if preds != labels:
            probs *= -1
        # Return idx and probability as dict
        return {idx: probs.item()}

    # Initiate when values are passed in
    def forward(self, xb):
        return self.network(xb)


# Return dl and list of images
def dl_images():
    # Get current path
    current_path = get_path()
    # Download dataset if not available
    dataset = MNIST(root=f"{current_path}/data/", download=True, transform=ToTensor())
    # Load data
    dl = DataLoader(dataset)
    # Append images to list
    images = [image for image, label in dl]
    # Return loaded data and list of images
    return dl, images


def init_model(batch_size):
    # Initiate model
    model = MnistCnnModel()
    # Get pth location
    pth_location = get_path("mnist-cnn.pth")
    # Load if pth available
    if os.path.exists(pth_location):
        model.load_state_dict(torch.load(pth_location))
    # Return model
    return model
