import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


vgg16_true = torchvision.models.vgg16(
    weights="VGG16_Weights.IMAGENET1K_V1", progress=True
)
vgg16_false = torchvision.models.vgg16(progress=True)

if __name__ == "__main__":
    vgg16_true.classifier.add_module(
        name=str(len(vgg16_true.classifier)),
        module=nn.ReLU(inplace=True),
    )
    vgg16_true.classifier.add_module(
        name=str(len(vgg16_true.classifier)),
        module=nn.Linear(in_features=1000, out_features=10),
    )
    print(vgg16_true)
    print(vgg16_false)
    vgg16_false.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
    print(vgg16_false)

    # View parameters quantity
    features_params = sum(p.numel() for p in vgg16_true.features.parameters())
    classifier_params = sum(p.numel() for p in vgg16_true.classifier.parameters())
    total_params = sum(p.numel() for p in vgg16_true.parameters())
    print(f"features parameters: {features_params}")
    print(f"classifier parameters   : {classifier_params}")
    print(f"Total parameters: {total_params}")
