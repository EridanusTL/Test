from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


if __name__ == "__main__":
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    test_set_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, drop_last=False
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    print(len(test_dataset))

    index = 0
    for data in test_set_loader:
        imgs, target = data
        writer.add_images("imgs", imgs, index)
        index += 1

    writer.close()
