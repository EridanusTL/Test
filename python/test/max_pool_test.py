import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import os


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

    def forward(self, input):
        output = self.max_pool2d(input)
        return output


if __name__ == "__main__":
    model_test = ModelTest()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)
    test_dataset_loader = DataLoader(
        model_test.test_dataset, batch_size=64, drop_last=False
    )
    img, target = model_test.test_dataset[0]

    input = torch.tensor(
        [
            [1, 2, 0, 3, 1],
            [0, 1, 2, 3, 1],
            [1, 2, 1, 0, 0],
            [5, 2, 3, 1, 1],
            [2, 1, 0, 1, 1],
        ]
    )

    index = 0
    for data in test_dataset_loader:
        imgs, target = data
        writer.add_images("befor", imgs, global_step=index)
        index += 1
        imgs_output = model_test(imgs)
        writer.add_images("after", imgs_output, global_step=index)

    writer.close()
