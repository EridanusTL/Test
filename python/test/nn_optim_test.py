import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
data_loader = DataLoader(test_set, batch_size=1)


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.cifar10_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.cifar10_model(x)
        return output


if __name__ == "__main__":
    model_test = ModelTest()
    print(model_test)
    optim = torch.optim.SGD(model_test.parameters(), lr=0.0001)

    for epoch in range(20):
        running_loss = 0.0
        for data in data_loader:
            imgs, targets = data
            output = model_test(imgs)
            res_loss = model_test.loss(output, targets)
            optim.zero_grad()
            res_loss.backward()
            optim.step()
            running_loss += res_loss
        print(res_loss)

    # input_test = torch.rand(1, 3, 32, 32)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    # writer.add_graph(model_test, input_test)

    writer.close()
