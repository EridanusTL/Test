import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time

train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)  # 50000
test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)  # 10000

train_dataloader = DataLoader(train_set, batch_size=64)
test_dataloader = DataLoader(test_set, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-2
epoch = 1000


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.cifar10_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=10),
        )
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, x):
        output = self.cifar10_model(x)
        return output


if __name__ == "__main__":
    model_test = ModelTest()
    model_test = model_test.to(device)
    print(model_test)
    optim = torch.optim.SGD(model_test.parameters(), lr=learning_rate, momentum=0.1)
    total_train_step = 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    start_time = time.time()
    for i in range(epoch):
        model_test.train()
        print(f"Epoch {i}/{epoch}")
        total_train_loss = 0
        train_step = 0
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model_test(imgs)
            res_loss = model_test.loss(output, targets)
            optim.zero_grad()
            res_loss.backward()
            optim.step()
            # print(f"tain step: {train_step} loss: {res_loss}")
            train_step += 1
            total_train_step += 1
            total_train_loss += res_loss.item()

            # if total_train_step % 100 == 0:
            #     end_time = time.time()
            #     print(f"time: {end_time - start_time:.2f}s")
            #     start_time = end_time
            #     print(f"train step: {total_train_step} loss: {res_loss.item()}")
        print(f"train_loss: {total_train_loss/train_step}")

        model_test.eval()
        with torch.no_grad():
            test_step = 0
            correct = 0
            total_test_loss = 0
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = model_test(imgs)
                res_loss = model_test.loss(output, targets)
                predicted = torch.argmax(output, 1)
                correct += (predicted == targets).sum().item()
                test_step += 1
                total_test_loss += res_loss.item()
            print(f"Test accuracy: {correct / len(test_set)}")
            print(f"test_loss: {total_test_loss/test_step}")

        writer.add_scalar("Loss/train", total_train_loss / train_step, i)
        writer.add_scalar("Loss/test", total_test_loss / train_step, i)
        writer.add_scalar("Accuracy", correct / len(test_set), i)

    writer.close()
