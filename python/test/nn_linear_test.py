import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import logging
import colorlog


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=196608, out_features=10)
        # with torch.no_grad():
        #     # 初始化权重和偏置为全 1
        #     self.linear.weight.fill_(1.0)
        #     self.linear.bias.fill_(0)

        self.test_set = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        self.data_loader = DataLoader(self.test_set, batch_size=64)

    def forward(self, x):
        print("self.linear.weight: ", self.linear.weight)
        print("self.linear.bias: ", self.linear.bias)

        return self.linear(x)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    model_test = ModelTest()

    # input = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    # input = torch.reshape(input, (1, 4))
    # print("input: ", input)
    # print(input.shape)
    # output = model_test(input)
    # print("output: ", output)
    # logger.error(f"output: {output}")

    for data in model_test.data_loader:
        imgs, targets = data
        print(imgs.shape)
        output = torch.reshape(imgs, (1, -1)).squeeze(0)
        print(output.shape)

    writer.close()
