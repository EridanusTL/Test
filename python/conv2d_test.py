import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0
        )

    def forward(self, input):
        x = self.conv1(input)
        return x


if __name__ == "__main__":
    tran_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False
    )

    my_model = ModelTest()
    print(my_model)
    # print(my_model.conv1.weight)

    # init tensorboard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("input", imgs, global_step=step)
        output = my_model(imgs)
        # print(output.shape)
        output_vis = torch.reshape(output, (-1, 3, 30, 30))
        # 只取前3个通道
        writer.add_images("output_vis", output_vis, global_step=step)
        step += 1

    # writer.close()
