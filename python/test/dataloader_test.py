import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

if __name__ == "__main__":
    dataset_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    tran_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10", train=True, transform=dataset_transform, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10", train=False, transform=dataset_transform, download=True
    )

    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False
    )
    test_loader_drop_last = DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True
    )

    img, target = test_set[0]
    print(img.shape)
    print(target)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    for epoch in range(2):
        step = 0
        for data in test_loader:
            imgs, targets = data
            # print(imgs.shape)
            # print(targets)
            writer.add_images(f"Epoch: {epoch}", imgs, step)
            step += 1

    # step = 0
    # for data in test_loader_drop_last:
    #     imgs, targets = data
    #     # print(imgs.shape)
    #     # print(targets)
    #     writer.add_images("test_loader_drop_last", imgs, step)
    #     step += 1
