import torchvision
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


if __name__ == "__main__":
    dataset_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10", train=True, transform=dataset_transform, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data/CIFAR10", train=False, transform=dataset_transform, download=True
    )

    img, target = test_set[0]
    print(img.shape)
    print(test_set.classes[target])

    # # plt
    # plt.figure()
    # plt.imshow(img.permute(1, 2, 0).numpy())
    # plt.show()

    # tensorboard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    for i in range(10):
        writer.add_image("test_set", test_set[i][0], i)
    writer.close()
