from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    # 数据加载和预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST的均值和标准差
        ]
    )
    # 下载并加载训练和测试数据
    data_dir = "./data"
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    writer = SummaryWriter()

    for i in range(100):
        writer.add_scalar("y=2x", 2 * i, i)
        writer.add_scalar("y=x^2", i**2, i)

    writer.add_image()
    writer.close()

    data_path = "data/MNIST/raw/t10k-labels-idx1-ubyte"
    images = read_idx_images(data_path)
    image_0 = Image.open(images)
    image_0.show()
s
