from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = Image.open("data/hymenoptera_data/train/ants/0013035.jpg")

    # ToTensor
    tensor_transform = transforms.ToTensor()
    img_tensor = tensor_transform(img)
    print("img_tensor shape:", img_tensor.shape)
    print("imgae_tensor", img_tensor)

    # Normalize
    normalize_transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_tensor_normalize = normalize_transform(img_tensor)

    # plt.figure("test")
    # plt.imshow(img)
    # plt.show()

    writer = SummaryWriter("logs")
    writer.add_image("test_image", img_tensor, 0)
    writer.add_image("test_image", img_tensor_normalize, 1)
    writer.close()
