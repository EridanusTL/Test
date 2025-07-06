from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime
import os


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)

    img = Image.open("data/pytorch.png")

    # ToTensor
    tensor_transform = transforms.ToTensor()
    img_tensor = tensor_transform(img)
    print("img_tensor shape:", img_tensor.shape)
    # print("imgae_tensor", img_tensor)

    # Normalize
    normalize_transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_tensor_normalize = normalize_transform(img_tensor)

    # Resize
    resize_transform = transforms.Resize((512, 512))
    img_resize = resize_transform(img)
    img_resize_tensor = tensor_transform(img_resize)
    print(img_resize)
    print(type(img_resize_tensor))

    # Compose
    trans_compose = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )
    img_compose = trans_compose(img)

    # RandomCrop
    random_crop_transform = transforms.RandomCrop(128)
    img_random_crop = random_crop_transform(img)
    img_random_crop = tensor_transform(img_random_crop)

    # plt.figure("test")
    # plt.imshow(img)
    # plt.show()

    writer = SummaryWriter(log_dir)
    writer.add_image("test_image", img_tensor, 0)
    writer.add_image("normalize", img_tensor_normalize, 0)
    writer.add_image("resize", img_resize_tensor, 0)
    writer.add_image("compose", img_compose, 0)
    writer.add_image("random_crop", img_random_crop, 0)
    writer.close()
