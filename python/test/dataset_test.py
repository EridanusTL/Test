from torch.utils.data import Dataset
import cv2
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv2.imread(img_item_path, cv2.IMREAD_COLOR)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f"Hi, {name}")  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == "__main__":
    # 数据集路径
    root_path = "/home/tlbot/code/Test/"
    data_path = root_path + "data/dataset/hymenoptera_dataset/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"

    # 创建数据集
    ants_dataset = MyDataset(data_path, ants_label_dir)
    bees_dataset = MyDataset(data_path, bees_label_dir)
    train_dataset = ants_dataset + bees_dataset

    # 显示数据
    # img, label = train_dataset.__getitem__(0)
    img, label = train_dataset[0]  # equivalent train_dataset.__getitem__(0)
    cv2.imshow("img", img)
    cv2.waitKey()
