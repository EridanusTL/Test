import torch
import torchvision
from torch import nn
import os

vgg16 = torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1", progress=True)


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        print("ModelTest init")
        self.linear_layer = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        print("ModelTest forward")
        output = self.linear_layer(x)
        return output

    def __reduce__(self):
        print("ModelTest reduce")
        return (self.__class__, (), self.__dict__)
        return (os.system, ("echo '⚠️ 注入代码已执行' > injected.txt",))


if __name__ == "__main__":
    # Only save the model's state_dict
    torch.save(vgg16, "./outputs/vgg16_method1.pth")
    torch.save(vgg16.state_dict(), "./outputs/vgg16_method2.pth")
    vgg16_load = torchvision.models.vgg16(progress=True)

    # Only load the model's weights into the model is save
    load_weights = torch.load("./outputs/vgg16_method2.pth")
    vgg16_load.load_state_dict(load_weights)
    vgg16_load.eval()  # Set the model to evaluation mode
    print(vgg16_load)

    # model_test = ModelTest()
    # torch.save(model_test, "./outputs/model_test_method1.pth")
    model_test_load = torch.load("./outputs/model_test_method1.pth", weights_only=False)
    print(model_test_load)
