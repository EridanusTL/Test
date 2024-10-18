import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextExtractionModule(nn.Module):
    def __init__(self):
        super(ContextExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class GlobalGuidanceModule(nn.Module):
    def __init__(self):
        super(GlobalGuidanceModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(32, 32, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class MultiLayerFeatureFusionModule(nn.Module):
    def __init__(self):
        super(MultiLayerFeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class CGNet(nn.Module):
    def __init__(self):
        super(CGNet, self).__init__()
        self.cem = ContextExtractionModule()
        self.ggm = GlobalGuidanceModule()
        self.mlfm = MultiLayerFeatureFusionModule()
        self.final_conv = nn.Conv2d(
            32, 19, kernel_size=1, stride=1
        )  # assuming 19 classes for segmentation

    def forward(self, x):
        x = self.cem(x)
        ggm_output = self.ggm(x)
        x = self.mlfm(x, ggm_output)
        x = self.final_conv(x)
        return x


# Create the CGNet model
cgnet = CGNet()

# Print the model
print(cgnet)

# Print the parameters of the model
for name, param in cgnet.named_parameters():
    print(name, param.size())
