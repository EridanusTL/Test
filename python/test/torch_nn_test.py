import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, input):
        output = input + 1
        return output
        # return self.linear(input)


if __name__ == "__main__":
    input = torch.tensor(
        [
            [1, 2, 0, 3, 1],
            [0, 1, 2, 3, 1],
            [1, 2, 1, 0, 0],
            [5, 2, 3, 1, 1],
            [2, 1, 0, 1, 1],
        ]
    )
    kernel = torch.tensor([[1, 2, 1], [0, 1, 0], [2, 1, 0]])
    input = torch.reshape(input, (1, 1, 5, 5))
    kernel = torch.reshape(kernel, (1, 1, 3, 3))
    output = F.conv2d(input, kernel)
    output_2 = F.conv2d(input, kernel, padding=1)

    print(input.shape)
    print(input)
    print(output)
    print(output_2)
