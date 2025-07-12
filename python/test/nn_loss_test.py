import torch
import torch.nn as nn

if __name__ == "__main__":
    loss_l1 = nn.L1Loss(reduction="mean")
    loss_mse = nn.MSELoss(reduction="mean")
    loss_cross = nn.CrossEntropyLoss()

    input = torch.tensor([1, 2, 3]).float()
    target = torch.tensor([1, 2, 5]).float()
    print(input)
    print(target)
    output_l1 = loss_l1(input, target)
    output_mse = loss_mse(input, target)
    print(output_l1)
    print(output_mse)

    # CrossEntropyLoss
    input = torch.tensor([0.1, 0.2, 0.3])
    input = torch.reshape(input, (1, 3))
    y = torch.tensor([1])
    output_cross = loss_cross(input, y)
    print(output_cross)
