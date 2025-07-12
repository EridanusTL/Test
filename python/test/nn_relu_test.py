import torch
import torch.nn as nn


input = torch.tensor([[1, -0.5], [-1, 3]])
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
output_1 = relu(input)
output_2 = sigmoid(input)

print(output_2)
