import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, 128))

        # Hidden layers (14 layers of 128 neurons each)
        for _ in range(14):
            self.layers.append(nn.Linear(128, 128))

        # Output layer
        self.layers.append(nn.Linear(128, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply all hidden layers with ReLU
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))

        # Apply output layer (no ReLU)
        x = self.layers[-1](x)
        return x
