import numpy as np
import torch
import torch.nn as nn

from optimizers import MyAdam


class SimpleCubicNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleCubicNet, self).__init__()
        # Define the single neuron in the hidden layer
        self.hidden = nn.Linear(input_size, 1, bias=False)
        # Output layer: Maps from the hidden layer to the output
        self.output = nn.Linear(1, output_size, bias=False)

    def forward(self, x):
        # Apply the cubic activation function on the hidden layer
        x = self.hidden(x).pow(3)
        # Pass the result through the output layer
        x = self.output(x)
        return x


def correct_adam(x, y, a0, b0, h, beta=0.9, rho=0.999, eps=1e-8):
    a, b = a0, b0
    nu1, nu2 = 0., 0.
    m1, m2 = 0., 0.
    for n in range(5):
        print(f'Iteration {n}')
        grad1 = 2 * b * x * 3 * (a * x) ** 2 * (b * (a * x) ** 3 - y)  # note the two
        grad2 = 2 * (a * x) ** 3 * (b * (a * x) ** 3 - y)  # note the two
        print(f'    grad1 = {grad1}, grad2 = {grad2}')

        nu1 = rho * nu1 + (1 - rho) * grad1 ** 2
        nu2 = rho * nu2 + (1 - rho) * grad2 ** 2
        m1 = beta * m1 + (1 - beta) * grad1
        m2 = beta * m2 + (1 - beta) * grad2

        a = a - h / np.sqrt(nu1 / (1 - rho ** (n + 1)) + eps) * m1 / (1 - beta ** (n + 1))
        b = b - h / np.sqrt(nu2 / (1 - rho ** (n + 1)) + eps) * m2 / (1 - beta ** (n + 1))

        print(f'    a = {a}, b = {b}')

    return a, b


def test_my_adam():
    model = SimpleCubicNet(input_size=1, output_size=1)

    # Manually set the initial weights
    model.hidden.weight.data = torch.tensor([[6.8]])  # Weight for the hidden neuron
    model.output.weight.data = torch.tensor([[6.73]])  # Weight for the output neuron

    # Move the model to the appropriate device
    device = torch.device('cpu')
    model.to(device)

    lr = 1e-1
    eps = 100000.

    optimizer = MyAdam(model.parameters(), lr=lr, eps=eps)

    criterion = nn.MSELoss()

    inputs = torch.tensor([[2.0]], device=device)
    targets = torch.tensor([[1.5]], device=device)

    for iteration in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        a = model.hidden.weight.item()
        b = model.output.weight.item()

    correct_a, correct_b = correct_adam(x=2.0, y=1.5, a0=6.8, b0=6.73, h=lr, beta=0.9, rho=0.999, eps=eps)

    assert abs(a - correct_a) < 1e-6
    assert abs(b - correct_b) < 1e-6
