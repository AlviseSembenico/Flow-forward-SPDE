from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from basis import span_torch
from icecream import ic

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def generate(
    t: float,
    partition: torch.Tensor,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
) -> torch.Tensor:

    assert noise_dim <= 7
    torch.manual_seed(seed)
    # initial value
    x0 = torch.empty(size, 7, device=device).uniform_(-1 / 2, 1 / 2)

    # Add the first value from x0
    space = span_torch(partition + t)
    xt = torch.matmul(x0, space)

    dt = t / N
    dW = torch.normal(
        0, dt**0.5, size=(partition.shape[0], noise_dim, N), device=device
    )

    lebesgue = torch.zeros(N, device=device)
    ito = torch.zeros(N, device=device)

    for i in range(N):
        s = dt * i
        p = span_torch(partition + (t - s))
        values = (p**2).sum(dim=0) * dt

        lebesgue += values
        ito += torch.sum(p * dW[:, :, i].T, dim=0)  # shape: (N,)

    # Broadcast the sum now
    xt += -(1 / 2) * lebesgue + ito

    yt = torch.exp(xt)
    return yt


def generate_prices(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
    strike: int = 1,
):
    yt = generate(t, partition, seed, noise_dim, size, N)
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return yt, val


def generate_MC(
    t: float,
    partition: torch.Tensor,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
    M: int = 50,
):

    assert noise_dim <= 7
    torch.manual_seed(seed)

    # initial value
    x0 = torch.empty(size, 7, device=device).uniform_(-1 / 2, 1 / 2)

    # Add the first value from x0
    space = span_torch(partition + t)
    xt = torch.matmul(x0, space)  # shape: (size, partition.shape[0])

    dt = t / N
    dW = torch.normal(
        0, dt**0.5, size=(partition.shape[0], noise_dim, N, M), device=device
    )

    lebesgue = torch.zeros(N, device=device)
    # Independent of N since it's the same for each initial condition
    ito = torch.zeros((N, M), device=device)

    for i in range(N):
        s = dt * i
        p = span_torch(partition + (t - s))
        values = (p**2).sum(dim=0) * dt
        lebesgue += values

        increment = (p * dW[:, :, i, :].permute(2, 1, 0)).sum(dim=1)
        ito += increment.T

    # Broadcast the sum now
    xt = xt.unsqueeze(-1) - (1 / 2) * lebesgue + ito.unsqueeze(0)

    yt = torch.exp(xt)
    yt_mc = yt.mean(dim=2)
    return yt_mc


def generate_prices_MC(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
    strike: int = 1,
    M: int = 1,
):

    yt = generate_MC(t, partition, seed, noise_dim, size, N, M)
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return yt, val


@click.command()
@click.option("--t", type=float, default=1 / 12)
@click.option("--noise_dim", type=int, default=7)
@click.option("--size_train", type=int, default=100000)
@click.option("--size_test", type=int, default=10000)
@click.option("--N", type=int, default=100)
@click.option("--strike", type=int, default=1)
def generate_full_dataset(
    t: float, noise_dim: int, size_train: int, size_test: int, n: int, strike: int
):
    partition = torch.linspace(0, 1, n)
    train_x, train_y = generate_prices(
        t, partition, noise_dim=noise_dim, size=size_train, N=n, strike=strike
    )
    test_x, test_y = generate_prices_MC(
        t, partition, noise_dim=noise_dim, size=size_test, N=n, strike=strike
    )

    # save the data to the data folder
    Path("data").mkdir(exist_ok=True)
    torch.save(train_x, "data/train_x.pt")
    torch.save(train_y, "data/train_y.pt")
    torch.save(test_x, "data/test_x.pt")
    torch.save(test_y, "data/test_y.pt")


if __name__ == "__main__":
    generate_full_dataset()
