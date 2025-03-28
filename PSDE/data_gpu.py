from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from basis import span_torch
from icecream import ic

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


def generate(
    t: float,
    partition: torch.Tensor,
    x0: torch.Tensor = None,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    dW: torch.Tensor = None,
    N: int = 30,
) -> torch.Tensor:
    X_size = partition.shape[0]
    assert noise_dim <= 7
    torch.manual_seed(seed)
    # initial value
    if x0 is None:
        x0 = torch.empty(size, 7, device=device, dtype=torch.float64).uniform_(
            -1 / 2, 1 / 2
        )
    else:
        size = x0.shape[0]

    # Add the first value from x0
    space = span_torch(partition + t)
    xt = torch.matmul(x0, space)

    dt = t / N
    if dW is None:
        dW = torch.normal(0, dt, size=(size, noise_dim, N), device=device)
    else:
        assert dW.shape == (size, noise_dim, N), (
            "dW shape is wrong "
            + str(dW.shape)
            + " expected "
            + str((x0.shape[0], noise_dim, N))
        )

    lebesgue = torch.zeros(X_size, device=device)
    ito = torch.zeros_like(xt, device=device)

    for i in range(N):
        s = dt * i
        p = span_torch(partition + (t - s))

        values = (p**2).sum(dim=0) * dt
        lebesgue += values

        p = p.unsqueeze(0).repeat(size, 1, 1)  # dim: size, 7, 365
        val = torch.sum(p.T * dW[:, :, i].T, dim=1).T
        ito += val

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
    strike: int = 1,
    N: int = 30,
):
    yt = generate(t, partition, seed=seed, noise_dim=noise_dim, size=size, N=N)
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return yt, val


def generate_MC(
    t: float,
    partition: torch.Tensor,
    x0: torch.Tensor = None,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    dW: torch.Tensor = None,
    M: int = 50,
    N: int = 30,
):
    X_size = partition.shape[0]
    assert noise_dim <= 7
    torch.manual_seed(seed)

    # initial value
    if x0 is None:
        x0 = torch.empty(size, 7, device=device).uniform_(-1 / 2, 1 / 2)
    else:
        size = x0.shape[0]

    # Add the first value from x0
    space = span_torch(partition + t)
    xt = torch.matmul(x0, space)  # shape: (size, partition.shape[0])

    dt = t / N
    if dW is None:
        dW = torch.normal(0, dt, size=(size, noise_dim, N, M), device=device)
    else:
        dW *= np.sqrt(dt)
        assert dW.shape == (size, noise_dim, N, M), (
            "dW shape is wrong "
            + str(dW.shape)
            + " expected "
            + str((size, noise_dim, N, M))
        )

    # Independent of N since it's the same for each initial condition
    lebesgue = torch.zeros(X_size, device=device)
    ito = torch.zeros(size, X_size, M, device=device)

    for i in range(N):
        s = dt * i
        p = span_torch(partition + (t - s))

        values = (p**2).sum(dim=0) * dt
        lebesgue += values

        p = p.repeat(size, 1, 1, M)  # dim: size, 7, 365
        p = p.reshape(X_size, size, noise_dim, M)
        increment = (p * dW[:, :, i, :]).sum(dim=2).reshape(size, X_size, M)
        ito += increment

    # Broadcast the sum now
    xt = xt - (1 / 2) * lebesgue
    xt = (xt.T + ito.T).T

    yt = torch.exp(xt)
    yt_mc = yt.mean(dim=2)
    return yt_mc


def generate_prices_MC(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    strike: int = 1,
    M: int = 50,
    N: int = 30,
):
    yt = generate_MC(t, partition, seed=seed, noise_dim=noise_dim, size=size, M=M, N=N)
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return yt, val


@click.command()
@click.option("--t", type=float, default=1 / 12)
@click.option("--noise_dim", type=int, default=7)
@click.option("--size_train", type=int, default=100000)
@click.option("--size_test", type=int, default=10000)
@click.option("--partition_size", type=int, default=365)
@click.option("--strike", type=int, default=1)
@click.option("--N", type=int, default=30)
@click.option("--M", type=int, default=1000)
def generate_full_dataset(
    t: float,
    noise_dim: int,
    size_train: int,
    size_test: int,
    partition_size: int,
    strike: int,
    n: int,
    m: int,
):
    partition = torch.linspace(0, t, partition_size)
    train_x, train_y = generate_prices(
        t, partition, noise_dim=noise_dim, size=size_train, strike=strike, N=n
    )
    test_x, test_y = generate_prices_MC(
        t, partition, noise_dim=noise_dim, size=size_test, strike=strike, N=n, M=m
    )

    # save the data to the data folder
    Path("data").mkdir(exist_ok=True)
    torch.save(train_x, "data/train_x.pt")
    torch.save(train_y, "data/train_y.pt")
    torch.save(test_x, "data/test_x.pt")
    torch.save(test_y, "data/test_y.pt")


if __name__ == "__main__":
    generate_full_dataset()
