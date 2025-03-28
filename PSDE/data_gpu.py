from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from basis import span_torch
from icecream import ic
from tqdm import tqdm

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
    return x0, yt


def generate_prices(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    strike: int = 1,
    N: int = 30,
):
    x0, yt = generate(t, partition, seed=seed, noise_dim=noise_dim, size=size, N=N)
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return x0, val


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
    return x0, yt_mc


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
    x0, yt = generate_MC(
        t, partition, seed=seed, noise_dim=noise_dim, size=size, M=M, N=N
    )
    val = yt.sum(dim=1) * (12 / 365)
    val = torch.maximum(val - strike, torch.zeros_like(val))

    return x0, val


@click.command()
@click.option("--t", type=float, default=1 / 12)
@click.option("--noise_dim", type=int, default=7)
@click.option("--size_train", type=int, default=100000)
@click.option("--size_test", type=int, default=10000)
@click.option("--partition_size", type=int, default=100)
@click.option("--strike", type=int, default=1)
@click.option("--N", type=int, default=30)
@click.option("--M", type=int, default=1000)
@click.option("--batch_size", type=int, default=1000)
@click.option("--test_only", is_flag=True, default=False)
@click.option("--train_only", is_flag=True, default=False)
def generate_full_dataset(
    t: float,
    noise_dim: int,
    size_train: int,
    size_test: int,
    partition_size: int,
    strike: int,
    n: int,
    m: int,
    batch_size: int,
    test_only: bool,
    train_only: bool,
):
    partition = torch.linspace(0, 1, partition_size, device=device)
    batch_size = min(batch_size, size_test)

    train_x, train_y = [], []
    if not test_only:
        for i in tqdm(range(0, size_train, 10 * batch_size)):
            x, y = generate_prices(
                t,
                partition,
                noise_dim=noise_dim,
                size=batch_size * 10,
                strike=strike,
                N=n,
            )
            train_x.append(x.cpu())
            train_y.append(y.cpu())
        train_x = torch.cat(train_x, dim=0)
        train_y = torch.cat(train_y, dim=0)
    # run the MC simulation in batches
    test_x, test_y = [], []
    if not train_only:
        for i in tqdm(range(0, size_test, batch_size)):
            x, y = generate_prices_MC(
                t,
                partition,
                noise_dim=noise_dim,
                size=batch_size,
                strike=strike,
                N=n,
                M=m,
            )
            test_x.append(x.cpu())
            test_y.append(y.cpu())
        test_x = torch.cat(test_x, dim=0)
        test_y = torch.cat(test_y, dim=0)

    # save the data to the data folder
    Path("data").mkdir(exist_ok=True)
    if not test_only:
        ic(train_x.shape, train_y.shape)
        torch.save(train_x, "data/train_x.pt")
        torch.save(train_y, "data/train_y.pt")
    if not train_only:
        ic(test_x.shape, test_y.shape)
        torch.save(test_x, "data/test_x.pt")
        torch.save(test_y, "data/test_y.pt")


if __name__ == "__main__":
    generate_full_dataset()
