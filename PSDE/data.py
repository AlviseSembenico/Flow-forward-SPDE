import matplotlib.pyplot as plt
import numpy as np
from basis import span
from icecream import ic


def generate(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
) -> np.ndarray:
    assert noise_dim <= 7
    generator = np.random.default_rng(seed)  # TODO: use this for BM generation
    # initial value
    x0 = np.random.uniform(-1 / 2, 1 / 2, size=(size, 7))
    # set the result matrix

    # add the first value from x0
    space = span(partition + t)
    xt = x0 @ space

    dt = t / N
    dW = np.random.normal(0, dt, size=(partition.shape[0], noise_dim, N))

    lebesgue = np.zeros((N))
    ito = np.zeros((N))
    for i in range(N):
        s = dt * i
        p = partition + (t - s)
        values = p**2 * dt

        lebesgue += values
        ito += (p * dW[:, :, i].T).sum(axis=0)  # shape: (N,)

    # broadcast the sum now
    xt += -(1 / 2) * lebesgue + ito

    yt = np.exp(xt)
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
    val = yt.sum(axis=1) * (12 / 365)
    val = np.maximum(val - strike, 0)

    return val


def generate_MC(
    t: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
    M: int = 50,
):
    assert noise_dim <= 7
    generator = np.random.default_rng(seed)  # TODO: use this for BM generation
    # initial value
    x0 = np.random.uniform(-1 / 2, 1 / 2, size=(size, 7))
    # set the result matrix

    # add the first value from x0
    space = span(partition + t)
    xt = x0 @ space  # shape: (size, M)

    dt = t / N
    dW = np.random.normal(0, dt, size=(partition.shape[0], noise_dim, N, M))

    lebesgue = np.zeros((N,))
    # independent of N since it's the same for each initial condition
    ito = np.zeros((N, M))
    for i in range(N):
        s = dt * i
        p = partition + (t - s)
        values = p**2 * dt
        lebesgue += values

        increment = (p * dW[:, :, i, :].T).sum(axis=1)
        ito += increment.T

    # broadcast the sum now
    xt += -(1 / 2) * lebesgue

    # Reshape xt from (1000,100) to (1000,100,1)
    # This adds a new dimension that can broadcast with ito
    xt_reshaped = xt[:, :, np.newaxis]  # shape becomes (1000,100,1)

    # Reshape ito from (100,50) to (1,100,50)
    # This adds a new dimension that can broadcast with xt
    ito_reshaped = ito[np.newaxis, :, :]  # shape becomes (1,100,50)

    # Now broadcasting will work correctly
    result = xt_reshaped + ito_reshaped  # shape will be (1000,100,50)

    yt = np.exp(result)
    yt_mc = yt.mean(axis=2)
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
    val = yt.sum(axis=1) * (12 / 365)
    val = np.maximum(val - strike, 0)

    return val


if __name__ == "__main__":
    t = 1 / 12
    partition = np.linspace(0, 1, 100)
    # yt = generate(t, partition)
    yt = generate_MC(t, partition)
    ic(yt.shape)
