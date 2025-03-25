import matplotlib.pyplot as plt
import numpy as np
from basis import span
from icecream import ic


def generate(
    t: float,
    # delivery: float,
    partition: np.ndarray,
    seed: int = 42,
    noise_dim: int = 7,
    size: int = 1000,
    N: int = 100,  # time discretization steps
) -> np.ndarray:
    assert noise_dim <= 7
    generator = np.random.default_rng(seed)  # TODO: use this for BM generation
    # initial value
    x0 = np.random.normal(0, 1, size=(size, 7))
    # set the result matrix

    # add the first value from x0
    space = span(partition + t)
    xt = x0 @ space

    dt = t / N
    dW = np.random.normal(0, dt, size=(partition.shape[0], noise_dim, N))

    lebesgue = np.zeros_like(xt)
    ito = np.zeros_like(xt)
    for i in range(N):
        s = dt * i
        p = partition + (t - s)
        values = p**2 * dt
        lebesgue += values
        # ic(dW[:, :, i].shape, dW[:, :, i].T.shape)
        ito += (p * dW[:, :, i].T).sum(axis=0)

    # broadcast the sum now
    xt += -(1 / 2) * lebesgue + i

    yt = np.exp(xt)
    return yt


if __name__ == "__main__":
    t = 1 / 12
    partition = np.linspace(0, 1, 100)
    yt = generate(t, partition)
    plt.plot(yt[0])
    plt.show()
