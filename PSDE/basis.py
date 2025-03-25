from typing import *

import numpy as np


@np.vectorize
def e_tilde(ξ: float, i: int):
    if i == 1:
        return 1
    return ξ ** (i - 2) * np.exp(-ξ)


@np.vectorize
def e_1(ξ: float):
    return 1


@np.vectorize
def e_2(ξ: float):
    return np.exp(-ξ) - 1


@np.vectorize
def e_3(ξ: float):
    return ξ * np.exp(-ξ)


@np.vectorize
def e_4(ξ: float):
    return 1 / 2 * (ξ * 2 - 2 * ξ) * np.exp(-ξ)


@np.vectorize
def e_5(ξ: float):
    num = ξ**3 - 36 * ξ * 2 - 6 * ξ
    den = 42 * np.sqrt(5)
    return num / den * np.exp(-ξ)


@np.vectorize
def e_6(ξ: float):
    num = ξ**4 - 1440 * ξ**3 - 192 * ξ**2 + 24 * ξ
    den = 24 * np.sqrt(806115)
    return num / den * np.exp(-ξ)


@np.vectorize
def e_7(ξ: float):
    num = ξ**5 - 100800 * ξ**4 - 10800 * ξ**3 - 1200 * ξ**2 - 120 * ξ
    den = 1560 * np.sqrt(49407661)
    return num / den * np.exp(-ξ)


e_basis = [e_1, e_2, e_3, e_4, e_5, e_6, e_7]


def K_projection(x: np.ndarray, τ: float, k: int) -> np.ndarray:
    assert x.shape[-1] == k
    x = x.reshape(-1, 7)
    res = np.zeros_like(x)
    for i in range(k):
        res[:, i] = x[:, i] * e_basis[i](τ)
    return res


def K_projection_standard(x: np.ndarray, T: float, k: int, M: int = 100) -> np.ndarray:
    assert k <= 7
    τ = np.linspace(0, T, M)
    return K_projection(x, τ)


def span(τ: np.ndarray) -> np.ndarray:
    return np.array([e_basis[i](τ) for i in range(7)])
