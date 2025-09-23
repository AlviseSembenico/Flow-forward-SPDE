from typing import *

import numpy as np
import torch


@np.vectorize
def e_tilde(ξ: float, i: int):
    if i == 1:
        return 1
    return ξ ** (i - 2) * np.exp(-ξ)


@np.vectorize
def e_1(ξ: float):
    return 1


def e_1_torch(ξ: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(ξ)


@np.vectorize
def e_2(ξ: float):
    return np.exp(-ξ) - 1


def e_2_torch(ξ: torch.Tensor) -> torch.Tensor:
    return torch.exp(-ξ) - 1


@np.vectorize
def e_3(ξ: float):
    return ξ * np.exp(-ξ)


def e_3_torch(ξ: torch.Tensor) -> torch.Tensor:
    return ξ * torch.exp(-ξ)


@np.vectorize
def e_4(ξ: float):
    return 1 / 2 * (ξ**2 - 2 * ξ) * np.exp(-ξ)


def e_4_torch(ξ: torch.Tensor) -> torch.Tensor:
    return 1 / 2 * (ξ**2 - 2 * ξ) * torch.exp(-ξ)


@np.vectorize
def e_5(ξ: float):
    num = ξ**3 - 36 * ξ * 2 - 6 * ξ
    den = 42 * np.sqrt(5)
    return num / den * np.exp(-ξ)


def e_5_torch(ξ: torch.Tensor) -> torch.Tensor:
    num = ξ**3 - 36 * ξ**2 - 6 * ξ
    den = 42 * torch.sqrt(torch.tensor(5.0))
    return (num / den) * torch.exp(-ξ)


@np.vectorize
def e_6(ξ: float):
    num = ξ**4 - 1440 * ξ**3 - 192 * ξ**2 + 24 * ξ
    den = 24 * np.sqrt(806115)
    return (num / den) * np.exp(-ξ)


def e_6_torch(ξ: torch.Tensor) -> torch.Tensor:
    num = ξ**4 - 1440 * ξ**3 - 192 * ξ**2 - 24 * ξ
    den = 24 * torch.sqrt(torch.tensor(806115.0))
    return (num / den) * torch.exp(-ξ)


@np.vectorize
def e_7(ξ: float):
    num = ξ**5 - 100800 * ξ**4 - 10800 * ξ**3 - 1200 * ξ**2 - 120 * ξ
    den = 1560 * np.sqrt(49407661)
    return num / den * np.exp(-ξ)


def e_7_torch(ξ: torch.Tensor) -> torch.Tensor:
    num = ξ**5 - 100800 * ξ**4 - 10800 * ξ**3 - 1200 * ξ**2 - 120 * ξ
    den = 1560 * torch.sqrt(torch.tensor(49407661.0))
    return (num / den) * torch.exp(-ξ)


e_basis = [e_1, e_2, e_3, e_4, e_5, e_6, e_7]

e_basis_torch = [
    e_1_torch,
    e_2_torch,
    e_3_torch,
    e_4_torch,
    e_5_torch,
    e_6_torch,
    e_7_torch,
]


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


def span_torch(τ: torch.Tensor, dim:int) -> torch.Tensor:
    return torch.stack([e_basis_torch[i](τ) for i in range(dim)])


# Laguerre polynomials


def e_0_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = 1.0
    return num * torch.exp(-xi)


def e_1_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = xi
    return num * torch.exp(-xi)


def e_2_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = xi * (2 - xi) / 2.0
    return num * torch.exp(-xi)


def e_3_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = xi * (xi**2 - 6 * xi + 6) / 6.0
    return num * torch.exp(-xi)


def e_4_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = xi * (-(xi**3) + 12 * xi**2 - 36 * xi + 24) / 24.0
    return num * torch.exp(-xi)


def e_5_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = xi * (xi**4 - 20 * xi**3 + 120 * xi**2 - 240 * xi + 120) / 120.0
    return num * torch.exp(-xi)


def e_6_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = (
        xi
        * (-(xi**5) + 30 * xi**4 - 300 * xi**3 + 1200 * xi**2 - 1800 * xi + 720)
        / 720.0
    )
    return num * torch.exp(-xi)


def e_7_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = (
        xi
        * (
            xi**6
            - 42 * xi**5
            + 630 * xi**4
            - 4200 * xi**3
            + 12600 * xi**2
            - 15120 * xi
            + 5040
        )
        / 5040.0
    )
    return num * torch.exp(-xi)


def e_8_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = (
        xi
        * (
            -(xi**7)
            + 56 * xi**6
            - 1176 * xi**5
            + 11760 * xi**4
            - 58800 * xi**3
            + 141120 * xi**2
            - 141120 * xi
            + 40320
        )
        / 40320.0
    )
    return num * torch.exp(-xi)


def e_9_laguerre(xi: torch.Tensor) -> torch.Tensor:
    num = (
        xi
        * (
            xi**8
            - 72 * xi**7
            + 2016 * xi**6
            - 28224 * xi**5
            + 211680 * xi**4
            - 846720 * xi**3
            + 1693440 * xi**2
            - 1451520 * xi
            + 362880
        )
        / 362880.0
    )
    return num * torch.exp(-xi)


def span_torch_laguerre(τ: torch.Tensor, dim:int) -> torch.Tensor:
    return torch.stack(
        [
            e_0_laguerre(τ),
            e_1_laguerre(τ),
            e_2_laguerre(τ),
            e_3_laguerre(τ),
            e_4_laguerre(τ),
            e_5_laguerre(τ),
            e_6_laguerre(τ),
            e_7_laguerre(τ),
            e_8_laguerre(τ),
            e_9_laguerre(τ),
        ][:dim]
    )
