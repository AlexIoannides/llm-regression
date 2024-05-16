"""Helpful functions."""
from numpy import sqrt
from numpy.random import default_rng
from pandas import DataFrame


def make_univariate_linear_test_data(
    n_samples: int = 1000, *, rho: float = 0.75, seed: int = 42
) -> DataFrame:
    """Simulate a y = rho * x + sqrt(1 - rho ** 2) * epsilon.

    This paradign ensures that the standard deviation of x and y is always 1, and that
    x has correlation with y of rho.

    Args:
    ----
        n_samples: Number of samples to generate. Defaults to 1000.
        rho: Rho coeffcient (correlation coefficient). Defaults to 0.75.
        seed: Random seed. Defaults to 42.

    Returns:
    -------
        Dataframe of test data.
    """
    if not (rho >= 0 and rho <= 1):
        raise ValueError(f"rho = {rho} - must in [0, 1]")
    rng = default_rng(seed)
    x = rng.standard_normal(n_samples)
    epsilon = rng.standard_normal(n_samples)
    y = rho * x + sqrt(1 - rho * rho) * epsilon
    return DataFrame({"x": x, "y": y})
