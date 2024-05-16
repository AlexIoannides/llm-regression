"""The llm_regression package."""
from .models import OpenAiRegressor
from .utils import make_univariate_linear_test_data

__all__ = [
    "OpenAiRegressor",
    "make_univariate_linear_test_data",
]
