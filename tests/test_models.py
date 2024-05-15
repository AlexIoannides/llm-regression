"""Tests for LLM regression modelling."""
from llm_regression.models import OpenAiRegressor


def test_OpenAiRegressor_parse_model_output():
    assert OpenAiRegressor._parse_model_output("\nOutput: 0.101") == 0.101
    assert OpenAiRegressor._parse_model_output("Output: -1.101") == -1.101
    assert OpenAiRegressor._parse_model_output("Output: 1") == 1.0
