"""Tests for LLM regression modelling."""
from llm_regression.models import _parse_model_output


def test_parse_model_output():
    assert _parse_model_output("\nOutput: 0.101") == 0.101
    assert _parse_model_output("Output: -1.101") == -1.101
    assert _parse_model_output("Output: 1") == 1.0
