"""Regression models using LLMs."""
import re

import numpy as np
from dotenv import load_dotenv
from numpy.random import default_rng
from openai import OpenAI
from pandas import DataFrame


def predict(test_instance: DataFrame, train_data: DataFrame) -> DataFrame:
    """Score a dataset using an LLM.

    Args:
    ----
        test_instance: Dataframe of features/variables to use for prediction.
        train_data: Dataframe of labelled features/variables to use for training.

    Returns:
    -------
        A dataframe with predicted values for the test data.
    """
    load_dotenv()  # load OPEN_API_KEY from .env file (if present)
    client = OpenAI()

    system_prompt = (
        "Your task is to provide your best estimate for ”Output”. Please provide that "
        "and only that, without any additional text."
    )

    prompt_test_data = [
        f"Feature 0: {row.x}\nOutput: {row.y}" for row in train_data.itertuples()
    ]

    prompt_test_data = [f"Feature 0: {test_instance['x'].values[0]}\nOutput:"]

    user_prompt = "\n".join(prompt_test_data + prompt_test_data)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    prediction = completion.choices[0].message.content
    if prediction:
        y_pred = _parse_model_output(prediction)
    else:
        raise ModelError("prediciton failed")
    return DataFrame({"y_pred": [y_pred]})


def _parse_model_output(output: str) -> float:
    """Parse the models's output."""
    try:
        result = re.findall(r"-?\d+\.\d+", output)[0]
        return float(result)
    except (ValueError, IndexError) as e:
        raise ModelError("invalid model prediction") from e


def make_univariate_linear_test_data(
        n_samples: int = 1000, rho: float = 0.5, seed: int = 42
    ) -> DataFrame:
    """Simulate a y = beta * x + sigma * epsilon.

    Args:
    ----
        n_samples: Number of samples to generate. Defaults to 1000.
        rho: Rho coeffcient (correlation coefficient). Defaults to 1.
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
    y = rho * x + np.sqrt(1 - rho * rho) * epsilon
    return DataFrame({"x": x, "y": y})


class ModelError(Exception):
    """Custom exception class for model errors."""

    pass


if __name__ == "__main__":
    data = make_univariate_linear_test_data(1001)
    train_data = data.iloc[:1000,]
    test_instance = data.iloc[1000:,]
    y_pred = predict(test_instance, train_data)
    print(f"x = {test_instance['x'].values[0]}, y_pred = {y_pred['y_pred'].values[0]}")
