"""Regression models using LLMs."""
import re

import numpy as np
from dotenv import load_dotenv
from numpy.random import default_rng
from openai import BadRequestError, OpenAI
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, r2_score


def predict(
        test_data: DataFrame, train_data: DataFrame, *, verbose: bool = False
    ) -> DataFrame:
    """Score a dataset using an LLM.

    Args:
    ----
        test_data: Dataframe of features/variables to use for prediction.
        train_data: Dataframe of labelled features/variables to use for training.
        verbose: Print prompt for first test data instances?

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

    prompt_train_data = [
        f"Feature 0: {row.x}\nOutput: {row.y}" for row in train_data.itertuples()
    ]

    y_pred: list[float] = []
    for row in test_data.itertuples():
        prompt_test_data = [f"Feature 0: {row.x}\nOutput:"]

        user_prompt = "\n".join(prompt_train_data + prompt_test_data)
        if verbose:
            print(user_prompt)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        try:
            prediction = completion.choices[0].message.content
        except BadRequestError as e:
            raise ModelError("API call to LLM failed") from e
        if prediction:
            y_pred += [_parse_model_output(prediction)]
        else:
            raise ModelError("prediciton failed")

    return DataFrame({"y_pred": y_pred})


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
    n_samples = 550
    train_test_split_idx = 500
    data = make_univariate_linear_test_data(n_samples)
    train_data = data.iloc[:train_test_split_idx,]
    test_data = data.iloc[train_test_split_idx:,]
    y_pred = predict(test_data, train_data)

    results = (
        test_data.copy()
        .reset_index(drop=True)
        .assign(y_pred=y_pred["y_pred"])
    )
    mean_abs_err = mean_absolute_error(results["y"], results["y_pred"])
    r_squared = r2_score(results["y"], results["y_pred"])
    print(f"mean_abs_error = {mean_abs_err}")
    print(f"r_squared = {r_squared}")
    print(results)
