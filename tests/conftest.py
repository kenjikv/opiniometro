"""Shared fixtures: a small model trained once, in a temporary folder."""

import copy
import os
import tempfile

# Set before anything imports src.config: tests never touch artifacts/ or the database.
os.environ["ARTIFACTS_DIR"] = tempfile.mkdtemp(prefix="opiniometro-test-")
os.environ.pop("DATABASE_URL", None)

import pytest  # noqa: E402

from src.config import load_params  # noqa: E402
from src.data.generate import generate  # noqa: E402


@pytest.fixture(scope="session")
def small_params():
    """The real parameters with a tiny search space, so training takes seconds."""
    params = copy.deepcopy(load_params())
    params["experiments"].update(
        cv_folds=3, normalizers=["stem"], stopword_lists=["adjusted"], ngram_ranges=[[1, 2]],
        models={"naive_bayes": {"alpha": [1.0]}, "logistic_regression": {"C": [1.0]}},
    )
    return params


@pytest.fixture(scope="session")
def trained(small_params):
    from src.models.train import train
    return train(generate(600, seed=7), small_params)


@pytest.fixture(scope="session")
def client(trained):
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as test_client:
        yield test_client
