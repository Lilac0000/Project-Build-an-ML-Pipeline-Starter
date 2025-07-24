import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store", help="Path to the CSV artifact")
    parser.addoption("--ref", action="store", help="Path to the reference CSV artifact")
    parser.addoption("--kl_threshold", action="store", help="KL divergence threshold")
    parser.addoption("--min_price", action="store", help="Minimum price filter")
    parser.addoption("--max_price", action="store", help="Maximum price filter")


@pytest.fixture(scope='session')
def data(request):
    csv_artifact = request.config.getoption("--csv")
    if not csv_artifact:
        pytest.fail("You must provide the --csv option on the command line")

    run = wandb.init(
        project="Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning",
        entity="nataliashmyreva-western-governors-university",
        job_type="data_tests",
        resume=True,
    )
    data_path = run.use_artifact(csv_artifact).file()
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    ref_artifact = request.config.getoption("--ref")
    if not ref_artifact:
        pytest.fail("You must provide the --ref option on the command line")

    run = wandb.init(
        project="Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning",
        entity="nataliashmyreva-western-governors-university",
        job_type="data_tests",
        resume=True,
    )
    data_path = run.use_artifact(ref_artifact).file()
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl = request.config.getoption("--kl_threshold")
    if kl is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(kl)


@pytest.fixture(scope='session')
def min_price(request):
    min_p = request.config.getoption("--min_price")
    if min_p is None:
        pytest.fail("You must provide min_price")
    return float(min_p)


@pytest.fixture(scope='session')
def max_price(request):
    max_p = request.config.getoption("--max_price")
    if max_p is None:
        pytest.fail("You must provide max_price")
    return float(max_p)
