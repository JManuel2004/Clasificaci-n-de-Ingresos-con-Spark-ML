import pytest

from income_pipeline.session import build_session


@pytest.fixture(scope="session")
def spark():
    session = build_session("income-pipeline-tests", master="local[2]")
    yield session
    session.stop()
