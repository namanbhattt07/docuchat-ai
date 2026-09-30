import atexit
import os
import shutil
import tempfile

# The test suite must never touch a developer's real data. Settings are read
# once, on first import of `app`, so these have to be set before anything
# imports it -- conftest.py is imported first. (Previously the health test's
# `with TestClient(app)` ran the app's startup hook -- create_all +
# ensure_schema -- against the real data/docuchat.db on every test run.)
# Deliberately unconditional: a stray DATABASE_URL in the environment must not
# be able to point the tests at real data either.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="docuchat-tests-")
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DATA_DIR}/test.db"
os.environ["CHROMA_PERSIST_DIRECTORY"] = f"{_TEST_DATA_DIR}/chroma"
os.environ["UPLOAD_DIRECTORY"] = f"{_TEST_DATA_DIR}/uploads"
os.environ["PROCESSED_DIRECTORY"] = f"{_TEST_DATA_DIR}/processed"

import pytest  # noqa: E402 -- must follow the environment setup above
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import Base  # noqa: E402
from app.services.ocr import set_ocr_engine  # noqa: E402
from app.services.providers import reset_providers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_injected_dependencies():
    """A test that installs a fake provider / OCR engine must not leak it into
    the next test (they are process-wide singletons)."""
    yield
    reset_providers()
    set_ocr_engine(None)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
