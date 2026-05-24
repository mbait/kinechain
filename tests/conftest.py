from pathlib import Path

import pytest

from tests.fixtures.make_hinge import export as export_hinge


@pytest.fixture(scope="session")
def hinge_step(tmp_path_factory) -> Path:
    """Build the hinge.step fixture in a session-scoped tmp dir.

    Avoids leaving artifacts in the repo and keeps each test run hermetic.
    """
    out = tmp_path_factory.mktemp("fixtures") / "hinge.step"
    return export_hinge(out)
