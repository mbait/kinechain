from pathlib import Path

import pytest

from tests.fixtures.make_bolted_plates import export as export_bolted_plates
from tests.fixtures.make_hinge import export as export_hinge
from tests.fixtures.make_slider import export as export_slider


@pytest.fixture(scope="session")
def hinge_step(tmp_path_factory) -> Path:
    """Build the hinge.step fixture in a session-scoped tmp dir.

    Avoids leaving artifacts in the repo and keeps each test run hermetic.
    """
    out = tmp_path_factory.mktemp("fixtures") / "hinge.step"
    return export_hinge(out)


@pytest.fixture(scope="session")
def bolted_step(tmp_path_factory) -> Path:
    """Build the bolted_plates.step fixture in a session-scoped tmp dir."""
    out = tmp_path_factory.mktemp("fixtures") / "bolted_plates.step"
    return export_bolted_plates(out)


@pytest.fixture(scope="session")
def slider_step(tmp_path_factory) -> Path:
    """Build the slider.step fixture in a session-scoped tmp dir."""
    out = tmp_path_factory.mktemp("fixtures") / "slider.step"
    return export_slider(out)
