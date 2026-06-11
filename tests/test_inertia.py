"""Inertial-property tests against closed-form solutions.

inertial_properties integrates the exact B-Rep, so box and cylinder results must
match the textbook formulas (at the default density of 1000 kg/m³). Off-center
solids verify the tensor is COM-referenced — the placement offset must not leak in.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import cadquery as cq
import numpy as np
import pytest

from kinechain.export_common import DEFAULT_DENSITY_KG_PER_M3, inertial_properties


def test_off_center_box_matches_analytic() -> None:
    # 20 x 30 x 40 mm box centered at (10, 20, 30) mm.
    shape = cq.Workplane("XY", origin=(10, 20, 30)).box(20, 30, 40).val().wrapped
    mass, com, inertia = inertial_properties(shape)

    a, b, c = 0.020, 0.030, 0.040  # m
    m_ref = a * b * c * DEFAULT_DENSITY_KG_PER_M3
    assert mass == pytest.approx(m_ref, rel=1e-6)
    assert com == pytest.approx([0.010, 0.020, 0.030], rel=1e-6)

    # COM-referenced tensor is that of a centered box: the offset must cancel.
    assert inertia[0, 0] == pytest.approx(m_ref / 12 * (b**2 + c**2), rel=1e-6)
    assert inertia[1, 1] == pytest.approx(m_ref / 12 * (a**2 + c**2), rel=1e-6)
    assert inertia[2, 2] == pytest.approx(m_ref / 12 * (a**2 + b**2), rel=1e-6)
    off_diagonal = inertia[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diagonal, 0.0, atol=1e-12)


def test_cylinder_matches_analytic() -> None:
    # Radius 5 mm, length 40 mm, axis Z, base at the origin (COM at z = 20 mm).
    shape = cq.Workplane("XY").circle(5).extrude(40).val().wrapped
    mass, com, inertia = inertial_properties(shape)

    r, h = 0.005, 0.040  # m
    m_ref = math.pi * r**2 * h * DEFAULT_DENSITY_KG_PER_M3
    assert mass == pytest.approx(m_ref, rel=1e-6)
    assert com == pytest.approx([0.0, 0.0, 0.020], abs=1e-9)

    assert inertia[2, 2] == pytest.approx(m_ref * r**2 / 2, rel=1e-6)
    i_perp = m_ref * (3 * r**2 + h**2) / 12
    assert inertia[0, 0] == pytest.approx(i_perp, rel=1e-6)
    assert inertia[1, 1] == pytest.approx(i_perp, rel=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed (install kinechain[sim] to enable)",
)
def test_mujoco_compiles_exact_pin_inertia(hinge_step: Path, tmp_path: Path) -> None:
    """The compiled MuJoCo model must carry the analytic mass and principal moments
    of the hinge pin (a plain cylinder), proving the emitted fullinertia survives
    MuJoCo's inertial processing intact."""
    import mujoco

    from kinechain.contact import find_contacts
    from kinechain.export_mjcf import write_mjcf
    from kinechain.graph import build_tree
    from kinechain.io_step import load_step
    from kinechain.joints import classify_all
    from kinechain.surfaces import extract_surfaces

    model = load_step(hinge_step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    tree = build_tree(model, classify_all(find_contacts(shapes, surfaces)))
    out = write_mjcf(tree, tmp_path / "hinge.xml")

    mj_model = mujoco.MjModel.from_xml_path(str(out))
    pin_id = mj_model.body("hinge_pin").id

    # rel=1e-5: the XML serializes values with 6 significant digits.
    r, h = 0.00395, 0.050  # pin radius / length from the fixture, in metres
    m_ref = math.pi * r**2 * h * DEFAULT_DENSITY_KG_PER_M3
    assert mj_model.body_mass[pin_id] == pytest.approx(m_ref, rel=1e-5)

    moments_ref = sorted([m_ref * r**2 / 2, *([m_ref * (3 * r**2 + h**2) / 12] * 2)])
    assert sorted(mj_model.body_inertia[pin_id]) == pytest.approx(moments_ref, rel=1e-5)
