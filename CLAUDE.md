# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

kinechain infers kinematic chains (joints + link tree) from STEP assembly files that lack explicit mating information, and exports MJCF (MuJoCo) models. Currently an MVP covering a single end-to-end slice: detecting a revolute joint from a two-part hinge and exporting working MJCF. The full design rationale and roadmap (prismatic/fixed joints, SDF export, loop closure) live in `docs/PLAN.md`.

## Commands

```sh
source .venv/bin/activate        # venv at repo root; package installed editable
pip install -e ".[dev]"          # install with dev deps (cadquery, pytest)
pytest                           # run all tests
pytest tests/test_hinge.py::test_pipeline_yields_one_revolute  # single test
kinechain assembly.step --out model.xml [--root PartName]      # CLI
```

The MuJoCo round-trip test (`test_mjcf_loads_in_mujoco`) auto-skips unless `mujoco` is installed (`pip install -e ".[sim]"`). CI (`.github/workflows/ci.yml`) runs `pytest -v` on Python 3.11 and 3.12.

## Architecture

The pipeline is a linear data flow through `src/kinechain/`, one module per stage (see `cli.py:main` for the canonical wiring):

```
io_step.load_step        STEP file → AssemblyModel (flat list of named Parts,
                         shapes already transformed into world coordinates)
surfaces.extract_surfaces  per-part B-Rep faces → PartSurfaces (analytic
                         PlaneFace / CylinderFace records with axis/normal/radius)
contact.find_contacts    broad-phase AABB overlap + narrow-phase coincidence
                         tests → PartContact features (coaxial cylinder pairs,
                         coincident plane pairs)
joints.classify_all      rule-based pattern matching on contact features → Joint list
graph.build_tree         joints → KinematicTree (BFS spanning tree from a root;
                         non-tree edges become loop_joints for cycle closure)
export_mjcf.write_mjcf   KinematicTree → MJCF XML + per-part binary STL meshes
```

Key design points that span modules:

- **OCCT bindings come from `cadquery-ocp`** — imports are `from OCP.X import Y` (not `OCC.Core` as `docs/PLAN.md` says; the plan predates the binding choice). `cadquery` itself is a dev-only dependency used solely to build test fixtures.
- **Units**: everything upstream is millimetres (STEP convention); `export_mjcf.py` scales to metres (`_MM_TO_M`) for MuJoCo. Tolerances in `contact.Tolerances` are in mm/mm².
- **Joint classification is deterministic and conservative** (`joints.py`): rules return `None` rather than guessing, so unmatched part pairs stay unjoined (the CLI reports them on stderr). Rules run most-constraining first: Fixed → Revolute (Prismatic planned). The bolt-aspect heuristic and shoulder-plane lock keep bolted joints from misclassifying as revolute.
- **Indices, not references**: `Joint.parent`/`Joint.child` and `PartContact.i`/`j` are integer indices into `AssemblyModel.parts`, carried through the whole pipeline.
- **Triangulation is shared**: `extract_surfaces` meshes the shape (`BRepMesh_IncrementalMesh`) to estimate cylinder axial extent; the triangulation is cached on the shape and reused by the STL export.

## Tests

Test fixtures are built programmatically with cadquery (`tests/fixtures/make_hinge.py`, `make_bolted_plates.py`) and exported to STEP in a session-scoped tmp dir via fixtures in `conftest.py` — no binary STEP files need to be committed. Each fixture's geometry is deliberately shaped to discriminate rules: the hinge pin protrudes past both bracket faces (no shoulder contact → only revolute can fire); the bolted plates have coaxial stud/hole cylinders that *look* revolute but must classify as fixed. When adding a new joint type, follow this pattern: a `make_<fixture>.py` builder plus assertions on joint count, type, and axis direction (see `docs/PLAN.md` § Verification).
