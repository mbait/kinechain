# kinechain

Infer kinematic chains (joints + link tree) from 3D assembly files (STEP) that do not contain explicit mating information.

## Status

Early MVP. Currently targets a single end-to-end slice: detect a revolute joint from a two-part hinge fixture and export a working MJCF model.

## Approach

1. Load STEP with OCCT (via `cadquery-ocp`) preserving the assembly hierarchy and world transforms.
2. Extract analytic surfaces per part (planes, cylinders) with axis/origin/radius.
3. Detect surface-pair contacts using broad-phase AABB overlap + narrow-phase coincidence tests.
4. Pattern-match joints (revolute / prismatic / fixed) from contact features.
5. Build a kinematic tree, breaking cycles into loop constraints.
6. Emit MJCF (MuJoCo) — SDF emitter shares the same intermediate representation.

## Install

Requires Python 3.10–3.13 and your distro's Python venv package:

- Debian/Ubuntu: `sudo apt install python3-venv`
- Fedora/RHEL: `sudo dnf install python3-virtualenv` (or just `python3` which already ships `venv`)
- Arch: ships with `python` by default

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

All native dependencies (OCCT) ship as `manylinux_2_31` wheels via `cadquery-ocp`, so no `apt`/`dnf` packages are needed beyond the Python venv tooling.

## Usage

```sh
kinechain assembly.step --out model.xml --format mjcf
```

## Development

```sh
pytest
```
