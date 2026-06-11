"""kinechain CLI: STEP → MJCF/SDF (plus per-pair diagnostics on stderr)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contact import Tolerances, assembly_diagonal, find_contacts
from .export_mjcf import write_mjcf
from .export_sdf import write_sdf
from .graph import build_tree
from .io_step import load_step
from .joints import classify_all
from .surfaces import extract_surfaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinechain")
    parser.add_argument("step", type=Path, help="input STEP file")
    parser.add_argument("--out", type=Path, required=True, help="output model path")
    parser.add_argument("--format", choices=("mjcf", "sdf"), default="mjcf", help="output format")
    parser.add_argument("--root", type=str, default=None, help="part name to treat as root (optional)")
    args = parser.parse_args(argv)

    model = load_step(args.step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]

    diag = assembly_diagonal(shapes)
    tol = Tolerances.from_diagonal(diag)
    print(f"tolerances: pos={tol.pos:.3g} mm (from {diag:.3g} mm assembly diagonal)", file=sys.stderr)

    contacts = find_contacts(shapes, surfaces, tol)
    joints = classify_all(contacts, tol)

    # Diagnostics: contacts that matched no joint rule stay unjoined — make that visible
    # rather than silently dropping them.
    matched = {(j.parent, j.child) for j in joints}
    for c in contacts:
        if (c.i, c.j) in matched:
            continue
        print(
            f"unmatched: {model.parts[c.i].name} <-> {model.parts[c.j].name} "
            f"({len(c.coaxial_cylinders)} coaxial cylinder pairs, "
            f"{len(c.coincident_planes)} coincident plane pairs) — no joint rule matched",
            file=sys.stderr,
        )

    root_idx = None
    if args.root is not None:
        names = [p.name for p in model.parts]
        try:
            root_idx = names.index(args.root)
        except ValueError as e:
            raise SystemExit(f"--root {args.root!r} not found in: {names}") from e

    tree = build_tree(model, joints, root=root_idx)
    writer = write_mjcf if args.format == "mjcf" else write_sdf
    out = writer(tree, args.out)

    print(f"parts: {len(model.parts)}")
    print(f"joints: {len(tree.joints)} (tree) + {len(tree.loop_joints)} (loop)")
    for j in tree.joints:
        print(f"  {j.type.value}: {model.parts[j.parent].name} -> {model.parts[j.child].name}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
