"""kinechain CLI: STEP → MJCF (or diagnostic dump)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contact import find_contacts
from .export_mjcf import write_mjcf
from .graph import build_tree
from .io_step import load_step
from .joints import classify_all
from .surfaces import extract_surfaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinechain")
    parser.add_argument("step", type=Path, help="input STEP file")
    parser.add_argument("--out", type=Path, required=True, help="output MJCF path")
    parser.add_argument("--root", type=str, default=None, help="part name to treat as root (optional)")
    args = parser.parse_args(argv)

    model = load_step(args.step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)

    root_idx = None
    if args.root is not None:
        names = [p.name for p in model.parts]
        try:
            root_idx = names.index(args.root)
        except ValueError as e:
            raise SystemExit(f"--root {args.root!r} not found in: {names}") from e

    tree = build_tree(model, joints, root=root_idx)
    out = write_mjcf(tree, args.out)

    print(f"parts: {len(model.parts)}")
    print(f"joints: {len(joints)} (tree) + {len(tree.loop_joints)} (loop)")
    for j in tree.joints:
        print(f"  {j.type.value}: {model.parts[j.parent].name} -> {model.parts[j.child].name}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
