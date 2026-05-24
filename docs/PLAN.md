# Kinematic Chain Generation from 3D Assemblies — MVP Plan

## Context

Goal: derive a kinematic model (joints + link tree) from a static 3D assembly that lacks explicit mating information. The downstream use is mechanical simulation in MuJoCo / Gazebo, so the output must be an SDF or MJCF model with correctly classified joints, axes, and origins.

The fundamental difficulty: assembly files describe *geometry in place*, not *intent*. Two coaxial cylinders touching at a planar shoulder *might* be a hinge, *might* be a press-fit (fixed), *might* be a roller bearing. The MVP commits to a deterministic, conservative rule set that returns *something plausible* and is easy to extend, rather than a learned/probabilistic classifier.

### Question 1 — Format

**Decision: STEP (AP203 / AP214 / AP242).**

Why STEP over STL: STL is triangle soup with no part boundaries and no analytic surfaces. STEP preserves (a) part hierarchy and assembly transforms, and (b) analytic B-Rep surfaces — planes, cylinders, cones, spheres, tori — with their *axes, origins, and radii directly available*. Joint inference is essentially "find pairs of analytic surfaces that share an axis or normal," which is one-line math on STEP and a multi-stage segmentation pipeline on STL.

Other candidates considered: glTF / 3MF (separated parts but still meshes — no analytic surfaces), native CAD formats (proprietary), URDF/SDF (these are the *output*).

### Question 2 — Starting point

**Decision: contact graph + analytic-surface pattern matching.**

The minimum useful behaviour: for each ordered pair of parts whose bounding boxes intersect, enumerate coincident analytic-surface pairs and emit a joint when a pattern matches one of three templates:

- **Revolute**: two cylindrical surfaces with coaxial axes (parallel + perpendicular distance ≈ 0), radii within tolerance, *no* large planar contact with normal *parallel* to the axis on both sides (which would lock rotation → fixed).
- **Prismatic**: two planar surface pairs whose normals are antiparallel, with a second, non-parallel planar pair (or non-cylindrical extruded contact) preventing rotation about the slide axis. Slide direction = intersection of the two plane normals.
- **Fixed**: large planar contact area plus at least one short coaxial cylindrical pair (bolt-like — length < diameter × k), OR multiple non-coplanar planar contacts that over-constrain motion.

Defaults if nothing matches: leave the parts unjoined and report them as floating in a diagnostics output (do not silently weld).

## Implementation Plan

Stack: **Python 3.11 + pythonocc-core + numpy + networkx**. Output: **MJCF** for MuJoCo as primary target (richer contact model, well-documented schema), with an SDF emitter as a thin secondary path sharing the same intermediate representation.

### Module layout (single-package, flat)

```
kinechain/
  io_step.py         # STEP loader → AssemblyModel
  surfaces.py        # extract analytic surfaces per part
  contact.py         # broad-phase (AABB) + narrow-phase coincidence tests
  joints.py          # rule-based joint classification
  graph.py           # kinematic tree assembly, cycle detection
  export_mjcf.py     # MJCF writer
  export_sdf.py      # SDF writer (shares IR with MJCF)
  cli.py             # entry point: step → mjcf/sdf
tests/
  fixtures/          # tiny hand-built STEP assemblies per joint type
```

### Step-by-step

1. **`io_step.py` — Load STEP into an `AssemblyModel`.**
   Use `OCC.Core.STEPControl.STEPControl_Reader` plus `XCAFApp` / `XCAFDoc_DocumentTool` to preserve the assembly tree (part names, occurrences, world transforms). Produce `AssemblyModel = { parts: [Part], transforms: {part_id: 4x4} }` where `Part` carries its `TopoDS_Shape` already transformed into world coordinates.

2. **`surfaces.py` — Extract analytic surfaces.**
   For each part, walk faces via `TopExp_Explorer(shape, TopAbs_FACE)`. For each face use `BRepAdaptor_Surface.GetType()` and emit a typed record:
   - Plane → `(point, normal, bounded_area)`
   - Cylinder → `(axis_point, axis_dir, radius, axial_extent)`
   - Cone, Sphere, Torus → captured but not used by MVP rules.
   Bounded area / axial extent comes from `BRepGProp.SurfaceProperties` + the face's UV bounds. Skip free-form (BSpline) surfaces for MVP.

3. **`contact.py` — Find touching surface pairs.**
   Broad phase: build an AABB tree over part bounding boxes (`Bnd_Box` + `BRepBndLib.Add`); query overlapping pairs. Narrow phase: for each overlapping part pair, test surface pairs of compatible types:
   - **Plane–plane coincidence**: normals antiparallel (dot < −1 + ε), signed distance ≈ 0, projected overlap area > area-threshold.
   - **Cylinder–cylinder coaxiality**: axes parallel (|dot| > 1 − ε), perpendicular distance between axis lines < ε_pos, radius difference < ε_rad, overlapping axial extent > min-overlap.
   Use `BRepExtrema_DistShapeShape` only as a fallback verification — analytic tests are orders of magnitude faster.

4. **`joints.py` — Classify joints from contact features.**
   For each part pair with contacts, run the three rules above in priority order: Fixed → Prismatic → Revolute (most-constraining first). Emit a `Joint = { type, parent, child, axis_point, axis_dir, limits=None }`. Tolerances live in a single `Tolerances` dataclass so they are easy to tune per assembly scale.

5. **`graph.py` — Build the kinematic tree.**
   Construct an undirected graph from joints (nodes = parts, edges = joints). Choose the root: the part with the largest mass-equivalent volume (proxy for "the base") unless the user supplies a `--root` part name. Run BFS to orient edges parent→child. **Closed loops** (cycles in the joint graph) are real — a four-bar linkage will produce one. Break each cycle by demoting one edge to a `loop_joint` (MJCF: `<equality>` constraint between two added sites; SDF: `<joint>` with the second body explicitly named).

6. **`export_mjcf.py` — Emit MJCF.**
   Write `<mujoco><worldbody>…</worldbody><equality>…</equality></mujoco>`. Each part becomes a `<body>` with `<inertial>` (computed via `BRepGProp.VolumeProperties` × user-supplied density, default 1000 kg/m³) and a `<geom type="mesh">` referencing an STL exported per-part. Each joint becomes a `<joint type="hinge|slide">` (omit for fixed — fixed children become rigid children of the parent body, no joint element). Loop-closing joints become `<equality><connect>` between two sites.

7. **`cli.py` — Wire it together.**
   `python -m kinechain assembly.step --out model.xml --format mjcf [--root PartName]`. Stream a per-pair diagnostics log so unclassified contacts are visible: `unmatched: PartA ↔ PartB (3 cyl pairs, 1 plane pair) — radii mismatch`.

### Critical files / APIs to reuse

- `OCC.Core.STEPCAFControl.STEPCAFControl_Reader` — preserves assembly hierarchy (not just `STEPControl_Reader`).
- `OCC.Core.BRepAdaptor.BRepAdaptor_Surface` — face → analytic surface type and parameters.
- `OCC.Core.Bnd.Bnd_Box` + `BRepBndLib.brepbndlib_Add` — broad-phase bounding boxes.
- `OCC.Core.BRepGProp` — surface area + volume + center of mass for inertial properties.
- `networkx.Graph` + `networkx.cycle_basis` — tree extraction and loop detection.

No existing codebase to integrate with (fresh project at `/home/mbait`), so the plan is greenfield.

## Verification

End-to-end test plan, ordered easiest first:

1. **Unit fixtures** — three hand-modelled STEP assemblies in `tests/fixtures/`:
   - `hinge.step`: door hinge (one revolute joint expected).
   - `slider.step`: dovetail slide (one prismatic joint expected).
   - `bolted_plates.step`: two plates with four bolt-cylinders (one fixed joint expected, no revolutes despite the cylinders — verifies the "short cylinder + planar contact = fixed" override).
   For each, assert joint count, joint type, axis direction (dot-product to expected), and parent/child assignment.

2. **Four-bar linkage** — verifies cycle handling. Assembly produces three revolute joints in the tree plus one loop-closing constraint. Assert that `len(joints) == 4` and `len(loop_joints) == 1`.

3. **Round-trip simulation** — load generated MJCF into MuJoCo (`mujoco.MjModel.from_xml_path` + `viewer.launch_passive`). Manually verify that the hinge rotates, the slider translates, and the four-bar moves coherently. (Manual step — no automated assertion, just a smoke test the developer runs once.)

4. **Diagnostics** — run on a deliberately ambiguous assembly (two coaxial cylinders with no shoulder plane) and confirm it lands in the unmatched-pair log rather than being silently classified.

## Out of scope (explicitly deferred)

- Spherical / planar / universal joints (add as a fourth rule later).
- Joint *limits* (would require interpreting "stop" features — geometric reasoning beyond MVP).
- Friction, damping, actuator inference.
- STL ingestion (would require a mesh-segmentation + primitive-fitting front end; design the `surfaces.py` interface so it can be swapped in later).
- Learning-based classification — the rule set is intentionally deterministic for the MVP so failures are debuggable.
