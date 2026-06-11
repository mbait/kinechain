# kinechain — Technical Report

Inferring kinematic chains from static CAD assemblies without explicit mating information.

This document records the concepts, design decisions, and implementation details of the
kinechain system. It is written as source material for a future whitepaper: each section
states *what* was built, *why* that choice was made over alternatives, and *how* it is
implemented, with enough precision that the method could be reproduced from this text alone.

---

## 1. Problem statement

Mechanical simulation environments (MuJoCo, Gazebo) require a *kinematic model*: a tree of
rigid bodies connected by typed joints (revolute, prismatic, fixed, …), each with an axis,
an origin, and a parent/child orientation. CAD assembly files, by contrast, describe
*geometry in place* — parts positioned in space — and in the common case carry **no mating
or constraint information** (mates are stored in proprietary native formats and are lost on
export to neutral formats such as STEP).

The task: given a static assembly file, recover a plausible kinematic model automatically.

The fundamental difficulty is that geometry under-determines intent. Two coaxial cylinders
touching at a planar shoulder *might* be a hinge, *might* be a press-fit (fixed), *might*
be a roller bearing. Any inference system must therefore commit to a policy for ambiguity.
kinechain's policy is **deterministic conservatism**: a small set of explicit geometric
rules, applied most-constraining-first; when no rule matches, the part pair is left
*unjoined* and reported in diagnostics rather than silently welded or guessed. This makes
failures debuggable and the rule set incrementally extensible, at the cost of recall on
unusual mechanisms — a deliberate trade against learned/probabilistic classifiers, whose
failure modes are opaque.

## 2. Background and basic concepts

**B-Rep (boundary representation).** STEP files (ISO 10303, AP203/AP214/AP242) describe
each part as a closed shell of *faces*, each face carrying an underlying *analytic surface*
(plane, cylinder, cone, sphere, torus) or a free-form (NURBS) surface, trimmed by edge
loops. Crucially, analytic surfaces expose their defining parameters directly: a plane has
a point and a normal; a cylinder has an axis line (point + direction) and a radius.

**Why STEP and not mesh formats.** STL/glTF/3MF are triangle soup: no part boundaries (STL)
and no analytic surfaces (all of them). Joint inference is essentially "find pairs of
analytic surfaces that share an axis or normal" — one-line vector algebra on B-Rep, but a
multi-stage segmentation-and-primitive-fitting pipeline on meshes. STEP additionally
preserves the assembly hierarchy: named parts ("occurrences") with placement transforms.

**Kinematic chain.** A graph whose nodes are rigid bodies and whose edges are joints. Most
simulators want a *tree* (each body has one parent); genuine closed loops (e.g. a four-bar
linkage) are handled by exporting a spanning tree plus *loop-closure constraints* for the
removed edges.

**Contact graph.** The intermediate structure between geometry and kinematics: nodes are
parts, and an edge between two parts carries the set of *contact features* found between
their surfaces (coaxial cylinder pairs, coincident plane pairs). Joint classification is a
function from a single edge's feature set to a joint type (or to "no match").

**Joint signatures in contact features.** The core insight the rules encode:

| Joint | Geometric signature | Counter-signal |
|---|---|---|
| Revolute | coaxial cylinder pair with long axial overlap (a bearing bore) | a face-to-face plane whose normal is parallel to the axis (a shoulder), or short overlap (a bolt stub) |
| Fixed | large face-to-face planar contact + bolt-like cylinder pair, or contact-plane normals spanning 3D (every translation blocked) | — |
| Prismatic | ≥2 non-parallel face-to-face plane pairs whose normals all share a common perpendicular — the slide axis | a contact normal with a component along the slide axis (→ fixed) |

## 3. System design

### 3.1 Pipeline architecture

The system is a linear data flow of six stages, one module per stage, each communicating
through small typed dataclasses (`src/kinechain/`, wired in `cli.py`):

```
STEP file
  │  io_step.load_step          — STEP → AssemblyModel
  ▼
AssemblyModel { parts: [Part(name, shape)] }       shapes in world coordinates
  │  surfaces.extract_surfaces  — per part: B-Rep faces → analytic records
  ▼
PartSurfaces { planes: [PlaneFace], cylinders: [CylinderFace] }
  │  contact.find_contacts      — broad phase (AABB) + narrow phase (coincidence)
  ▼
[PartContact { i, j, coaxial_cylinders, coincident_planes }]
  │  joints.classify_all        — rule-based pattern matching, per part pair
  ▼
[Joint { type, parent, child, axis_point, axis_dir }]
  │  graph.build_tree           — root selection, BFS spanning tree, loop split
  ▼
KinematicTree { root, parts, joints, loop_joints }
  │  export_mjcf.write_mjcf     — MJCF XML + per-part STL meshes
  │  export_sdf.write_sdf       — SDF XML, same IR and meshes (--format sdf)
  ▼
model.xml + meshes/*.stl
```

Design properties of this decomposition:

- **Index-based references.** Parts are identified by integer index into
  `AssemblyModel.parts` throughout (`PartContact.i/j`, `Joint.parent/child`). This keeps
  every intermediate trivially serializable and decouples stages from OCCT handle
  lifetimes.
- **Format-agnostic intermediate representation.** `KinematicTree` is the IR shared by all
  emitters; the planned SDF emitter consumes the same structure, so emitters are thin.
- **World-coordinate normalization at the boundary.** All assembly placement transforms
  are applied once, at load time. Every downstream stage reasons in a single world frame,
  so coincidence tests never need to compose transforms.

### 3.2 Surface extraction

For each part, faces are enumerated and only analytic types relevant to the current rules
are kept:

- `PlaneFace(point, normal, area)` — *normal is the outward (material-away) normal*, see
  §4.2; area from surface integration.
- `CylinderFace(axis_point, axis_dir, radius, axial_extent, area)` — `axial_extent` is the
  length of the face's projection onto its own axis, used to distinguish a long bearing
  bore from a short bolt stub (§4.3).

Cones, spheres, tori, and free-form surfaces are recorded nowhere and simply skipped; they
become relevant only when corresponding joint rules (spherical, etc.) are added.

### 3.3 Contact detection

Two-phase, as in collision-detection practice:

**Broad phase.** Axis-aligned bounding boxes per part, inflated by the positional
tolerance; only AABB-overlapping part pairs proceed. O(n²) pair enumeration is acceptable
at assembly scale (tens to hundreds of parts); an AABB tree is the obvious upgrade path.

**Narrow phase.** For each candidate part pair, all cross-part surface pairs of compatible
type are tested with closed-form predicates (ε values from a single `Tolerances`
dataclass; defaults: 1 mm positional, cos 2.5° angular, 0.5 mm radius):

- *Cylinder–cylinder coaxiality:* axes parallel, |d̂ₐ · d̂ᵦ| ≥ cos ε_θ; radii within ε_r;
  perpendicular distance between the (parallel) axis lines within ε_pos, computed as
  ‖(p_b − p_a) − ((p_b − p_a) · d̂ₐ) d̂ₐ‖. The pair additionally records the **axial
  overlap**: project both faces' extents onto the shared axis and intersect the intervals.
  Pairs with overlap below a minimum are discarded (grazing contact).
- *Plane–plane coincidence:* normals parallel up to sign, |n̂ₐ · n̂ᵦ| ≥ cos ε_θ; out-of-plane
  separation |(p_b − p_a) · n̂ₐ| ≤ ε_pos. The pair records:
  - `antiparallel` — n̂ₐ · n̂ᵦ < 0. Because normals are outward, antiparallel means the
    faces *face each other*: genuine face-to-face contact. Parallel-same-direction
    coincident faces are merely *flush* (e.g. a pin end sitting level with a bracket face)
    and are not contact.
  - `overlap_area` — the *exact* shared contact area (up to tessellation): both faces'
    cached triangulations are projected into the first face's plane and intersected
    triangle-by-triangle with convex (Sutherland–Hodgman) clipping, summing the clipped
    areas. This correctly returns zero for coplanar faces that do not overlap
    laterally — a case a min-of-face-areas estimate gets maximally wrong.

Analytic predicates are used exclusively; exact distance queries (`BRepExtrema`) are
reserved as a fallback verification because they are orders of magnitude slower.

### 3.4 Joint classification rules

Rules run **most-constraining first** — FIXED → PRISMATIC → REVOLUTE — and the first
match wins. Ordering matters: a bolted joint exhibits revolute-like coaxial cylinders,
so the more-constraining interpretation must get the first claim.

The plane-contact rules share one piece of screw-theory-flavoured reasoning. Each
face-to-face plane contact blocks translation along its (outward, see §4.2) normal.
Collect the *distinct* contact normals (deduplicated up to sign — parallel planes
constrain the same direction):

- **1 distinct normal** — translation blocked along it, leaving 2 in-plane translations
  + spin about the normal: a *planar pair*, not a joint kinechain emits. It becomes
  FIXED only with corroborating fastener evidence (below).
- **≥2 distinct normals** — at most one translation survives: the direction
  **s = n̂₁ × n̂₂ / ‖n̂₁ × n̂₂‖** perpendicular to all normals, *if* every further normal is
  also perpendicular to s. Rotations are fully blocked (any rotation would tilt at least
  one contact plane). One DOF along s ⇒ **prismatic**.
- **normals spanning 3D** (no common perpendicular) — zero DOF ⇒ **fixed** by geometry
  alone.

**FIXED** — matches when there is a *large* face-to-face planar contact (antiparallel,
overlap area ≥ threshold) **and** at least one of:

1. a **bolt-like** coaxial cylinder pair — axial overlap < diameter (aspect-ratio
   heuristic: a fastener engages over a short length relative to its diameter, whereas a
   bearing bore is long); or
2. **over-constraint**: ≥2 distinct contact normals with no common perpendicular
   direction — every translation is blocked (pocket/corner seating).

The joint frame is anchored on the dominant (largest-overlap) contact plane. The frame is
nominal — a fixed joint has no motion — but keeping it lets downstream consumers locate
the interface.

**PRISMATIC** — matches when the contact planes admit exactly one free direction: ≥2
distinct normals with a common perpendicular s. The joint axis direction is s; the axis
point is taken from the dominant contact plane (for a slide, only the direction is
physically meaningful). Note the fixed/prismatic boundary is precisely the existence of
s: a dovetail or V-rail (normals all perpendicular to the rail axis) slides; a part
seated in a pocket (normals spanning 3D) is welded. A bolted interface never reaches
this rule because the fixed rule's fastener branch claims it first.

Consistency guard: every coaxial cylinder pair in the contact must have its axis
parallel to s. A cylinder contact pins the parts to its axis line, so sliding in any
other direction would break it; a perpendicular cylinder therefore vetoes the prismatic
interpretation (the pair falls through, usually to "unjoined").

**REVOLUTE** — matches the first coaxial cylinder pair that is

1. *not* bolt-like (axial overlap ≥ one diameter: a genuine bearing-length fit), and
2. *not* shoulder-locked: no face-to-face plane pair has its normal parallel to the
   cylinder axis. A loaded shoulder plane is evidence of clamping (press-fit/bolted)
   rather than a free-running fit.

The joint axis is the cylinder axis (point + direction) verbatim.

Note the **defense in depth** on the bolted case: it is rejected as revolute *twice*
(bolt-aspect and shoulder-lock) and accepted as fixed once. Each guard alone suffices on
the test fixtures; both exist because they fail independently on different real-world
geometry (a long bolt defeats the aspect test but not the shoulder test; a countersunk
fastener with no exposed shoulder defeats the shoulder test but not the aspect test).

**No match** — the pair stays unjoined. The CLI prints a per-pair diagnostic
(`unmatched: A <-> B (n cylinder pairs, m plane pairs)`) on stderr so that ambiguous
geometry is visible to the user instead of being silently dropped — a direct consequence
of the conservatism policy (§1).

### 3.5 Kinematic tree construction

Joints induce an undirected multigraph over parts. The tree is extracted as follows:

- **Root selection:** the largest-volume part participating in at least one joint
  (volume as a proxy for "the base/ground body"), overridable by the user (`--root`).
- **Orientation:** BFS from the root; each tree edge's joint is re-oriented so `parent`
  is the BFS-parent side.
- **Loops:** edges not in the BFS spanning tree become `loop_joints`. For a four-bar
  linkage this yields three tree joints plus one loop-closure constraint, matching
  simulator expectations (MJCF `<equality><connect>`; SDF allows explicit loop joints).
  The resulting MuJoCo model has the physically correct mobility: three hinge DOF
  minus two constrained translations at the connect point leaves the four-bar's
  single effective DOF.

### 3.6 MJCF emission

- Each part body carries a mesh geom; meshes are exported per-part as **binary STL**
  (MuJoCo's STL reader rejects ASCII) tessellated from the same cached triangulation used
  for axial-extent estimation (§4.3), so geometry is meshed exactly once.
- **Units:** STEP/OCCT work in millimetres, MuJoCo in metres. All emitted lengths (mesh
  scale attribute, joint positions, anchors) are scaled by 10⁻³ at emission; the rest of
  the pipeline stays in mm, including all tolerances.
- **Joints:** revolute → `<joint type="hinge">`, prismatic → `type="slide"`, **fixed → no
  joint element at all** — in MJCF a child body with no joint is rigidly welded to its
  parent, which is exactly the fixed-joint semantics.
- **Inertia:** exact, from B-Rep volume integration (`BRepGProp.VolumeProperties` at a
  default density of 1000 kg/m³): mass, centre of mass, and the full COM-referenced
  inertia tensor (OCCT's `MatrixOfInertia` is already relative to the centre of mass —
  no parallel-axis transfer is needed, and adding one produces tensors that MuJoCo
  rejects as violating the triangle inequality). MJCF gets `pos` = COM and
  `fullinertia`; SDF gets an inertial `<pose>` at the COM and the six `<inertia>`
  components. Units: the mm⁵ moments scale by density × 10⁻¹⁵ to kg·m².
- Loop joints → `<equality><connect>` between the two bodies, anchored at the joint's
  axis point.
- The root body is welded to the world (no free joint), matching the "grounded mechanism"
  interpretation; a floating base can be added manually.

### 3.7 SDF emission

The SDF emitter consumes the identical `KinematicTree` and shares mesh export, naming,
unit scaling, and mass computation with the MJCF emitter (factored into
`export_common.py`), so the two outputs are guaranteed consistent. The structural
differences are instructive about the two formats:

- **Flat links vs. body tree.** SDF links are siblings inside `<model>`; joints
  reference parent and child by name. No tree recursion is needed, and the spanning
  tree's *orientation* matters only for the parent/child labels.
- **Fixed joints are explicit** (`<joint type="fixed">`), where MJCF expresses them by
  body nesting with no joint element.
- **Loops are first-class.** Loop-closure joints are emitted as ordinary `<joint>`
  elements — SDF permits kinematic graphs, not just trees — whereas MJCF demotes them to
  `<equality><connect>` constraints. The tree/loop split computed by `graph.py` is thus
  load-bearing for MJCF and merely cosmetic for SDF.
- **Grounding.** The root link is welded to the world via an explicit fixed joint with
  parent `world` (an SDF-blessed pseudo-link), mirroring MJCF's world-attached root.
- **Frames.** SDF joint poses are resolved relative to the child link frame. Because
  every emitted link sits at the world origin (meshes are already in world
  coordinates), world-frame joint coordinates can be written directly.
- Joint names are `parent__child`: with loops, a child name alone is not unique (the
  loop child already has a tree joint), but at most one joint exists per part pair.

## 4. Implementation details

Stack: Python ≥3.10, OCCT 7.8 via the **`OCP`** bindings (shipped as `cadquery-ocp`
manylinux wheels — chosen over `pythonocc-core` because it needs no system packages),
`numpy` for vector algebra, `networkx` for graph utilities. ~700 lines of source.

### 4.1 STEP loading with hierarchy (io_step.py)

`STEPCAFControl_Reader` (the XCAF-aware reader, *not* the flat `STEPControl_Reader`)
transfers into an XCAF document, preserving names and the assembly tree. The tree is
walked recursively: assembly labels recurse into components, composing each component's
`TopLoc_Location` with the parent's; leaf labels emit a `Part` whose shape is transformed
into world coordinates (`BRepBuilderAPI_Transform`). Part names are slash-joined paths of
label names (`hinge/bracket`), giving stable, human-readable identifiers.

### 4.2 Outward normals from face orientation (surfaces.py)

A subtle but load-bearing detail: `BRepAdaptor_Surface.Plane()` returns the *geometric*
surface, whose normal direction is arbitrary with respect to the part's material. The
B-Rep face carries the missing bit as its topological orientation flag: a `REVERSED` face
has material on the surface-normal side. kinechain flips the geometric normal for
`REVERSED` faces, yielding true **outward** normals. This is what makes the
`antiparallel` test (§3.3) meaningful — without it, "facing each other" versus "flush"
is indistinguishable, and both the fixed rule's contact test and the revolute rule's
shoulder-lock would misfire on roughly half of real faces.

### 4.3 Axial extent via shared tessellation (surfaces.py)

A cylinder's *surface* is infinite along its axis; the *face* is bounded, and the bound is
what distinguishes a bolt stub from a bearing. The face's trimmed extent is estimated by
tessellating the shape once (`BRepMesh_IncrementalMesh`, 0.5 mm deflection), projecting
every triangulation vertex onto the axis, and taking the spread (max − min projection).
UV parameter bounds serve as a fallback when no triangulation exists. OCCT caches the
triangulation on the shape, so the later STL export reuses it for free.

### 4.4 Tolerance model

All dimensional thresholds live in one frozen dataclass (`contact.Tolerances`), consumed
by both the contact stage and the joint rules: positional ε, angular ε (cos 2.5°),
radius ε, minimum axial overlap, minimum plane overlap area, and the minimum "large
plane" area for the fixed/prismatic rules. The only threshold outside it is the
dimensionless bolt aspect ratio (1.0 overlap/diameter).

Two constructions exist:

- **Absolute defaults** (1 mm positional, 0.5 mm radius, 50 mm² large-plane …), tuned
  for desktop-scale mechanisms — used when calling library functions directly.
- **Scale-relative** via `Tolerances.from_diagonal(d)`, where d is the assembly's AABB
  diagonal: linear thresholds scale as fixed fractions of d (positional/radius
  2·10⁻³ d, axial overlap 5·10⁻³ d), area thresholds as the square of a fraction
  ((5·10⁻³ d)² and (5·10⁻² d)²), and the angular tolerance stays fixed (dimensionless).
  The CLI uses this construction, so the same pipeline handles watch-scale and
  excavator-scale assemblies; a 1/100-scale hinge that the absolute defaults miss
  entirely (its 0.4 mm bore overlap falls below the 1 mm minimum) is classified
  correctly under scale-relative tolerances (`tests/test_scale.py`). One caveat
  carried by the same test: the meshing deflection passed to surface extraction must
  scale with the geometry too.

### 4.5 Complexity

For n parts with f faces each: broad phase O(n²) box tests; narrow phase
O(f²) surface-pair tests per overlapping pair, with closed-form constant-time predicates.
No spatial indexing below part level yet. The dominant cost in practice is OCCT
tessellation, which is linear in geometry size and performed once per part.

## 5. Evaluation

### 5.1 Methodology: programmatic micro-fixtures

Test assemblies are *generated* by short cadquery scripts and exported to STEP in a
session-scoped temporary directory at test time — no binary CAD artifacts in the
repository, fully hermetic runs, and each fixture's geometry is documented executable
code. Each fixture is deliberately *minimal and discriminating*: it contains exactly the
features needed to make one rule fire and the competing rules' counter-signals.

### 5.2 Fixtures and assertions

**Hinge** (`make_hinge.py`) — revolute true-positive. A 40×20×20 mm bracket with a 4 mm
through-bore; a 3.95 mm pin (clearance fit, radius difference within ε_r) that protrudes
past both bracket faces, so *no* planar contact exists and only the revolute rule can
fire. Asserts: exactly one joint; type revolute; axis parallel to world X (|axis · x̂| >
0.999); bracket (larger volume) chosen as root; emitted MJCF contains a hinge joint and
both STL meshes.

**Bolted plates** (`make_bolted_plates.py`) — revolute false-positive guard / fixed
true-positive. Two 60×40×5 mm plates; the base carries four integral 3 mm studs that
engage 3.05 mm clearance holes in the top plate. The stud/hole pairs are four coaxial
cylinder pairs — superficially hinge-like — but their axial overlap (5 mm) is below their
diameter (6.1 mm) and the plates share a ~2280 mm² face-to-face contact whose normal is
parallel to every stud axis, so both revolute guards reject and the fixed rule accepts.
Asserts: contact detection finds exactly 4 coaxial cylinder pairs and an antiparallel
plane pair; exactly one joint, type fixed; base (larger volume) is root; emitted MJCF
contains *no* joint element and nests the top body inside the base body.

**Four-bar linkage** (`make_fourbar.py`) — loop-closure test. Four bar links (ground,
crank, coupler, rocker) in three Z-layers with 2 mm air gaps between layers, joined by
four integral-pin revolute joints with parallel Z axes at the loop corners. The layer
gaps guarantee *no* planar contact anywhere — each joint is evidenced purely by one
pin/hole coaxial pair (5 mm engagement vs. 4 mm pin diameter, clearing the bolt-aspect
guard) — and same-layer links (crank, rocker) are separated in X so the broad phase
rejects them. Flush pin-tip/boss-top faces additionally regression-test the
antiparallel filter (they must not register as shoulders). Asserts: exactly four
revolute joints, all axes ∥ ẑ, every part participating in exactly two joints (the
loop signature); spanning tree of three joints + one loop joint; ground (largest
volume) as root; MJCF with three hinge joints and an `<equality><connect>` element.

**V-rail slider** (`make_slider.py`) — prismatic true-positive / fixed false-positive
guard. An 80 mm rail with a full-length triangular ridge (45° flanks) and a shorter
40 mm carriage with the matching groove. The contact comprises three distinct
face-to-face plane orientations — the two flanks (normals (0, ±√2⁄2, √2⁄2) on the rail
side) and the flat seat (normal ±ẑ) — all perpendicular to x̂, so the slide direction
resolves to the rail axis. No cylinders exist, isolating the plane-only path; the
carriage is shorter than the rail so no x̂-normal contact blocks the slide. The
flush (parallel-same-direction) carriage/rail side faces also exercise the
antiparallel filter. Asserts: no coaxial cylinders; ≥2 face-to-face plane pairs with
non-parallel normals; exactly one joint, type prismatic, axis parallel to world X;
rail (larger volume) is root; emitted MJCF contains a `slide` joint.

### 5.3 Round-trip validation

Every fixture is additionally loaded into MuJoCo (`MjModel.from_xml_path`), asserting
the compiled model's joint count and joint types (hinge: `njnt == 1`,
`jnt_type == mjJNT_HINGE`; bolted: `njnt == 0`, `nbody == 3`; slider: `njnt == 1`,
`jnt_type == mjJNT_SLIDE`; four-bar: `njnt == 3` hinges plus `neq == 1` connect
constraint). This catches emission
errors that XML-level assertions miss (bad mesh references, scale errors, malformed
inertials). These tests skip gracefully when MuJoCo is not installed. A manual
interactive check (passive viewer, dragging the hinge leaf) complements the automated
suite.

### 5.4 Unit-level tests

Two pieces of machinery are additionally tested below the fixture level. The projected
overlap area is verified on synthetic square faces (full/half/quarter overlap,
containment, and the laterally disjoint coplanar case that motivated the exact
computation). The classification rules are exercised on hand-built contact-feature
structures with no CAD involved at all — this pins down rule *boundaries* cheaply:
prismatic acceptance, the cylinder-axis veto (and its interaction with the revolute
shoulder lock, leaving the pair unjoined), normals-spanning-3D → fixed,
plane+bolt → fixed, a lone contact plane → unjoined, and sliver planes below the area
threshold → ignored. The scale-relative tolerance behaviour is tested by shrinking the
hinge fixture 100× (see §4.4).

Inertial properties are validated against closed-form solutions: an off-center box and
a cylinder must reproduce the textbook tensors (the off-center case proves the tensor
is COM-referenced — the placement offset must not leak in), and a MuJoCo round trip
checks that the compiled model carries the analytic mass and principal moments of the
hinge pin to within the XML's 6-significant-digit serialization.

The SDF emitter is validated structurally (no Gazebo runtime assumed): every fixture's
SDF is parsed back and checked for the format's semantics — flat links with
visual/collision mesh geometry and positive mass, a single world-anchor joint on the
root, the bolted fixed joint present as an explicit element with no axis, the slider's
prismatic axis along X, and the four-bar's loop edge present as a fourth ordinary
revolute joint with a unique `parent__child` name. The CLI `--format sdf` path is
exercised end-to-end as well.

## 6. Discussion and limitations

- **Geometry under-determines intent** (§1). The rules encode *manufacturing
  conventions* (fasteners are short relative to diameter; bearings are long; mating faces
  clamp), not physics. A press-fit pin of bearing-like proportions will be classified
  revolute; nothing in the geometry distinguishes it. The conservative-default policy
  bounds the damage: ambiguity yields "unjoined + diagnostic", never a wrong weld.
- **Contact areas are exact only up to tessellation**, and the projection ignores the
  (tolerance-bounded) angle between near-parallel faces — both effects are second-order
  at the default 2.5° angular tolerance.
- **Limits, friction, damping, actuators** are not inferred; joint *limits* in particular
  would require reasoning about stop features.
- **Mesh-only inputs (STL)** are out of scope by design; the `surfaces.py` interface is
  the seam where a segmentation/primitive-fitting front end could substitute.

## 7. Status and roadmap

Implemented and tested end-to-end: STEP→**MJCF and SDF** with **revolute**,
**prismatic**, and **fixed** classification, root selection,
spanning-tree/**loop-closure** split (exercised through MuJoCo by the four-bar
fixture), per-pair diagnostics, a `--format` CLI switch, **exact projected contact
areas**, **scale-relative tolerances** (used by the CLI), and the **cylinder-axis
consistency guard** on the prismatic rule, and **exact inertial properties** (mass,
COM, full tensor) validated against closed-form solutions.

Next: Gazebo round-trip validation of the SDF output (structural checks only today);
spherical/planar/universal joint rules; joint-limit inference from stop features;
a user-settable density (currently fixed at 1000 kg/m³).
