"""Build a kinematic tree from a list of joints.

For the hinge slice the graph is trivially a single edge between two parts — no cycles
to break. The interface is shaped so adding loop-closure handling later is a small change
(detect cycles with `nx.cycle_basis`, demote one edge per cycle to a `loop_joints` list).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .io_step import AssemblyModel
from .joints import Joint


@dataclass
class KinematicTree:
    root: int                          # part index of the world-anchored body
    parts: list                        # AssemblyModel.parts, kept for downstream consumers
    joints: list[Joint]                # tree edges, oriented parent -> child
    loop_joints: list[Joint] = field(default_factory=list)  # cycle-closure constraints


def _volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()  # for VolumeProperties this is the volume


def _pick_root(parts, joints: list[Joint]) -> int:
    """Default root: the part with the largest volume that participates in at least one joint.

    Falls back to part 0 if no joints exist.
    """
    if not joints:
        return 0
    touched = {j.parent for j in joints} | {j.child for j in joints}
    return max(touched, key=lambda i: _volume(parts[i].shape))


def build_tree(model: AssemblyModel, joints: list[Joint], root: int | None = None) -> KinematicTree:
    g = nx.Graph()
    g.add_nodes_from(range(len(model.parts)))
    for idx, j in enumerate(joints):
        g.add_edge(j.parent, j.child, joint_idx=idx)

    if root is None:
        root = _pick_root(model.parts, joints)

    tree_joints: list[Joint] = []
    loop_joints: list[Joint] = []

    # Pull a spanning tree by BFS from root, orient edges parent→child along BFS order.
    # Any joint whose edge is not in the spanning tree becomes a loop-closure joint.
    if g.number_of_edges() == 0:
        return KinematicTree(root=root, parts=model.parts, joints=[], loop_joints=[])

    visited = {root}
    parent_of: dict[int, int] = {}
    queue = [root]
    spanning_edges: set[tuple[int, int]] = set()
    while queue:
        u = queue.pop(0)
        for v in g.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            parent_of[v] = u
            spanning_edges.add(tuple(sorted((u, v))))
            queue.append(v)

    for j in joints:
        edge = tuple(sorted((j.parent, j.child)))
        if edge not in spanning_edges:
            loop_joints.append(j)
            continue
        # Re-orient joint so .parent is the BFS-parent direction.
        if parent_of.get(j.child) == j.parent:
            tree_joints.append(j)
        else:
            tree_joints.append(
                Joint(
                    type=j.type,
                    parent=j.child,
                    child=j.parent,
                    axis_point=j.axis_point,
                    axis_dir=j.axis_dir,
                )
            )
    return KinematicTree(root=root, parts=model.parts, joints=tree_joints, loop_joints=loop_joints)
