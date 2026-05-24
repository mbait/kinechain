"""STEP loader: reads an assembly file into a flat list of named Parts with world transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_DocumentTool


@dataclass(frozen=True)
class Part:
    name: str
    shape: TopoDS_Shape  # already transformed into world coordinates


@dataclass(frozen=True)
class AssemblyModel:
    parts: list[Part]


def _label_name(label: TDF_Label) -> str | None:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return None


def _apply_location(shape: TopoDS_Shape, loc: TopLoc_Location) -> TopoDS_Shape:
    if loc.IsIdentity():
        return shape
    return BRepBuilderAPI_Transform(shape, loc.Transformation(), True).Shape()


def load_step(path: str | Path) -> AssemblyModel:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    doc = TDocStd_Document(TCollection_ExtendedString("kinechain"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)

    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {status}")
    if not reader.Transfer(doc):
        raise RuntimeError("STEP transfer to XCAF document failed")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free_shapes = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_shapes)

    parts: list[Part] = []
    for i in range(1, free_shapes.Length() + 1):
        _collect(shape_tool, free_shapes.Value(i), TopLoc_Location(), "", parts)
    return AssemblyModel(parts=parts)


def _collect(
    shape_tool,
    label: TDF_Label,
    parent_loc: TopLoc_Location,
    name_prefix: str,
    out: list[Part],
) -> None:
    """Recursively walk the XCAF assembly tree, emitting transformed leaf shapes."""
    own_name = _label_name(label) or "unnamed"
    full_name = f"{name_prefix}/{own_name}" if name_prefix else own_name

    if shape_tool.IsAssembly_s(label):
        comps = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, comps)
        for i in range(1, comps.Length() + 1):
            comp = comps.Value(i)
            comp_loc = shape_tool.GetLocation_s(comp)
            combined = parent_loc.Multiplied(comp_loc)
            referred = TDF_Label()
            if shape_tool.GetReferredShape_s(comp, referred):
                _collect(shape_tool, referred, combined, full_name, out)
        return

    # Leaf: a free shape that isn't an assembly
    shape = shape_tool.GetShape_s(label)
    if shape.IsNull():
        return
    out.append(Part(name=full_name, shape=_apply_location(shape, parent_loc)))
