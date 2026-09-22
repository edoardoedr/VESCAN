#!/usr/bin/env python3
"""
Standalone network extraction for vascular/airway surfaces.

Replicates ExtractCenterlineLogic.extractNetwork() with computeGeometry=True
(SlicerExtension-VMTK/ExtractCenterline/ExtractCenterline.py) - the branch
used by the "Apply" button to populate the "NetworkModel" output, as opposed
to the simpler computeGeometry=False call used only internally by the
auto-detect-endpoints flow (already replicated in
vescan.stages.endpoints). Runs without requiring 3D Slicer.

Input: a preprocessed surface (see vescan.stages.preprocess) and a
Slicer Markups fiducial file with endpoints (see
vescan.stages.endpoints, or one placed manually in Slicer and
exported). Output: a .vtk/.vtp polydata with per-point Radius/Topology/Marks
arrays and, if computeGeometry is enabled, per-point Length/Curvature/
Torsion/Tortuosity/FrenetTangent/FrenetNormal/FrenetBinormal arrays - the
same content as Slicer's "NetworkModel" node. Optionally also writes a
Slicer Markups .mrk.json with one open curve per network branch (cell),
mirroring ExtractCenterlineLogic.addNetworkCurves() - the same decomposition
Slicer uses to populate the "NetworkCurve" node (minus the per-point Radius
array, which has no slot in the plain mrk.json schema).

Works both as a standalone VMTK build (`from vmtk import ...`) and inside the
3D Slicer Python console/environment (bare `import vtkvmtk*Python`), like
vescan.stages.endpoints.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.network preprocessed.vtk endpoints.mrk.json network.vtp
    python -m vescan.stages.network preprocessed.vtk endpoints.mrk.json network.vtp --no-geometry
    python -m vescan.stages.network preprocessed.vtk endpoints.mrk.json network.vtp --network-curve network_curve.mrk.json

Usage (from Python - main.py, a notebook, or the Slicer Python console):
    import sys
    sys.path.append("/path/to/vmtk_building")
    from vescan.stages import network
    network.run(
        "preprocessed.vtk",
        "endpoints.mrk.json",
        "network.vtp",
        output_curve_path="network_curve.mrk.json",
    )
"""

import argparse
import json
import logging
import sys

import vtk

try:
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
    import vtkvmtkMiscPython as vtkvmtkMisc
except ImportError:
    # Standalone VMTK build: compiled modules live inside the `vmtk` package.
    from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
    from vmtk import vtkvmtkMiscPython as vtkvmtkMisc

from vescan.io import (load_surface, save_surface, detect_coordinate_space, flip_positions_lps_ras,
                               sanitize_nan_arrays)

logger = logging.getLogger(__name__)

CELL_ID_ARRAY_NAME = "CellId"

RADIUS_ARRAY_NAME = "Radius"
TOPOLOGY_ARRAY_NAME = "Topology"
MARKS_ARRAY_NAME = "Marks"
LENGTH_ARRAY_NAME = "Length"
CURVATURE_ARRAY_NAME = "Curvature"
TORSION_ARRAY_NAME = "Torsion"
TORTUOSITY_ARRAY_NAME = "Tortuosity"
FRENET_TANGENT_ARRAY_NAME = "FrenetTangent"
FRENET_NORMAL_ARRAY_NAME = "FrenetNormal"
FRENET_BINORMAL_ARRAY_NAME = "FrenetBinormal"

# vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio (same default VMTK
# itself uses) - see vescan.stages.endpoints' own copy of this same
# parameter.
NETWORK_ADVANCEMENT_RATIO = 1.05


def load_endpoints_markups(markups_path):
    """Reads a Slicer Markups fiducial .mrk.json file.
    Returns (endpoints, coordinateSystem) where endpoints is a list of
    (position, selected) tuples in file order (position is [x, y, z])."""
    with open(markups_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    markup = data["markups"][0]
    coordinateSystem = markup.get("coordinateSystem", "LPS")
    endpoints = [(list(cp["position"]), bool(cp.get("selected", True)))
                 for cp in markup["controlPoints"]]
    return endpoints, coordinateSystem


def start_point_index(endpoints):
    """Mirrors ExtractCenterlineLogic.startPointIndexFromEndPointsMarkupsNode():
    the first unselected control point is the start point; if all points are
    selected (or the list is empty), the first point is used / -1 if empty."""
    if not endpoints:
        return -1
    for i, (_position, selected) in enumerate(endpoints):
        if not selected:
            return i
    return 0


def open_surface_at_point(polyData, holePosition):
    """Mirrors ExtractCenterlineLogic.openSurfaceAtPoint(). Modifies polyData in place."""
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(polyData)
    pointLocator.BuildLocator()
    holePointIndex = pointLocator.FindClosestPoint(holePosition)

    if holePointIndex < 0:
        raise ValueError("openSurfaceAtPoint failed: empty input polydata")

    polyData.BuildLinks()
    cellIds = vtk.vtkIdList()
    polyData.GetPointCells(holePointIndex, cellIds)
    if cellIds.GetNumberOfIds() > 0:
        polyData.DeleteCell(cellIds.GetId(0))
        polyData.RemoveDeletedCells()


def _add_cell_id_array(polyData, name=CELL_ID_ARRAY_NAME):
    """Adds a sequential per-cell integer array (0..numCells-1), not part of
    any Slicer/VMTK filter output. Lets the network/centerline model be
    loaded as a single Model node in Slicer and colored per-branch (Display
    > Scalars > Active Scalar = this array, Cell location, "Random" color
    table) - the same one-distinct-color-per-cellId effect
    addNetworkCurves()/addCenterlineCurves() get in Slicer via
    vtkMRMLColorTableNodeRandom, but on one node instead of one per branch."""
    cellIdArray = vtk.vtkIntArray()
    cellIdArray.SetName(name)
    numberOfCells = polyData.GetNumberOfCells()
    cellIdArray.SetNumberOfValues(numberOfCells)
    for cellId in range(numberOfCells):
        cellIdArray.SetValue(cellId, cellId)
    polyData.GetCellData().AddArray(cellIdArray)
    return polyData


def extract_network(surfacePolyData, endpoints=None, computeGeometry=True,
                     advancement_ratio=NETWORK_ADVANCEMENT_RATIO):
    """Mirrors ExtractCenterlineLogic.extractNetwork().
    endpoints: list of (position, selected) tuples in the SAME coordinate
    space as surfacePolyData, or None/empty to fall back to the
    bounds-corner heuristic (matches passing no endPointsMarkupsNode)."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfacePolyData)
    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputConnection(cleaner.GetOutputPort())
    triangleFilter.Update()
    simplifiedPolyData = triangleFilter.GetOutput()

    if endpoints:
        startIndex = start_point_index(endpoints)
        startPosition = endpoints[startIndex][0]
    else:
        bounds = [0.0] * 6
        simplifiedPolyData.GetBounds(bounds)
        startPosition = [bounds[0], bounds[2], bounds[4]]
    open_surface_at_point(simplifiedPolyData, startPosition)

    networkExtraction = vtkvmtkMisc.vtkvmtkPolyDataNetworkExtraction()
    networkExtraction.SetInputData(simplifiedPolyData)
    networkExtraction.SetAdvancementRatio(advancement_ratio)
    networkExtraction.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    networkExtraction.SetTopologyArrayName(TOPOLOGY_ARRAY_NAME)
    networkExtraction.SetMarksArrayName(MARKS_ARRAY_NAME)
    networkExtraction.Update()

    if not computeGeometry:
        return _add_cell_id_array(networkExtraction.GetOutput())

    centerlineGeometry = vtkvmtkComputationalGeometry.vtkvmtkCenterlineGeometry()
    centerlineGeometry.SetInputData(networkExtraction.GetOutput())
    centerlineGeometry.SetLengthArrayName(LENGTH_ARRAY_NAME)
    centerlineGeometry.SetCurvatureArrayName(CURVATURE_ARRAY_NAME)
    centerlineGeometry.SetTorsionArrayName(TORSION_ARRAY_NAME)
    centerlineGeometry.SetTortuosityArrayName(TORTUOSITY_ARRAY_NAME)
    centerlineGeometry.SetFrenetTangentArrayName(FRENET_TANGENT_ARRAY_NAME)
    centerlineGeometry.SetFrenetNormalArrayName(FRENET_NORMAL_ARRAY_NAME)
    centerlineGeometry.SetFrenetBinormalArrayName(FRENET_BINORMAL_ARRAY_NAME)
    centerlineGeometry.Update()
    geometryOutput = centerlineGeometry.GetOutput()
    # Curvature/Torsion/FrenetNormal/FrenetBinormal are undefined on straight
    # or near-degenerate segments and come out as NaN instead of raising -
    # see sanitize_nan_arrays()'s own docstring for why that's worth fixing
    # here rather than leaving in the saved network (Slicer/older VTK legacy
    # readers can fail to load a model with a literal "nan" in it).
    sanitize_nan_arrays(geometryOutput)
    return _add_cell_id_array(geometryOutput)


def build_curve_markups(branches, coordinate_space="LPS", base_name="branch", indices=None):
    """Builds a Slicer Markups .mrk.json dict (schema v1.0.3) with one open
    curve markup per branch, matching the vtkMRMLMarkupsCurveNode objects
    Slicer's addNetworkCurves()/addCenterlineCurves() create (one per branch).
    `branches` is a list of branches, each a list of [x, y, z] positions in
    curve traversal order. Point-wise Radius (stored in Slicer as a curve
    measurement array) has no equivalent slot in the plain mrk.json schema
    and is not included here.
    `indices`: optional list (same length as `branches`) of the number to use
    in each curve's "(N)" name suffix - pass the source cellId to match
    Slicer's addCenterlineCurves()/addNetworkCurves() naming (curveNode name
    is "{baseName} ({cellId})", the same cellId as the CenterlineProperties/
    NetworkProperties table's CellId column), instead of the default
    enumeration order."""
    if indices is None:
        indices = range(len(branches))
    markups = []
    for branchIndex, positions in zip(indices, branches):
        controlPoints = []
        for i, position in enumerate(positions):
            controlPoints.append({
                "id": str(i + 1),
                "label": "",
                "description": "",
                "associatedNodeID": "",
                "position": list(position),
                "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            })
        markups.append({
            "type": "Curve",
            "coordinateSystem": coordinate_space,
            "coordinateUnits": "mm",
            "locked": False,
            "labelFormat": "%N-%d",
            "name": f"{base_name} ({branchIndex})",
            "controlPoints": controlPoints,
        })
    return {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups,
    }


def build_network_curve_markups(networkPolyData, coordinate_space="LPS", base_name="branch"):
    """Mirrors ExtractCenterlineLogic.addNetworkCurves(): decomposes networkPolyData
    into one open curve per cell (branch), with the cell's points as control
    points in cell order - the same per-cell split Slicer uses to populate one
    vtkMRMLMarkupsCurveNode per branch under the "NetworkCurve" node."""
    branches = []
    for cellId in range(networkPolyData.GetNumberOfCells()):
        pointIds = networkPolyData.GetCell(cellId).GetPointIds()
        branches.append([list(networkPolyData.GetPoint(pointIds.GetId(i)))
                          for i in range(pointIds.GetNumberOfIds())])
    return build_curve_markups(branches, coordinate_space=coordinate_space, base_name=base_name)


def save_network_curve_markups(networkPolyData, output_path, coordinate_space="LPS", base_name="branch"):
    """Writes the dict from build_network_curve_markups() to a .mrk.json file."""
    data = build_network_curve_markups(networkPolyData, coordinate_space=coordinate_space, base_name=base_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(surface_path, endpoints_path, output_path, output_curve_path=None, computeGeometry=True,
        advancement_ratio=NETWORK_ADVANCEMENT_RATIO):
    """Extracts the network from surface_path (using endpoints_path's start
    point) and writes it to output_path, plus a per-branch curve markups file
    at output_curve_path if given. Returns the network polydata."""
    # The surface and the endpoints must be expressed in the same coordinate
    # space for FindClosestPoint()/the bounds-corner heuristic to locate the
    # correct hole position on the surface (see the LPS/RAS discussion in
    # vescan.stages.endpoints - the same mechanism applies here).
    surfaceSpace = detect_coordinate_space(surface_path) or "LPS"
    surfacePolyData = load_surface(surface_path)

    endpoints, endpointsSpace = load_endpoints_markups(endpoints_path)
    if endpointsSpace != surfaceSpace:
        logger.info("Endpoints are tagged %s, surface is %s - converting endpoints to match the "
                    "surface's coordinate space.", endpointsSpace, surfaceSpace)
        flippedPositions = flip_positions_lps_ras([position for position, _selected in endpoints])
        endpoints = list(zip(flippedPositions, [selected for _position, selected in endpoints]))

    networkPolyData = extract_network(surfacePolyData, endpoints, computeGeometry=computeGeometry,
                                       advancement_ratio=advancement_ratio)

    save_surface(networkPolyData, output_path, coordinate_space=surfaceSpace)
    logger.info("Saved network (%d points, %d cells) to %s",
                networkPolyData.GetNumberOfPoints(), networkPolyData.GetNumberOfCells(), output_path)

    if output_curve_path:
        save_network_curve_markups(networkPolyData, output_curve_path, coordinate_space=surfaceSpace)
        logger.info("Saved network curve (%d branches) to %s",
                    networkPolyData.GetNumberOfCells(), output_curve_path)

    return networkPolyData


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_surface", help="Path to the preprocessed input surface (.vtk or .vtp)")
    parser.add_argument("endpoints_markups", help="Path to the endpoints Slicer Markups fiducial file (.mrk.json)")
    parser.add_argument("output_network", help="Path to the output network polydata (.vtk or .vtp)")
    parser.add_argument("--network-curve", dest="output_network_curve", default=None,
                         help="Optional path to also write the network as a Slicer Markups .mrk.json "
                              "with one open curve per branch (.mrk.json), matching Slicer's 'NetworkCurve' "
                              "output (see addNetworkCurves()). Omit to skip.")
    parser.add_argument("--no-geometry", action="store_true",
                         help="Skip vtkvmtkCenterlineGeometry (Length/Curvature/Torsion/Tortuosity/Frenet* "
                              "arrays), matching extractNetwork(computeGeometry=False) - the variant Slicer "
                              "uses internally for auto-detect-endpoints rather than for the NetworkModel output.")
    parser.add_argument("--advancement-ratio", type=float, default=NETWORK_ADVANCEMENT_RATIO, metavar="F",
                         help="vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio parameter "
                              "(default: %(default)s).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.input_surface,
        args.endpoints_markups,
        args.output_network,
        output_curve_path=args.output_network_curve,
        computeGeometry=not args.no_geometry,
        advancement_ratio=args.advancement_ratio,
    )


if __name__ == "__main__":
    sys.exit(main())