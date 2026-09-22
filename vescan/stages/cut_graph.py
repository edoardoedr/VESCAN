#!/usr/bin/env python3
"""
Cuts a centerline tree after a fixed number of bifurcations from the root,
working entirely from vescan.stages.build_graph's ALREADY-SAVED
outputs - no VMTK/branch-extractor re-run needed:
  - 05_branch_tree_topology.json (--graph-output): topology - per-group
    generation/blanked and the parent->child GroupId edges. Has no point
    positions.
  - the branch tree file (--split-output, 05_branch_tree.vtk): geometry -
    one cell per group (or per candidate cell, see
    vescan.stages.build_graph's module docstring) with GroupId/
    IsBifurcation/Generation/Length/AverageRadius cell arrays and a
    per-point Radius array. Has no adjacency information. Positional
    argument name below is still `combined_model` for historical reasons
    (it used to be a separate, since-removed combined-model file) - any
    file with those cell arrays works, which the branch tree file now is.
Together they're everything needed to cut: the topology JSON decides which
GroupIds survive and where the tree gets severed, the branch tree file
supplies the actual coordinates for the kept geometry and for the cut point
positions (the last point of the kept group immediately upstream of a
dropped one).

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.cut_graph 05_branch_tree_topology.json 05_branch_tree.vtk \\
        06_cut_centerline.vtk --max-generations 3 --cut-points-output 06_cut_points.mrk.json
"""

import argparse
import json
import logging
import sys
import time

import vtk

from vescan.io import load_surface, save_surface, detect_coordinate_space, Stage

logger = logging.getLogger(__name__)


def load_graph_json(graph_json_path):
    """Reads a vescan.stages.build_graph --graph-output file.
    Returns (nodes, edges, roots): nodes is {group_id(int): node dict},
    edges is a list of {"parent": group_id, "child": group_id}."""
    with open(graph_json_path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = {int(groupId): node for groupId, node in data["nodes"].items()}
    return nodes, data["edges"], data["roots"]


def _best_cell_per_group(polyData, groupIdArray, lengthArray):
    """Maps GroupId -> its longest cell index in polyData (a combined model
    can have more than one cell for the same GroupId - see
    vescan.stages.build_graph's build_group_instances()/module
    docstring). Used to pick a single representative cell per group for
    cut-point endpoint lookup."""
    best = {}
    for cellId in range(polyData.GetNumberOfCells()):
        groupId = int(groupIdArray.GetValue(cellId))
        length = lengthArray.GetValue(cellId) if lengthArray else 0.0
        if groupId not in best or length > best[groupId][1]:
            best[groupId] = (cellId, length)
    return {groupId: cellId for groupId, (cellId, _length) in best.items()}


def _extract_cells(polyData, cellIds):
    """Builds a new vtkPolyData containing just the given cells (as
    polylines), copying every existing point-data and cell-data array by
    name - generic over whatever arrays the combined model happens to
    carry (Radius per point; GroupId/IsBifurcation/Generation/Length/
    AverageRadius per cell), instead of hardcoding that list.

    Shared with vescan.stages.build_graph (save_cleaned_split()) and
    vescan.stages.clip_vessel (build_dropped_centerline())."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pointArrays = [(polyData.GetPointData().GetArray(i), _new_matching_array(polyData.GetPointData().GetArray(i)))
                   for i in range(polyData.GetPointData().GetNumberOfArrays())]
    cellArrays = [(polyData.GetCellData().GetArray(i), _new_matching_array(polyData.GetCellData().GetArray(i)))
                  for i in range(polyData.GetCellData().GetNumberOfArrays())]

    for cellId in cellIds:
        cellPointIds = polyData.GetCell(cellId).GetPointIds()
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(cellPointIds.GetNumberOfIds())
        for i in range(cellPointIds.GetNumberOfIds()):
            srcPointId = cellPointIds.GetId(i)
            newPointId = points.InsertNextPoint(polyData.GetPoint(srcPointId))
            polyLine.GetPointIds().SetId(i, newPointId)
            for src, dst in pointArrays:
                dst.InsertNextTuple(src.GetTuple(srcPointId))
        lines.InsertNextCell(polyLine)
        for src, dst in cellArrays:
            dst.InsertNextTuple(src.GetTuple(cellId))

    outputPolyData = vtk.vtkPolyData()
    outputPolyData.SetPoints(points)
    outputPolyData.SetLines(lines)
    for _src, dst in pointArrays:
        outputPolyData.GetPointData().AddArray(dst)
    for _src, dst in cellArrays:
        outputPolyData.GetCellData().AddArray(dst)
    return outputPolyData


def _new_matching_array(srcArray):
    newArray = srcArray.NewInstance()
    newArray.SetName(srcArray.GetName())
    newArray.SetNumberOfComponents(srcArray.GetNumberOfComponents())
    return newArray


def find_cut_points(nodes, edges, bestCellForGroup, polyData, max_generations, keptGroupIds):
    """For every edge whose parent survived the cutoff and whose child did
    not, returns the position where that cut happened - the parent's own
    last point (its representative cell's final point, where the dropped
    child would have continued from)."""
    cutPoints = []
    for edge in edges:
        parentId, childId = edge["parent"], edge["child"]
        if parentId not in keptGroupIds or childId in keptGroupIds:
            continue
        if parentId not in bestCellForGroup:
            continue  # parent group not present in this combined model (e.g. filtered by min-branch-length)
        cellPointIds = polyData.GetCell(bestCellForGroup[parentId]).GetPointIds()
        lastPointId = cellPointIds.GetId(cellPointIds.GetNumberOfIds() - 1)
        cutPoints.append({
            "parent_group_id": parentId,
            "dropped_group_id": childId,
            "position": polyData.GetPoint(lastPointId),
        })
    return cutPoints


def save_cut_points_markups(cutPoints, output_path, coordinate_space="LPS"):
    """Writes a Markups fiducial .mrk.json with one point per cut, labeled
    with which kept group it's on and which dropped group used to continue
    from there."""
    controlPoints = []
    for cut in cutPoints:
        controlPoints.append({
            "id": str(len(controlPoints) + 1),
            "label": f"cut_group{cut['parent_group_id']}_dropped_group{cut['dropped_group_id']}",
            "description": "",
            "associatedNodeID": "",
            "position": list(cut["position"]),
            "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        })
    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": coordinate_space,
            "coordinateUnits": "mm",
            "locked": False,
            "labelFormat": "%N-%d",
            "name": "cut_points",
            "controlPoints": controlPoints,
        }],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return len(controlPoints)


def run(graph_json_path, combined_model_path, output_centerline_path, max_generations,
        cut_points_output_path=None):
    pipelineStart = time.monotonic()
    nodes, edges, roots = load_graph_json(graph_json_path)

    maxAvailableGeneration = max((node["generation"] for node in nodes.values() if node["generation"] >= 0),
                                  default=0)
    if max_generations > maxAvailableGeneration:
        logger.info("--max-generations %d is >= the tree's actual max generation (%d) - nothing will be cut.",
                    max_generations, maxAvailableGeneration)

    keptGroupIds = {groupId for groupId, node in nodes.items() if 0 <= node["generation"] <= max_generations}
    if not keptGroupIds:
        raise ValueError(f"No group survives generation <= {max_generations} (root is generation 0) - "
                          f"check --max-generations against 05_branch_tree_topology.json's node generations")

    coordinateSpace = detect_coordinate_space(combined_model_path) or "LPS"
    with Stage("Loading combined model"):
        polyData = load_surface(combined_model_path)
    groupIdArray = polyData.GetCellData().GetArray("GroupId")
    lengthArray = polyData.GetCellData().GetArray("Length")
    if groupIdArray is None:
        raise ValueError(f"{combined_model_path} has no 'GroupId' cell array - pass the file saved via "
                          f"vescan.stages.build_graph's --split-output, not a per-branch/bifurcation model")

    bestCellForGroup = _best_cell_per_group(polyData, groupIdArray, lengthArray)
    keptCellIds = [cellId for cellId in range(polyData.GetNumberOfCells())
                   if int(groupIdArray.GetValue(cellId)) in keptGroupIds]
    if not keptCellIds:
        raise ValueError("None of the kept GroupIds from 05_branch_tree_topology.json were found in the "
                          "combined model's 'GroupId' cell array - are they from the same build_graph run?")

    with Stage(f"Extracting {len(keptCellIds)} kept cell(s)"):
        outputPolyData = _extract_cells(polyData, keptCellIds)
    save_surface(outputPolyData, output_centerline_path, coordinate_space=coordinateSpace)
    isBifurcationArray = outputPolyData.GetCellData().GetArray("IsBifurcation")
    nBifurcations = sum(1 for i in range(outputPolyData.GetNumberOfCells()) if isBifurcationArray.GetValue(i)) \
        if isBifurcationArray else 0
    logger.info("Saved cut centerline (generation <= %d: %d group(s), %d cell(s), %d bifurcation(s), out of "
                "%d total groups) to %s", max_generations, len(keptGroupIds), outputPolyData.GetNumberOfCells(),
                nBifurcations, len(nodes), output_centerline_path)

    cutPoints = find_cut_points(nodes, edges, bestCellForGroup, polyData, max_generations, keptGroupIds)
    if cut_points_output_path:
        n = save_cut_points_markups(cutPoints, cut_points_output_path, coordinate_space=coordinateSpace)
        logger.info("Saved %d cut point(s) to %s", n, cut_points_output_path)
    else:
        logger.info("%d cut point(s) found (pass --cut-points-output to save them)", len(cutPoints))

    logger.info("Total time: %.1fs", time.monotonic() - pipelineStart)
    return outputPolyData, cutPoints


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph_json", help="Path to the 05_branch_tree_topology.json saved by build_graph.py's "
                                            "--graph-output (topology: per-group generation, parent->child edges)")
    parser.add_argument("combined_model", help="Path to the branch tree file saved by build_graph.py's "
                                                "--split-output (05_branch_tree.vtk) - geometry: one cell per "
                                                "group/candidate cell with GroupId/IsBifurcation/Generation/"
                                                "Length/AverageRadius arrays")
    parser.add_argument("output_centerline", help="Path to save the cut centerline: a single combined .vtk/.vtp "
                                                    "containing only the kept groups' cells, same cell arrays as "
                                                    "the input combined model")
    parser.add_argument("--max-generations", type=int, required=True,
                         help="Keep groups up to this many bifurcations crossed from the root (0 = trunk only, "
                              "no bifurcation crossed); drop everything past it")
    parser.add_argument("--cut-points-output", default=None,
                         help="Optional path to save a Markups fiducial .mrk.json marking every point where a "
                              "branch got truncated (the last point of the last kept group before a dropped one)")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.graph_json,
        args.combined_model,
        args.output_centerline,
        args.max_generations,
        cut_points_output_path=args.cut_points_output,
    )


if __name__ == "__main__":
    sys.exit(main())
