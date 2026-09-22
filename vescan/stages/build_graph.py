#!/usr/bin/env python3
"""
Builds a bifurcation graph/tree from a centerline model (.vtk/.vtp - the raw
CenterlineModel produced by vescan.stages.centerline) using VMTK's own
authoritative branch topology (vtkvmtkCenterlineBranchExtractor +
vtkvmtkCenterlineUtilities) - the same approach SlicerExtension-VMTK's
CenterlineDisassembly module uses (see
SlicerExtension-VMTK/CenterlineDisassembly/CenterlineDisassembly.py) - and
writes it out in forms 3D Slicer can load directly.

An earlier version of this script built the graph by clustering branch
endpoints within a spatial --tolerance, because vtkvmtkMergeCenterlines'
output loses point-ID continuity between adjacent branches (each cell's
points are inserted independently - see vtkvmtkMergeCenterlines.cxx). That
worked but left ~20% of branches "unreachable" (spurious parallel/duplicate
branches from dense bifurcation clusters) and had no reliable way to pick
the true tree root.

Saved outputs, once per run (each optional, enabled via its own CLI flag):
  - output_dir/Branch_Model_*.vtk / Bifurcation_Model_*.vtk: one model per
    branch/bifurcation candidate cell - the direct Slicer-facing output,
    matching CenterlineDisassembly's one-node-per-branch scene.
  - --split-output (05_branch_tree.vtk): a SINGLE file that is both (a) the
    branch-split centerline with degenerate/short groups (below
    --min-branch-length) physically removed - unlike the raw
    vtkvmtkCenterlineBranchExtractor output, which can have hundreds of
    near-zero-length stub tracts (see repair_orphan_roots()'s docstring)
    that show up as isolated point-like "centerlines" in Slicer - AND (b)
    a per-group geometry+topology summary: on top of the original
    GroupIds/CenterlineIds/TractIds/Blanking/Radius arrays (still there,
    so this file also works as ensure_split()'s cache and as
    clip_vessel.py's split_centerline input), it carries friendly
    GroupId/IsBifurcation/Generation/Length/AverageRadius cell arrays (see
    add_group_metadata_arrays()) - previously these two concerns were
    split across three separate, largely-overlapping files (a raw split,
    a "cleaned" split, and a "combined model"); this is that same
    information compacted into one.
  - --curve-output / --bifurcation-output / --graph-output: as before (see
    below).

This version reads VMTK's own topology bookkeeping directly:
  - vtkvmtkCenterlineBranchExtractor splits the raw overlapping source->
    target paths into per-branch/per-bifurcation "tracts", tagging each
    with GroupIds (which anatomical branch/bifurcation it belongs to),
    CenterlineIds (which source->target path it came from) and TractIds
    (its order along that path). This is the SAME expensive filter (same
    array names) vescan.stages.centerline's own split_centerline()
    runs when building CenterlineCurve/CenterlineProperties - ensure_split()
    below skips re-running it when fed an already-split file (see
    run()/main.py's own wiring, which passes centerline.run()'s
    split_output_path straight in here instead of recomputing it).
  - vtkvmtkCenterlineUtilities.FindAdjacentCenterlineGroupIds() finds, for
    a given GroupId, the upstream/downstream neighboring GroupIds by
    matching TractIds +/-1 on the SAME CenterlineId - genuine topological
    adjacency, not a spatial guess. (Only works on the branch-extractor's
    raw output - on a vtkvmtkMergeCenterlines'd file every cell keeps only
    one arbitrary CenterlineId/TractId value, so this lookup finds nothing;
    confirmed empirically before writing this version - this is also why
    centerline.py's MERGED output can't be reused here, only its pre-merge
    split.)
  - vtkvmtkCenterlineUtilities.GetGroupUniqueCellIds() gives the (still
    occasionally >1) geometrically-distinct cell(s) for a group. The
    longest is kept as that group's representative for topology bookkeeping
    (adjacency/root-finding/generations - all per-GroupId, so which cell
    represents the group doesn't matter there). For the actual exported
    models/curves, EVERY candidate cell is written out separately - one
    file/curve per cell, matching CenterlineDisassembly.processGroupIds()/
    _createPolyData(), which does the same (one output polydata per cell
    returned by GetGroupUniqueCellIds(), not just the longest).
  - Groups with an empty upstream list are, unambiguously, the tree
    root(s) - no --endpoints/--source-point guessing needed.
  - "Blanked" groups (IsGroupBlanked) are bifurcation/junction tracts;
    "non-blanked" groups are the actual named branches - the same split
    CenterlineDisassembly.py exposes as its "Bifurcations"/"Branches"
    output categories. A branch's "generation" here is the number of
    blanked (bifurcation) groups crossed from the root to reach it.

CAVEAT: vtkvmtkCenterlineBranchExtractor itself is the expensive step here
(confirmed: ~19 minutes on a real 390-path/207k-point centerline - skipping
the later vtkvmtkMergeCenterlines step, as this script does, does NOT save
time; the splitting itself is the bottleneck, not the merge/resampling). Feed
this run() a centerline that vescan.stages.centerline already
branch-split (its own split_output_path) to skip re-running it here; --save-
split still caches whatever split ends up being used, for reuse on a LATER,
separate invocation of just this stage.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.build_graph raw_centerline.vtk graph_model.vtk \\
        --split-output output_5000_1.0_1.0/05_branch_tree.vtk
    python -m vescan.stages.build_graph output_5000_1.0_1.0/05_branch_tree.vtk graph_model.vtk \\
        --curve-output branch_tree_curves.mrk.json --bifurcation-output bifurcation_points.mrk.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, deque

import vtk

try:
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
except ImportError:
    from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry

from vescan.io import load_surface, save_surface, detect_coordinate_space
from vescan.stages.network import build_curve_markups
from vescan.stages.cut_graph import _extract_cells

logger = logging.getLogger(__name__)

RADIUS_ARRAY_NAME = "Radius"
BLANKING_ARRAY_NAME = "Blanking"
GROUP_IDS_ARRAY_NAME = "GroupIds"
CENTERLINE_IDS_ARRAY_NAME = "CenterlineIds"
TRACT_IDS_ARRAY_NAME = "TractIds"

REQUIRED_ARRAYS = (GROUP_IDS_ARRAY_NAME, CENTERLINE_IDS_ARRAY_NAME, TRACT_IDS_ARRAY_NAME, BLANKING_ARRAY_NAME)

VIRTUAL_ROOT_GROUP_ID = -1  # never collides with a real VMTK GroupId (always >= 0)

# A root heading a subtree shorter than this fraction of the longest root's
# subtree is a degenerate stub, not a co-equal inflow - see
# find_substantial_roots().
VIRTUAL_ROOT_MIN_LENGTH_FRACTION = 0.1


def _distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def run_branch_extractor(centerlinePolyData):
    """Runs vtkvmtkCenterlineBranchExtractor - the expensive step (see
    module docstring). Splits the raw overlapping source->target paths into
    GroupIds/CenterlineIds/TractIds/Blanking-tagged tracts. Only reached via
    ensure_split() when the input wasn't already split by someone else (e.g.
    vescan.stages.centerline's split_centerline())."""
    branchExtractor = vtkvmtkComputationalGeometry.vtkvmtkCenterlineBranchExtractor()
    branchExtractor.SetInputData(centerlinePolyData)
    branchExtractor.SetBlankingArrayName(BLANKING_ARRAY_NAME)
    branchExtractor.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    branchExtractor.SetGroupIdsArrayName(GROUP_IDS_ARRAY_NAME)
    branchExtractor.SetCenterlineIdsArrayName(CENTERLINE_IDS_ARRAY_NAME)
    branchExtractor.SetTractIdsArrayName(TRACT_IDS_ARRAY_NAME)
    branchExtractor.Update()
    return branchExtractor.GetOutput()


def ensure_split(polyData):
    if all(polyData.GetCellData().GetArray(name) is not None for name in REQUIRED_ARRAYS):
        return polyData
    logger.info("Input has no GroupIds/CenterlineIds/TractIds/Blanking cell arrays - running "
                "vtkvmtkCenterlineBranchExtractor. This takes ~15-20 minutes on a dense centerline "
                "(confirmed on real data) - use --split-output to cache the result for next time, or feed "
                "in vescan.stages.centerline's own split_output_path to skip this entirely.")
    return run_branch_extractor(polyData)


def list_group_ids(splitCenterlines):
    cu = vtkvmtkComputationalGeometry.vtkvmtkCenterlineUtilities
    allGroups = vtk.vtkIdList()
    cu.GetGroupsIdList(splitCenterlines, GROUP_IDS_ARRAY_NAME, allGroups)
    return [allGroups.GetId(i) for i in range(allGroups.GetNumberOfIds())]


def build_group_data(splitCenterlines, groupIds, min_length=0.0):
    """One entry per GroupId: representative geometry (the longest of
    GetGroupUniqueCellIds()'s still-possibly-multiple candidates - see
    module docstring), blanked flag, length, radius. Points/radii are read
    into fresh per-group lists (never reusing the input's point IDs
    downstream), so nothing here depends on point-ID continuity holding
    across cells."""
    cu = vtkvmtkComputationalGeometry.vtkvmtkCenterlineUtilities
    radiusArray = splitCenterlines.GetPointData().GetArray(RADIUS_ARRAY_NAME)

    groups = {}
    for groupId in groupIds:
        uniqueCellIds = vtk.vtkIdList()
        cu.GetGroupUniqueCellIds(splitCenterlines, GROUP_IDS_ARRAY_NAME, groupId, uniqueCellIds)
        bestCellId, bestNPoints = None, -1
        for i in range(uniqueCellIds.GetNumberOfIds()):
            cellId = uniqueCellIds.GetId(i)
            nPoints = splitCenterlines.GetCell(cellId).GetNumberOfPoints()
            if nPoints > bestNPoints:
                bestCellId, bestNPoints = cellId, nPoints
        if bestCellId is None or bestNPoints < 2:
            continue

        pointIds = splitCenterlines.GetCell(bestCellId).GetPointIds()
        positions = [splitCenterlines.GetPoint(pointIds.GetId(i)) for i in range(pointIds.GetNumberOfIds())]
        radii = ([radiusArray.GetValue(pointIds.GetId(i)) for i in range(pointIds.GetNumberOfIds())]
                 if radiusArray is not None else [])
        length = sum(_distance(positions[i], positions[i + 1]) for i in range(len(positions) - 1))
        if length < min_length:
            continue
        isBlanked = bool(cu.IsGroupBlanked(splitCenterlines, GROUP_IDS_ARRAY_NAME, BLANKING_ARRAY_NAME, groupId))

        groups[groupId] = {
            "group_id": groupId,
            "cell_id": bestCellId,
            "n_candidate_cells": uniqueCellIds.GetNumberOfIds(),
            "blanked": isBlanked,
            "positions": positions,
            "radii": radii,
            "length": length,
            "avg_radius": (sum(radii) / len(radii)) if radii else None,
            "generation": -1,
        }
    return groups


def build_group_instances(splitCenterlines, groups, groupIds, min_length=0.0):
    """One entry per (GroupId, geometrically-distinct candidate cell) - i.e.
    what actually gets exported. Mirrors CenterlineDisassembly.processGroupIds()/
    _createPolyData(): every cell GetGroupUniqueCellIds() returns for a group
    becomes its own output, not just the longest (contrast with
    build_group_data(), whose one-per-group `groups` dict above is for
    topology bookkeeping only - see module docstring)."""
    cu = vtkvmtkComputationalGeometry.vtkvmtkCenterlineUtilities
    radiusArray = splitCenterlines.GetPointData().GetArray(RADIUS_ARRAY_NAME)

    instances = []
    for groupId in groupIds:
        if groupId not in groups:
            continue  # dropped by build_group_data (degenerate group / representative below min_length)
        uniqueCellIds = vtk.vtkIdList()
        cu.GetGroupUniqueCellIds(splitCenterlines, GROUP_IDS_ARRAY_NAME, groupId, uniqueCellIds)
        isBlanked = groups[groupId]["blanked"]
        generation = groups[groupId]["generation"]
        nInstances = uniqueCellIds.GetNumberOfIds()
        for i in range(nInstances):
            cellId = uniqueCellIds.GetId(i)
            pointIds = splitCenterlines.GetCell(cellId).GetPointIds()
            if pointIds.GetNumberOfIds() < 2:
                continue
            positions = [splitCenterlines.GetPoint(pointIds.GetId(j)) for j in range(pointIds.GetNumberOfIds())]
            radii = ([radiusArray.GetValue(pointIds.GetId(j)) for j in range(pointIds.GetNumberOfIds())]
                     if radiusArray is not None else [])
            length = sum(_distance(positions[j], positions[j + 1]) for j in range(len(positions) - 1))
            if length < min_length:
                continue
            instances.append({
                "group_id": groupId,
                "cell_id": cellId,
                "instance_index": i,
                "n_instances": nInstances,
                "blanked": isBlanked,
                "positions": positions,
                "radii": radii,
                "length": length,
                "avg_radius": (sum(radii) / len(radii)) if radii else None,
                "generation": generation,
            })
    return instances


def build_group_cell_ids(splitCenterlines, groups, groupIds):
    """All raw splitCenterlines cell indices belonging to a surviving group
    in `groups` (every GetGroupUniqueCellIds() candidate cell, not just the
    representative) - used to build a physically filtered copy of the raw
    split centerline (see save_split_output()) with degenerate/short
    groups' cells actually removed, not just excluded from the later
    per-group/instance exports. See module docstring / repair_orphan_roots()
    for why so many near-zero-length stub tracts show up in real data -
    those are exactly what --min-branch-length filters out here."""
    cu = vtkvmtkComputationalGeometry.vtkvmtkCenterlineUtilities
    cellIds = []
    for groupId in groupIds:
        if groupId not in groups:
            continue
        uniqueCellIds = vtk.vtkIdList()
        cu.GetGroupUniqueCellIds(splitCenterlines, GROUP_IDS_ARRAY_NAME, groupId, uniqueCellIds)
        for i in range(uniqueCellIds.GetNumberOfIds()):
            cellIds.append(uniqueCellIds.GetId(i))
    return cellIds


def build_group_adjacency(splitCenterlines, groupIds):
    """Authoritative upstream/downstream GroupId adjacency, from
    vtkvmtkCenterlineUtilities.FindAdjacentCenterlineGroupIds() (matches
    TractIds +/-1 on the same CenterlineId - see module docstring). Works
    on blanked (bifurcation) and non-blanked (branch) groups alike."""
    cu = vtkvmtkComputationalGeometry.vtkvmtkCenterlineUtilities
    adjacency = {}
    for groupId in groupIds:
        up = vtk.vtkIdList()
        down = vtk.vtkIdList()
        cu.FindAdjacentCenterlineGroupIds(splitCenterlines, GROUP_IDS_ARRAY_NAME, CENTERLINE_IDS_ARRAY_NAME,
                                           TRACT_IDS_ARRAY_NAME, groupId, up, down)
        adjacency[groupId] = {
            "upstream": [up.GetId(i) for i in range(up.GetNumberOfIds())],
            "downstream": [down.GetId(i) for i in range(down.GetNumberOfIds())],
        }
    return adjacency


def find_roots(adjacency):
    """Groups with no upstream neighbor - unambiguously the tree root(s)
    (normally exactly one: the trunk). No spatial/--endpoints guessing
    needed, unlike the tolerance-clustering version of this script."""
    return [groupId for groupId, links in adjacency.items() if not links["upstream"]]


def repair_orphan_roots(groups, adjacency, roots, tolerance=0.5):
    """FindAdjacentCenterlineGroupIds occasionally fails to find a group's
    true upstream neighbor - confirmed on real data: of 112 groups that
    came back with an empty upstream list, 111 were near-zero-length
    degenerate stub tracts (mean length 0.36mm, vs the one genuine root's
    34mm) scattered across the WHOLE tree, not clustered near the source -
    a topology-matching edge case on very short tracts, not a data/anatomy
    issue. For every such orphan except the longest (kept as the true
    root), tries to geometrically match its start point to another group's
    end point within `tolerance` and splices in that upstream link -
    exactly the endpoint-clustering approach the previous version of this
    script used as its *primary* mechanism, now scoped to only the small
    minority of cases where the authoritative topology lookup misses.

    An orphan not in `groups` (its own tract was below --min-branch-length)
    has no position data to attempt a geometric match with, but MUST still
    come back out in the returned root list - it may be the sole upstream
    link into an otherwise perfectly healthy, arbitrarily large subtree
    (confirmed on real data: a single dropped ~1mm stub silently orphaned
    697 of 754 groups downstream of it before this was handled). Dropping
    it here instead of keeping it as "still orphan" would make
    compute_generations() unable to ever reach that subtree, no matter how
    it handles missing-from-groups roots itself. Returns the repaired root
    list (should shrink to ~1 for a normal single-trunk vessel, but can
    legitimately stay >1 for something like a venous confluence with more
    than one true co-equal inflow)."""
    if len(roots) <= 1:
        return roots

    primaryRoot = max(roots, key=lambda r: groups[r]["length"] if r in groups else -1)
    endpointIndex = [(groupId, group["positions"][-1]) for groupId, group in groups.items()]

    repairedCount = 0
    stillOrphan = []
    for orphan in roots:
        if orphan == primaryRoot:
            continue
        if orphan not in groups:
            stillOrphan.append(orphan)
            continue
        startPosition = groups[orphan]["positions"][0]
        bestParent, bestDistance = None, tolerance
        for groupId, endPosition in endpointIndex:
            if groupId == orphan:
                continue
            d = _distance(startPosition, endPosition)
            if d <= bestDistance:
                bestParent, bestDistance = groupId, d
        if bestParent is not None:
            adjacency[orphan]["upstream"].append(bestParent)
            adjacency[bestParent]["downstream"].append(orphan)
            repairedCount += 1
        else:
            stillOrphan.append(orphan)

    if repairedCount:
        logger.info("Repaired %d orphan group(s) by geometric endpoint matching (topology-based "
                    "adjacency missed their upstream link - see module docstring)", repairedCount)
    return [primaryRoot] + stillOrphan


def compute_generations(roots, adjacency, groups):
    """BFS from the root(s) over the downstream adjacency. A group's
    "generation" is the number of BLANKED (bifurcation) groups crossed to
    reach it - the trunk is generation 0. Groups not reached (disconnected
    fragments) are left at generation -1 (see build_group_data).

    A root missing from `groups` (below --min-branch-length) still seeds
    the BFS - it just doesn't get a generation written to a (nonexistent)
    groups[root] entry. Skipping it entirely here, instead of just skipping
    the write, would silently strand its whole downstream subtree as
    "unreached" even though every group in it is otherwise perfectly
    healthy - confirmed on real data (see repair_orphan_roots()'s own
    docstring): a single ~1mm stub below the threshold orphaned 697 of 754
    groups this way before this was fixed."""
    groupGeneration = {}
    visited = set()
    queue = deque()
    for root in roots:
        groupGeneration[root] = 0
        if root in groups:
            groups[root]["generation"] = 0
        visited.add(root)
        queue.append(root)

    while queue:
        groupId = queue.popleft()
        crossesBifurcation = groups[groupId]["blanked"] if groupId in groups else False
        for neighbor in adjacency.get(groupId, {}).get("downstream", []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            generation = groupGeneration[groupId] + (1 if crossesBifurcation else 0)
            groupGeneration[neighbor] = generation
            if neighbor in groups:
                groups[neighbor]["generation"] = generation
            queue.append(neighbor)

    return groupGeneration


def print_statistics(groups, adjacency, roots, add_virtual_root=False,
                      virtual_root_min_length_fraction=VIRTUAL_ROOT_MIN_LENGTH_FRACTION):
    branches = [g for g in groups.values() if not g["blanked"]]
    bifurcations = [g for g in groups.values() if g["blanked"]]
    unreachable = [g for g in groups.values() if g["generation"] < 0]

    logger.info("=== Centerline tree statistics ===")
    logger.info("Groups: %d total (%d branches, %d bifurcation zones)",
                len(groups), len(branches), len(bifurcations))
    if len(roots) == 1:
        rootsNote = ""
    elif add_virtual_root:
        nSubstantial = len(find_substantial_roots(roots, groups, adjacency,
                                                    min_length_fraction=virtual_root_min_length_fraction))
        rootsNote = (f"  <- {nSubstantial} of these head a substantial subtree and will be collapsed under one "
                     f"virtual root node in graph.json; the rest head degenerate stubs only (see "
                     f"build_graph_data()/find_substantial_roots())") if nSubstantial > 1 else \
                    ("  <- only one of these heads a substantial subtree, the rest are degenerate stubs: a "
                     "normal single-trunk tree, no virtual root added (see find_substantial_roots())")
    else:
        rootsNote = "  <- expected exactly 1, double check the input"
    logger.info("Roots found (no upstream group): %s%s", roots, rootsNote)
    if unreachable:
        logger.warning("%d group(s) not reached from the root(s) - disconnected fragment in the "
                        "centerline, or a root-finding edge case.", len(unreachable))

    childCounts = Counter(len(adjacency[g["group_id"]]["downstream"]) for g in bifurcations)
    for childCount, count in sorted(childCounts.items()):
        logger.info("  %d bifurcation(s) splitting into %d children", count, childCount)

    reachableBranchGenerations = [g["generation"] for g in branches if g["generation"] >= 0]
    maxGeneration = max(reachableBranchGenerations) if reachableBranchGenerations else 0
    logger.info("Max depth: %d bifurcation(s) from root to the deepest branch", maxGeneration)

    logger.info("Branches per generation (0 = trunk from root):")
    perGeneration = Counter(g["generation"] for g in branches)
    for generation in sorted(perGeneration):
        label = str(generation) if generation >= 0 else "unreachable"
        logger.info("  generation %s: %d branch(es)", label, perGeneration[generation])

    multiCandidate = [g for g in groups.values() if g["n_candidate_cells"] > 1]
    if multiCandidate:
        logger.info("Note: %d group(s) had more than one geometrically-distinct candidate cell (minor "
                    "duplicate tracts from the raw branch split) - each candidate cell is exported as its "
                    "own model/curve file (matching CenterlineDisassembly), so the file count exceeds the "
                    "group count.", len(multiCandidate))


def build_single_group_polydata(group):
    """One vtkPolyData containing just this group's own geometry - mirrors
    CenterlineDisassembly._createPolyData()'s per-cell polydata (one
    branch/bifurcation = one file/node, not a shared multi-cell dataset)."""
    points = vtk.vtkPoints()
    radiusArray = vtk.vtkDoubleArray()
    radiusArray.SetName(RADIUS_ARRAY_NAME)

    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(len(group["positions"]))
    for i, position in enumerate(group["positions"]):
        pointId = points.InsertNextPoint(position)
        polyLine.GetPointIds().SetId(i, pointId)
        radiusArray.InsertNextValue(group["radii"][i] if group["radii"] else 0.0)

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(polyLine)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    polyData.GetPointData().AddArray(radiusArray)
    return polyData


def save_group_models(instances, output_dir, coordinate_space="LPS"):
    """Mirrors CenterlineDisassembly's onApplyButton()/_createModelComponent():
    one .vtk model file per exported cell instance - "Branch_Model_..." for
    non-blanked (branch) groups, "Bifurcation_Model_..." for blanked
    (bifurcation zone) groups - instead of bundling everything into one
    multi-cell polydata. This is the direct standalone equivalent of what the
    Slicer module creates as separate scene nodes, including creating more
    than one file for the same GroupId when it has more than one candidate
    cell (see build_group_instances()). File names embed GroupId and
    Generation (plus a _cellN suffix when a group has multiple instances) so
    a later "cut after N bifurcations" step can just filter the file list by
    generation, no need to touch the geometry again. Returns the list of
    paths written."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for instance in instances:
        kind = "Bifurcation_Model" if instance["blanked"] else "Branch_Model"
        suffix = "" if instance["n_instances"] <= 1 else f"_cell{instance['instance_index']}"
        filename = f"{kind}_group{instance['group_id']}_gen{instance['generation']}{suffix}.vtk"
        path = os.path.join(output_dir, filename)
        polyData = build_single_group_polydata(instance)
        save_surface(polyData, path, coordinate_space=coordinate_space)
        paths.append(path)
    return paths


def add_group_metadata_arrays(splitCenterlines, groups):
    """Adds friendly per-cell arrays - GroupId/IsBifurcation/Generation/
    Length/AverageRadius - to splitCenterlines' OWN CellData, in place, one
    entry per EXISTING cell (reading each cell's own GroupIds/Blanking value
    directly, not re-deriving GetGroupUniqueCellIds() groupings). This is
    what lets save_split_output() below export a single file that is both
    the branch-split centerline (GroupIds/CenterlineIds/TractIds/Blanking/
    Radius, for ensure_split()/clip_vessel.py) AND the per-group geometry
    summary previously only available in a separate "combined model" file
    (for cut_graph.py/lobe_reachability.py/transfer_centerline_labels.py) -
    see module docstring.

    GroupId (singular) duplicates the existing GroupIds (plural) array's
    values - kept as its own int array anyway since downstream code already
    expects that exact name/dtype. IsBifurcation duplicates Blanking under a
    self-explanatory name (VMTK's own "blanking" terminology means "this
    tract is a bifurcation/junction zone, not a named branch" - not obvious
    without reading vtkvmtkCenterlineBranchExtractor's own docs). Cells
    whose GroupId isn't in `groups` (dropped by --min-branch-length) get
    Generation=-1/Length=0/AverageRadius=0 - harmless, since
    save_split_output() only ever exports cells whose group survived."""
    groupIdSourceArray = splitCenterlines.GetCellData().GetArray(GROUP_IDS_ARRAY_NAME)
    blankingArray = splitCenterlines.GetCellData().GetArray(BLANKING_ARRAY_NAME)

    groupIdArray = vtk.vtkIntArray()
    groupIdArray.SetName("GroupId")
    isBifurcationArray = vtk.vtkIntArray()
    isBifurcationArray.SetName("IsBifurcation")
    generationArray = vtk.vtkIntArray()
    generationArray.SetName("Generation")
    lengthArray = vtk.vtkDoubleArray()
    lengthArray.SetName("Length")
    avgRadiusArray = vtk.vtkDoubleArray()
    avgRadiusArray.SetName("AverageRadius")

    for cellId in range(splitCenterlines.GetNumberOfCells()):
        groupId = int(groupIdSourceArray.GetValue(cellId))
        group = groups.get(groupId)
        groupIdArray.InsertNextValue(groupId)
        isBifurcationArray.InsertNextValue(int(blankingArray.GetValue(cellId)))
        generationArray.InsertNextValue(group["generation"] if group else -1)
        lengthArray.InsertNextValue(group["length"] if group else 0.0)
        avgRadiusArray.InsertNextValue((group["avg_radius"] or 0.0) if group else 0.0)

    splitCenterlines.GetCellData().AddArray(groupIdArray)
    splitCenterlines.GetCellData().AddArray(isBifurcationArray)
    splitCenterlines.GetCellData().AddArray(generationArray)
    splitCenterlines.GetCellData().AddArray(lengthArray)
    splitCenterlines.GetCellData().AddArray(avgRadiusArray)


def save_bifurcation_markups(groups, adjacency, output_path, coordinate_space="LPS"):
    """Writes a Markups fiducial .mrk.json with one point per bifurcation
    (blanked) group, at its first position, labeled with its number of
    downstream children."""
    controlPoints = []
    for group in groups.values():
        if not group["blanked"]:
            continue
        childCount = len(adjacency[group["group_id"]]["downstream"])
        controlPoints.append({
            "id": str(len(controlPoints) + 1),
            "label": f"bifurcation_group{group['group_id']}_{childCount}children",
            "description": "",
            "associatedNodeID": "",
            "position": list(group["positions"][0]),
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
            "name": "bifurcations",
            "controlPoints": controlPoints,
        }],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return len(controlPoints)


def _subtree_length(rootId, groups, adjacency):
    """Total centerline length of everything reachable downstream of rootId,
    including rootId itself. Groups dropped by --min-branch-length contribute
    nothing but are still traversed, so a surviving subtree hanging off a
    dropped stub is still measured in full."""
    seen = set()
    stack = [rootId]
    total = 0.0
    while stack:
        groupId = stack.pop()
        if groupId in seen:
            continue
        seen.add(groupId)
        group = groups.get(groupId)
        if group:
            total += group["length"]
        for child in adjacency.get(groupId, {}).get("downstream", []):
            if child not in seen:
                stack.append(child)
    return total


def _root_start_position(rootId, groups, adjacency):
    """Where the trunk headed by rootId physically starts: rootId's own first
    point, or - when rootId was dropped by --min-branch-length and so has no
    geometry at all - the first point of its nearest surviving descendant.
    Confirmed on real data that the root of the main tree is frequently one
    of those dropped sub-millimetre stubs (see repair_orphan_roots()), so
    reading positions only off roots present in `groups` silently skips the
    very trunk whose position is wanted. Returns None if nothing downstream
    survived."""
    seen = set()
    queue = deque([rootId])
    while queue:
        groupId = queue.popleft()
        if groupId in seen:
            continue
        seen.add(groupId)
        group = groups.get(groupId)
        if group:
            return group["positions"][0]
        for child in adjacency.get(groupId, {}).get("downstream", []):
            if child not in seen:
                queue.append(child)
    return None


def find_substantial_roots(roots, groups, adjacency,
                            min_length_fraction=VIRTUAL_ROOT_MIN_LENGTH_FRACTION):
    """The roots that actually head a real vessel trunk, judged by their
    subtree's total length against the longest root's.

    repair_orphan_roots() returns two very different kinds of thing in one
    list: the genuine root(s), and the near-zero-length degenerate stubs it
    could not splice back in (confirmed on real data: 150 roots for one
    vein tree - one carrying 6164mm, i.e. 100% of the tree, and 149 carrying
    <= 1.2mm each). Both must stay in that list, since compute_generations()
    seeds its BFS from it and a dropped stub can be the sole link into a
    large healthy subtree - but only the former are "co-equal inflows" in
    the sense build_graph_data()'s virtual root means."""
    lengths = {root: _subtree_length(root, groups, adjacency) for root in roots}
    longest = max(lengths.values(), default=0.0)
    if longest <= 0.0:
        return list(roots)
    return [root for root in roots if lengths[root] >= longest * min_length_fraction]


def build_graph_data(groups, adjacency, roots, add_virtual_root=False,
                      virtual_root_min_length_fraction=VIRTUAL_ROOT_MIN_LENGTH_FRACTION):
    """Serializable node/edge view of the tree - the piece missing from every
    other output (see module docstring: filenames/cell arrays carry group_id/
    generation/blanked, but never the upstream/downstream edges themselves).
    nodes: one entry per group, keyed by GroupId as a string (JSON object
    keys must be strings). edges: one entry per (parent, child) adjacency,
    downstream-only (each edge appears once, from its upstream/parent side).

    add_virtual_root=True and more than one SUBSTANTIAL root
    (find_substantial_roots()): adds ONE synthetic VIRTUAL_ROOT_GROUP_ID node
    as their shared parent - for vessels (veins, at a venous confluence) that
    genuinely have more than one true co-equal root instead of a single
    directional trunk. Deliberately NOT added to `nodes` in the normal sense
    with real branch/bifurcation stats - it has no geometry, blanked=False
    (never crosses a bifurcation, so its children's generation numbers are
    unaffected: `groupGeneration[root] + 0`), and generation=-1, so it's
    automatically invisible to everything that already treats generation<0
    as "not a real counted group" (print_statistics() never sees it at all
    since it only reads `groups`, not this `nodes` dict; cut_graph.py's
    max-generation/kept-groups logic and lobe_reachability.py's node/length
    aggregation both already skip generation<0 or fall back to 0 via
    .get() for an id with no entry).

    Only the substantial roots are parented under it; the degenerate stub
    roots repair_orphan_roots() could not splice stay separate entries in
    the returned "roots" list. Both halves of that matter downstream:
    lobe_reachability.find_disconnected_fragment_ids() seeds one connected
    component per reported root and discards all but the biggest, so leaving
    the stubs out of the virtual node keeps them detectable as the noise
    fragments they are, while gathering the genuine co-equal trunks INTO it
    keeps them in one component where that same "biggest wins" rule can
    never throw the smaller ones away.

    `position` is the centroid of the substantial roots' own start points,
    read via _root_start_position() so that a root dropped by
    --min-branch-length still contributes the position of its first
    surviving descendant instead of being skipped (confirmed on real data
    that the main tree's own root is often exactly such a dropped stub, and
    that averaging only over the roots present in `groups` put the node
    halfway between two unrelated ~1mm noise fragments instead of on the
    confluence). The real trunks all begin at (or immediately next to) the
    same physical confluence, so this lands right there without needing the
    endpoints stage's own file plumbed in here."""
    nodes = {
        str(groupId): {
            "group_id": groupId,
            "blanked": group["blanked"],
            "generation": group["generation"],
            "length": group["length"],
            "avg_radius": group["avg_radius"],
            "n_candidate_cells": group["n_candidate_cells"],
        }
        for groupId, group in groups.items()
    }
    edges = [
        {"parent": groupId, "child": childId}
        for groupId, links in adjacency.items()
        for childId in links["downstream"]
    ]

    if add_virtual_root and len(roots) > 1:
        substantialRoots = find_substantial_roots(roots, groups, adjacency,
                                                    min_length_fraction=virtual_root_min_length_fraction)
        if len(substantialRoots) > 1:
            startPositions = [position for position in
                              (_root_start_position(r, groups, adjacency) for r in substantialRoots)
                              if position is not None]
            centroid = ([sum(p[i] for p in startPositions) / len(startPositions) for i in range(3)]
                        if startPositions else [0.0, 0.0, 0.0])
            nodes[str(VIRTUAL_ROOT_GROUP_ID)] = {
                "group_id": VIRTUAL_ROOT_GROUP_ID,
                "virtual": True,
                "blanked": False,
                "generation": -1,
                "length": 0.0,
                "avg_radius": None,
                "n_candidate_cells": 0,
                "position": centroid,
            }
            edges = edges + [{"parent": VIRTUAL_ROOT_GROUP_ID, "child": r} for r in substantialRoots]
            stubRoots = [r for r in roots if r not in set(substantialRoots)]
            roots = [VIRTUAL_ROOT_GROUP_ID] + stubRoots
            logger.info("Collapsed %d co-equal root(s) %s under one virtual root at %s; the remaining %d "
                        "root(s) head only degenerate stubs and stay separate.",
                        len(substantialRoots), substantialRoots, [round(c, 1) for c in centroid], len(stubRoots))
        else:
            logger.info("No virtual root added: of %d root(s), only one heads a substantial subtree - the rest "
                        "are degenerate stubs (see find_substantial_roots()), so this is a normal "
                        "single-trunk tree, not a confluence.", len(roots))

    return {"roots": roots, "nodes": nodes, "edges": edges}


def save_graph_json(groups, adjacency, roots, output_path, add_virtual_root=False,
                     virtual_root_min_length_fraction=VIRTUAL_ROOT_MIN_LENGTH_FRACTION):
    """Writes build_graph_data()'s nodes/edges to a JSON file. Returns the
    roots actually written alongside the counts - build_graph_data() decides
    for itself whether a virtual root replaces them (see
    find_substantial_roots()), so callers must report what it wrote rather
    than predict it."""
    data = build_graph_data(groups, adjacency, roots, add_virtual_root=add_virtual_root,
                             virtual_root_min_length_fraction=virtual_root_min_length_fraction)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return len(data["nodes"]), len(data["edges"]), data["roots"]


def save_split_output(splitCenterlines, groups, groupIds, output_path, coordinate_space="LPS"):
    """Writes THE split file (see module docstring): every cell belonging to
    a group dropped by --min-branch-length is physically removed (not just
    excluded from the later per-group/graph exports) - real data can have
    hundreds of near-zero-length degenerate stub tracts (see
    repair_orphan_roots()'s docstring) scattered through the raw split
    output, each showing up as an isolated point-like "centerline" if you
    load that raw file directly in Slicer. What's left keeps the original
    GroupIds/CenterlineIds/TractIds/Blanking/Radius arrays intact (so it's
    still valid input for ensure_split()/clip_vessel_surface.py) PLUS the
    friendly GroupId/IsBifurcation/Generation/Length/AverageRadius arrays
    add_group_metadata_arrays() adds (for cut_graph.py/lobe_reachability.py/
    transfer_centerline_labels.py) - one file instead of the previous
    raw-split + cleaned-split + combined-model trio."""
    add_group_metadata_arrays(splitCenterlines, groups)
    keptCellIds = build_group_cell_ids(splitCenterlines, groups, groupIds)
    cleaned = _extract_cells(splitCenterlines, keptCellIds)
    save_surface(cleaned, output_path, coordinate_space=coordinate_space)
    return cleaned.GetNumberOfCells()


def run(input_centerline_path, output_dir, curve_output_path=None,
        bifurcation_output_path=None, graph_output_path=None, split_output_path=None,
        min_branch_length=0.0, add_virtual_root=False,
        orphan_root_repair_tolerance=0.5,
        virtual_root_min_length_fraction=VIRTUAL_ROOT_MIN_LENGTH_FRACTION):
    coordinateSpace = detect_coordinate_space(input_centerline_path) or "LPS"
    polyData = load_surface(input_centerline_path)
    splitCenterlines = ensure_split(polyData)

    groupIds = list_group_ids(splitCenterlines)
    logger.info("Found %d groups in %d split cells", len(groupIds), splitCenterlines.GetNumberOfCells())

    groups = build_group_data(splitCenterlines, groupIds, min_length=min_branch_length)
    adjacency = build_group_adjacency(splitCenterlines, groupIds)
    roots = find_roots(adjacency)
    if not roots:
        raise ValueError("No group with empty upstream found - could not determine the tree root")
    roots = repair_orphan_roots(groups, adjacency, roots, tolerance=orphan_root_repair_tolerance)
    compute_generations(roots, adjacency, groups)

    print_statistics(groups, adjacency, roots, add_virtual_root=add_virtual_root,
                      virtual_root_min_length_fraction=virtual_root_min_length_fraction)

    instances = build_group_instances(splitCenterlines, groups, groupIds, min_length=min_branch_length)

    paths = save_group_models(instances, output_dir, coordinate_space=coordinateSpace)
    nBranches = sum(1 for inst in instances if not inst["blanked"])
    nBifurcations = sum(1 for inst in instances if inst["blanked"])
    logger.info("Saved %d model(s) to %s/ (%d Branch_Model_*.vtk, %d Bifurcation_Model_*.vtk) - one file per "
                "branch/bifurcation candidate cell, matching CenterlineDisassembly's one-node-per-cell output "
                "(a GroupId with more than one geometrically-distinct candidate cell yields more than one file - "
                "see the note above if any showed up). Load them all as Models in 3D Slicer (drag the folder in, "
                "or File > Add Data). Filenames embed GroupId/Generation for filtering later.",
                len(paths), output_dir, nBranches, nBifurcations)

    if curve_output_path:
        branchInstances = [inst for inst in instances if not inst["blanked"]]
        curveLabels = [f"group{inst['group_id']}_gen{inst['generation']}" for inst in branchInstances]
        data = build_curve_markups([inst["positions"] for inst in branchInstances], coordinate_space=coordinateSpace,
                                    base_name="branch", indices=curveLabels)
        with open(curve_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d per-branch curve(s) to %s (curve names embed GroupId/Generation, matching the "
                    ".vtk model filenames)", len(branchInstances), curve_output_path)

    if bifurcation_output_path:
        n = save_bifurcation_markups(groups, adjacency, bifurcation_output_path, coordinate_space=coordinateSpace)
        logger.info("Saved %d bifurcation fiducial(s) to %s", n, bifurcation_output_path)

    if graph_output_path:
        nNodes, nEdges, savedRoots = save_graph_json(groups, adjacency, roots, graph_output_path,
                                                      add_virtual_root=add_virtual_root,
                                                      virtual_root_min_length_fraction=virtual_root_min_length_fraction)
        logger.info("Saved graph (%d nodes, %d edges, roots=%s) to %s", nNodes, nEdges, savedRoots,
                    graph_output_path)

    if split_output_path:
        nCells = save_split_output(splitCenterlines, groups, groupIds, split_output_path,
                                    coordinate_space=coordinateSpace)
        logger.info("Saved split centerline (%d cells, degenerate/short groups removed, with GroupId/"
                    "IsBifurcation/Generation/Length/AverageRadius cell arrays) to %s - use this as "
                    "clip_vessel_surface.py's split_centerline input, cut_graph.py's combined_model input, or a "
                    "later run's ensure_split() cache", nCells, split_output_path)

    return groups, adjacency, roots


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_centerline", help="Path to the centerline model (.vtk/.vtp) - either the raw "
                                                  "CenterlineModel or an already branch-split (--split-output) file")
    parser.add_argument("output_dir", help="Directory to save one .vtk model file per branch/bifurcation into "
                                            "(Branch_Model_groupN_genG.vtk / Bifurcation_Model_groupN_genG.vtk) - "
                                            "matches CenterlineDisassembly's one-node-per-branch/bifurcation output")
    parser.add_argument("--curve-output", default=None,
                         help="Optional path to also save a Markups curve .mrk.json, one open curve per branch "
                              "(non-bifurcation group) - Slicer unpacks this into one curve node per branch on load")
    parser.add_argument("--bifurcation-output", default=None,
                         help="Optional path to also save a single Markups fiducial .mrk.json marking every "
                              "bifurcation group (convenience overview, in addition to the per-bifurcation "
                              ".vtk models in output_dir)")
    parser.add_argument("--split-output", default=None,
                         help="Path to also save THE split centerline (see module docstring): the branch-split "
                              "centerline with degenerate/short groups (--min-branch-length) physically removed, "
                              "carrying both the original GroupIds/CenterlineIds/TractIds/Blanking/Radius arrays "
                              "(valid input for a later run's ensure_split() cache, or for "
                              "clip_vessel_surface.py's split_centerline argument) AND friendly GroupId/"
                              "IsBifurcation/Generation/Length/AverageRadius cell arrays (valid input for "
                              "cut_graph.py's/lobe_reachability.py's combined_model argument) - one file replacing "
                              "what used to be three (raw split, cleaned split, combined model)")
    parser.add_argument("--graph-output", default=None,
                         help="Optional path to also save the tree structure itself as JSON: one node per group "
                              "(group_id/blanked/generation/length/avg_radius) and one edge per parent->child "
                              "GroupId adjacency, plus the root group_id(s) - the piece none of the other outputs "
                              "carry (filenames/cell arrays give you each group's own data, never who connects to "
                              "whom)")
    parser.add_argument("--min-branch-length", type=float, default=0.0,
                         help="Drop groups shorter than this (mm) - default 0: keep everything")
    parser.add_argument("--add-virtual-root", action="store_true",
                         help="If more than one SUBSTANTIAL root is found (no single directional trunk - e.g. a "
                              "venous confluence with several co-equal inflows), add one synthetic root node in "
                              "--graph-output as their shared parent. Roots heading only a degenerate stub are "
                              "judged by subtree length (see find_substantial_roots()) and left as separate roots, "
                              "so a normal single-trunk tree that merely carries some stub noise gets no virtual "
                              "root at all. Purely a graph.json bookkeeping node (position = centroid of the "
                              "substantial roots' start points): never exported as its own branch/bifurcation "
                              "model, never counted in the statistics above, and doesn't shift any real group's "
                              "generation number")
    parser.add_argument("--orphan-root-repair-tolerance", type=float, default=0.5, metavar="MM",
                         help="Geometric matching tolerance (mm) for splicing an orphan root back onto its "
                              "true upstream neighbor when the authoritative topology lookup misses it "
                              "(default: %(default)s) - see repair_orphan_roots()'s docstring.")
    parser.add_argument("--virtual-root-min-length-fraction", type=float,
                         default=VIRTUAL_ROOT_MIN_LENGTH_FRACTION, metavar="F",
                         help="A root heading a subtree shorter than this fraction of the longest root's "
                              f"subtree is treated as a degenerate stub, not a co-equal inflow (default: "
                              f"%(default)s) - see find_substantial_roots()'s docstring.")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.input_centerline,
        args.output_dir,
        curve_output_path=args.curve_output,
        bifurcation_output_path=args.bifurcation_output,
        graph_output_path=args.graph_output,
        split_output_path=args.split_output,
        min_branch_length=args.min_branch_length,
        add_virtual_root=args.add_virtual_root,
        orphan_root_repair_tolerance=args.orphan_root_repair_tolerance,
        virtual_root_min_length_fraction=args.virtual_root_min_length_fraction,
    )


if __name__ == "__main__":
    sys.exit(main())