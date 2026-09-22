#!/usr/bin/env python3
"""
Marks every branch/bifurcation group of a centerline tree
(vescan.stages.build_graph's output) with the pulmonary lobe(s) it is
REACHABLE FROM - i.e. every downstream path that ultimately supplies a given
lobe, not just the branches that happen to sit spatially inside that lobe's
surface.

Two-stage algorithm:
  1. SPATIAL SEEDING (containment - not the final answer). Each non-
     bifurcation ("branch") group's own points are tested against every lobe
     surface with vtkSelectEnclosedPoints. A group is *directly* assigned to
     a lobe when at least one of its points is inside that lobe's volume AND
     the inside fraction clears --containment-threshold (default 0.1, kept
     low deliberately: real segmented arterial trees are frequently pruned
     well before the sub-pleural periphery, so often only a branch's
     distal-most portion - not a majority of its length - ever actually
     crosses into the lobe it feeds). Proximal/shared trunks (main pulmonary
     artery, lobar bifurcations, ...) sit in the mediastinum outside every
     lobe surface and get NO direct assignment here - expected, not a bug:
     that's exactly what step 2 exists to fix.
  2. GRAPH REACHABILITY (the actual algorithm - see propagate_reachability()).
     graph.json's parent->child GroupId edges (build_graph's --graph-output)
     are walked bottom-up (post-order from the root(s)), so every group's
     "reachable lobes" = its own direct lobe(s) UNION the reachable lobes of
     every one of its downstream children. A group that ends up with more
     than one reachable lobe is, unambiguously, a shared proximal trunk
     feeding more than one lobe further downstream, and is marked as such.

This is why step 2 - not step 1 - answers "which branches lead to lobe X":
a trunk carrying blood toward both, say, RUL and RML is never itself inside
either lobe's surface, so containment alone would miss it entirely; the
bottom-up union computed here does not.

Inputs are vescan.stages.build_graph's own outputs, no re-run of
VMTK's branch extractor needed:
  - graph.json (--graph-output): topology (per-group generation/blanked,
    parent->child GroupId edges, root(s)).
  - the split file (--split-output, 05_branch_tree.vtk): geometry, one cell per
    group/candidate-cell instance with a 'GroupId' cell array.
plus one closed/watertight lobe surface (.vtk/.vtp) per lobe via --lobe.

Outputs a combined single-file .vtk/.vtp (same cells/order as the input
split file, so its existing GroupId/IsBifurcation/Generation/Length/
AverageRadius cell arrays are carried over unchanged) with new cell arrays:
  - NumReachableLobes: how many lobes a group's downstream subtree reaches
    (0 = disconnected/unclassified or genuinely extra-lobar, 1 = exclusive
    to one lobe, >1 = shared trunk).
  - LobeMask: bitmask, bit i set if lobe i (see the legend JSON / --lobe
    order) is reachable.
  - Reaches_<LOBE>: one array per --lobe, values 0 (does not lead here),
    1 (leads here exclusively) or 2 (leads here AND at least one other lobe
    - a shared trunk) - the "marked distinctly" the algorithm was asked for.
  - LobeLabel: a compact 0..N+2 classification for coloring (see
    compute_lobe_labels()), PURELY LOCAL/positional rather than subtree-
    union-based like LobeMask/NumReachableLobes above - 0 none, 1..N
    exclusive to lobeOrder[label-1] (this group's own points, tested for
    EVERY group including bifurcations, sit inside exactly that one lobe),
    N+1 trunk (this group's own points are outside every lobe), N+2 a
    crossing point (this group's own points touch >1 lobe) OR anything
    downstream of one, which stays flagged rather than reverting to the
    destination lobe's plain color. Confirmed on real patient data that a
    group 100% inside one lobe must NOT be painted as a shared trunk just
    because some descendant many generations downstream happens to cross
    into another lobe - that was LobeMask/NumReachableLobes' subtree-union
    behavior, kept as-is for --extract-lobe/Reaches_<LOBE> since "does this
    subtree feed lobe X at all" is the right question there, but wrong for
    what a human wants highlighted when actually looking at the tree.
  - LobeLabelPoint (see compute_point_lobe_labels()): a POINT array (every
    other array here is per-cell/per-group), the same 0..N+2 scheme as
    LobeLabel but tested on every individual centerline point directly,
    with no group-level aggregation and no downstream taint propagation.
    Exists because LobeLabel paints an ENTIRE branch group with one value
    even when only its distal-most portion actually crosses into a lobe -
    LobeLabelPoint instead marks the exact point where a vessel enters or
    leaves a lobe (the value changes between consecutive points along the
    centerline), so walking the point order together with arc length gives
    the precise crossing location and length in mm. N+2 here is a literal,
    non-propagated point simultaneously testing inside >1 lobe surface -
    real lobes don't overlap, so expect this to be rare, only right at a
    fissure where two lobe surfaces geometrically overlap.
Also writes a legend JSON (--legend-output) with the lobe bit order and,
per group, its direct/reachable lobes and classification - and, optionally,
one standalone subset centerline per lobe (--extract-lobe NAME=PATH)
containing only the groups that reach it (its own branches plus every
shared upstream trunk) - same 'GroupId' cell-array convention as
vescan.stages.cut_graph's output, so it can be fed straight into
vescan.stages.clip_vessel to carve out that lobe's own feeding-artery
surface.

Load the main output in 3D Slicer like any other centerline model (Display >
Scalars > Active Scalar = Reaches_<LOBE> or NumReachableLobes, a discrete
color table) - or run vescan.stages.transfer_centerline_labels against
it (--arrays NumReachableLobes,LobeMask,Reaches_<LOBE>,...) to transfer the
same arrays onto a vessel surface, exactly like GroupId/IsBifurcation/
Generation/CellId are transferred today.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.lobe_reachability Artery_centerlines/05_branch_tree_topology.json \\
        Artery_centerlines/05_branch_tree.vtk Artery_centerlines/09_lobe_reachability.vtk \\
        --lobe RUL=Data/patient_1/upper_lobe_right.vtk \\
        --lobe RML=Data/patient_1/middle_lobe_right.vtk \\
        --lobe RLL=Data/patient_1/lower_lobe_right.vtk \\
        --legend-output Artery_centerlines/09_lobe_reachability.json \\
        --extract-lobe RUL=Artery_centerlines/09_centerline_RUL.vtk
"""

import argparse
import json
import logging
import re
import sys

import vtk

from vescan.io import load_surface, save_surface, detect_coordinate_space, flip_lps_ras, Stage
from vescan.stages.cut_graph import load_graph_json, _extract_cells

logger = logging.getLogger(__name__)

GROUP_ID_ARRAY_NAME = "GroupId"
BLANKED_ARRAY_NAME = "IsBifurcation"
LOBE_MASK_ARRAY_NAME = "LobeMask"
NUM_REACHABLE_ARRAY_NAME = "NumReachableLobes"
LOBE_LABEL_ARRAY_NAME = "LobeLabel"
POINT_LOBE_LABEL_ARRAY_NAME = "LobeLabelPoint"


def parse_lobe_args(raw_args):
    """Parses repeated --lobe/--extract-lobe NAME=PATH options into an
    ordered dict (first-seen order becomes the LobeMask bit order)."""
    lobes = {}
    for raw in raw_args:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=PATH, got '{raw}'")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty lobe NAME in '{raw}'")
        lobes[name] = path.strip()
    return lobes


def count_open_edge_points(polyData):
    """Number of points touching a boundary (open) or non-manifold edge -
    0 means genuinely closed/watertight. Used as our own up-front check
    instead of vtkSelectEnclosedPoints' built-in CheckSurfaceOn(), which
    HARD-fails (aborts Update() entirely, confirmed empirically) rather than
    just warning - not an option for a batch script over several lobes."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polyData)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfPoints()


def load_lobe_surfaces(lobes, target_space):
    """Loads every lobe surface, converting to `target_space` (the
    centerline's coordinate space) when its own file is tagged differently -
    same convention as vescan.stages.clip_vessel/label_surface_branches.py.
    Warns (doesn't fail - see the module docstring) when a surface isn't
    closed/watertight, since containment near the defect becomes unreliable.
    Runs vtkCleanPolyData ONLY as a repair attempt when the raw surface
    isn't already closed - some exporters emit unshared per-face-corner
    points that read as "open" purely from unmerged duplicates (fixed by
    cleaning), but blanket-cleaning an already-closed dense real-world mesh
    can itself introduce a tiny non-manifold edge at a near-self-touching
    region (confirmed empirically on real lobe segmentation data) - so an
    already-good mesh is left untouched."""
    surfaces = {}
    for name, path in lobes.items():
        space = detect_coordinate_space(path) or "LPS"
        polyData = load_surface(path)
        if space != target_space:
            logger.info("Lobe '%s' surface is tagged %s, centerline is %s - converting to match.",
                        name, space, target_space)
            polyData = flip_lps_ras(polyData)

        if count_open_edge_points(polyData) > 0:
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(polyData)
            cleaner.Update()
            polyData = cleaner.GetOutput()

        nOpen = count_open_edge_points(polyData)
        if nOpen > 0:
            logger.warning("lobe '%s' surface is not closed/watertight (%d point(s) on an open/non-manifold "
                            "edge, even after an attempted repair) - containment results near that defect "
                            "may be unreliable.", name, nOpen)
        surfaces[name] = polyData
    return surfaces


def build_enclosed_checker(lobeSurface, tolerance):
    """vtkSelectEnclosedPoints against one lobe surface. Deliberately leaves
    CheckSurfaceOn() off (the default): that built-in validity check HARD-
    fails (aborts Update() with no usable output, confirmed empirically) on
    a surface it decides isn't closed enough, instead of just warning -
    load_lobe_surfaces() already did its own closedness check/repair attempt
    with a graceful warning, so this filter should always run regardless and
    just do the best ray-casting job it can."""
    checker = vtk.vtkSelectEnclosedPoints()
    checker.SetSurfaceData(lobeSurface)
    checker.SetTolerance(tolerance)
    return checker


def collect_group_points(combinedPolyData):
    """group_id -> every point from every cell instance sharing that GroupId
    in the combined model (a group can have more than one geometrically-
    distinct candidate cell - see vescan.stages.build_graph's module
    docstring), plus group_id -> blanked flag."""
    groupIdArray = combinedPolyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    blankedArray = combinedPolyData.GetCellData().GetArray(BLANKED_ARRAY_NAME)

    pointsByGroup = {}
    blankedByGroup = {}
    for cellId in range(combinedPolyData.GetNumberOfCells()):
        groupId = int(groupIdArray.GetValue(cellId))
        blankedByGroup[groupId] = bool(blankedArray.GetValue(cellId)) if blankedArray else False
        cellPointIds = combinedPolyData.GetCell(cellId).GetPointIds()
        positions = pointsByGroup.setdefault(groupId, [])
        for i in range(cellPointIds.GetNumberOfIds()):
            positions.append(combinedPolyData.GetPoint(cellPointIds.GetId(i)))
    return pointsByGroup, blankedByGroup


def compute_direct_lobes(pointsByGroup, blankedByGroup, lobeSurfaces, tolerance, threshold):
    """Stage 1 (spatial SEEDING only - see module docstring): for every
    non-blanked ("branch") group, the set of lobes at least `threshold`
    fraction of its own points fall inside. Blanked (bifurcation) groups are
    never directly assigned - a junction has no lobe of its own, it only
    inherits reachable lobes from its children in propagate_reachability().
    One batched vtkSelectEnclosedPoints call per lobe (all branch groups'
    points at once), not one call per group - much cheaper, and the locator
    it builds is over the (small) lobe surface either way."""
    branchGroupIds = [gid for gid, blanked in blankedByGroup.items() if not blanked]

    fractionsByGroup = {gid: {} for gid in branchGroupIds}
    for lobeName, lobeSurface in lobeSurfaces.items():
        checker = build_enclosed_checker(lobeSurface, tolerance)

        queryPoints = vtk.vtkPoints()
        offsets = []  # (group_id, start_index, point_count)
        for gid in branchGroupIds:
            positions = pointsByGroup.get(gid, [])
            start = queryPoints.GetNumberOfPoints()
            for p in positions:
                queryPoints.InsertNextPoint(p)
            offsets.append((gid, start, len(positions)))
        if queryPoints.GetNumberOfPoints() == 0:
            continue

        queryPolyData = vtk.vtkPolyData()
        queryPolyData.SetPoints(queryPoints)
        checker.SetInputData(queryPolyData)
        checker.Update()
        insideArray = checker.GetOutput().GetPointData().GetArray("SelectedPoints")

        for gid, start, count in offsets:
            if count == 0:
                continue
            insideCount = sum(int(insideArray.GetValue(start + i)) for i in range(count))
            fractionsByGroup[gid][lobeName] = insideCount / count

    # >= 1 point strictly required (not just fraction >= threshold, which a
    # 0.0 threshold would trivially satisfy for a group with 0 points inside)
    # - real segmented arterial trees often stop well short of the pleura, so
    # only a branch's distal-most portion may ever cross into its lobe;
    # requiring a majority is too strict (see module docstring: threshold
    # default is deliberately low for this reason).
    directLobes = {
        gid: {lobeName for lobeName, fraction in fractions.items() if fraction > 0 and fraction >= threshold}
        for gid, fractions in fractionsByGroup.items()
    }
    return directLobes, fractionsByGroup


def build_downstream_adjacency(nodes, edges):
    downstream = {groupId: [] for groupId in nodes}
    for edge in edges:
        downstream.setdefault(edge["parent"], []).append(edge["child"])
    return downstream


def find_disconnected_fragment_ids(nodes, downstream, roots):
    """Group ids belonging to every connected component EXCEPT the one with
    the greatest total centerline length - i.e. small disconnected
    fragments: a root of its own, with no real link to the actual tree
    (confirmed empirically - e.g. a single-node ~1mm root sitting in the
    mediastinum). Stage 1 would otherwise seed these directly to whatever
    lobe they happen to sit next to purely by spatial proximity, punching a
    wrongly-colored hole into what should be a uniform main-trunk region -
    they carry no blood to or from the real tree, so they get no
    classification at all instead. Same "largest connected component wins"
    convention vescan.stages.preprocess already applies to the raw
    surface, one graph level up (root cause is likely upstream centerline-
    extraction noise, not something to special-case per vessel type here).

    These per-root "components" are reachability sets, and they OVERLAP: an
    orphan stub root frequently points straight at a hub that the main root
    also reaches, so the hub and its entire subtree belong to both. Anything
    the main component reaches is therefore subtracted back out at the end -
    it is part of the real tree by definition, no matter what else happens
    to reach it too. Without that subtraction a single sub-millimetre stub
    could silently strip its target's whole subtree of any classification:
    confirmed on real data (an airway tree where the dropped 0-length root
    75 pointed at hub 123, dragging 44 groups / 867mm - 20% of the tree, and
    45 of its 84 nodes - out of the classification as bogus "fragments")."""
    components = []
    for root in roots:
        seen = set()
        stack = [root]
        totalLength = 0.0
        while stack:
            groupId = stack.pop()
            if groupId in seen:
                continue
            seen.add(groupId)
            node = nodes.get(groupId)
            if node:
                totalLength += node.get("length", 0.0)
            for child in downstream.get(groupId, []):
                if child not in seen:
                    stack.append(child)
        components.append((totalLength, seen))

    if len(components) <= 1:
        return set()
    components.sort(key=lambda c: c[0], reverse=True)
    mainIds = components[0][1]
    fragmentIds = set()
    for _length, ids in components[1:]:
        fragmentIds |= ids
    return fragmentIds - mainIds


def propagate_reachability(roots, downstream, directLobes):
    """Stage 2 - the actual reachability algorithm (see module docstring):
    an iterative (stack-based, not recursive - real trees here run tens of
    generations deep) post-order walk of the downstream tree, children
    before parents, so each group's reachable lobes = its own direct
    lobe(s) unioned with every child's already-computed reachable lobes.
    Groups never reached from a root (disconnected fragments) simply never
    appear in the result and read back as "reaches no lobe"."""
    order = []
    visited = set()
    stack = [(root, False) for root in roots]
    while stack:
        groupId, expanded = stack.pop()
        if expanded:
            order.append(groupId)
            continue
        if groupId in visited:
            continue
        visited.add(groupId)
        stack.append((groupId, True))
        for child in downstream.get(groupId, []):
            if child not in visited:
                stack.append((child, False))

    reachable = {}
    for groupId in order:
        lobes = set(directLobes.get(groupId, ()))
        for child in downstream.get(groupId, []):
            lobes |= reachable.get(child, set())
        reachable[groupId] = lobes
    return reachable


def classify_group(lobes):
    if not lobes:
        return "none"
    if len(lobes) == 1:
        return f"exclusive:{next(iter(lobes))}"
    return "shared:" + "+".join(sorted(lobes))


def compute_lobe_labels(lobeOrder, ownLobes, downstream, roots, fragmentIds):
    """A compact 0..len(lobeOrder)+1 classification per group, PURELY LOCAL/
    positional (confirmed on real patient data that the previous subtree-
    union approach was wrong: a group whose OWN points sit 100% inside one
    lobe would still be painted as a shared trunk just because some
    unrelated-looking descendant 10 generations downstream happened to
    cross into another lobe - see the module's git history/conversation for
    the concrete example, group 344-350 in a real artery tree).

    Rule, walking the tree top-down (root to leaves):
      - a group whose own points sit in >1 lobe (own_lobes has >1 entry -
        note this is now tested for EVERY group including bifurcations, not
        just branches like directLobes/compute_direct_lobes() does for the
        separate reachability computation) IS a crossing point, and gets
        highlighted (label N+2) - so does every one of its descendants,
        regardless of what THEIR own points say, because once the vessel
        has crossed a lobe boundary the whole path past that point stays
        flagged rather than quietly reverting to the destination lobe's
        plain color.
      - otherwise: own_lobes empty -> trunk (label N+1, "not inside any
        lobe" - the only thing that still earns plain grey); own_lobes
        exactly one lobe -> that lobe's own color (label 1..N).
    fragmentIds (find_disconnected_fragment_ids()) are always forced to 0
    ("none") regardless of what their own points say - see that function's
    docstring for why a disconnected fragment's spatial position is
    meaningless here.
    Values: 0 = none, 1..N = exclusive to lobeOrder[label-1], N+1 = trunk
    (outside every lobe), N+2 = crossing point or downstream of one."""
    n = len(lobeOrder)
    labels = {}
    tainted = {}
    stack = list(roots)
    visited = set()
    while stack:
        groupId = stack.pop()
        if groupId in visited:
            continue
        visited.add(groupId)
        own = ownLobes.get(groupId, set())
        isCrossing = len(own) > 1
        isTainted = tainted.get(groupId, False) or isCrossing

        if isTainted:
            labels[groupId] = n + 2
        elif not own:
            labels[groupId] = n + 1
        else:
            labels[groupId] = 1 + lobeOrder.index(next(iter(own)))

        for child in downstream.get(groupId, []):
            if child not in visited:
                tainted[child] = tainted.get(child, False) or isTainted
                stack.append(child)

    for groupId in fragmentIds:
        labels[groupId] = 0
    return labels


def collect_fragment_point_ids(combinedPolyData, fragmentIds):
    """Point ids touched by any cell belonging to a disconnected-fragment
    group (find_disconnected_fragment_ids()) - excluded from point-level
    lobe classification the same way fragmentIds already excludes them at
    the group level (see compute_lobe_labels())."""
    if not fragmentIds:
        return set()
    groupIdArray = combinedPolyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    pointIds = set()
    for cellId in range(combinedPolyData.GetNumberOfCells()):
        if int(groupIdArray.GetValue(cellId)) in fragmentIds:
            cellPointIds = combinedPolyData.GetCell(cellId).GetPointIds()
            for i in range(cellPointIds.GetNumberOfIds()):
                pointIds.add(cellPointIds.GetId(i))
    return pointIds


def compute_point_lobe_labels(combinedPolyData, lobeSurfaces, lobeOrder, tolerance, fragmentPointIds):
    """Point-level analogue of compute_lobe_labels(): tests EVERY centerline
    point directly against every lobe surface, instead of aggregating a
    fraction over all of a GroupId's points the way compute_direct_lobes()
    does - so a vessel entering a lobe partway through a branch is marked
    exactly at the point where it crosses, not the whole branch uniformly.
    One vtkSelectEnclosedPoints call per lobe over ALL of combinedPolyData's
    points at once (cheap - same locator-reuse rationale as
    compute_direct_lobes()).

    Values: 0 = none (disconnected fragment, see fragmentPointIds), 1..N =
    exclusively inside lobeOrder[label-1], N+1 = trunk (this exact point is
    outside every lobe), N+2 = this exact point simultaneously tests inside
    more than one lobe surface. Unlike LobeLabel's N+2 (compute_lobe_labels,
    a taint propagated to every downstream group once ANY of its own points
    crossed), this N+2 is a literal, non-propagated double-containment of a
    single point - real pulmonary lobes don't overlap, so it is expected to
    be rare/thin, occurring only where two adjacent lobe surfaces
    geometrically overlap right at a fissure. The actual entry/exit
    boundary of a crossing instead shows up as a plain 1..N/N+1 value
    changing between consecutive points along the vessel - walk the
    centerline's point order (and arc length) to locate and measure it."""
    numPoints = combinedPolyData.GetNumberOfPoints()
    ownLobesByPoint = [set() for _ in range(numPoints)]

    for lobeName, lobeSurface in lobeSurfaces.items():
        checker = build_enclosed_checker(lobeSurface, tolerance)
        checker.SetInputData(combinedPolyData)
        checker.Update()
        insideArray = checker.GetOutput().GetPointData().GetArray("SelectedPoints")
        for pointId in range(numPoints):
            if insideArray.GetValue(pointId):
                ownLobesByPoint[pointId].add(lobeName)

    n = len(lobeOrder)
    labelArray = vtk.vtkIntArray()
    labelArray.SetName(POINT_LOBE_LABEL_ARRAY_NAME)
    labelArray.SetNumberOfValues(numPoints)
    for pointId in range(numPoints):
        if pointId in fragmentPointIds:
            labelArray.SetValue(pointId, 0)
            continue
        own = ownLobesByPoint[pointId]
        if len(own) > 1:
            labelArray.SetValue(pointId, n + 2)
        elif not own:
            labelArray.SetValue(pointId, n + 1)
        else:
            labelArray.SetValue(pointId, 1 + lobeOrder.index(next(iter(own))))
    return labelArray


def sanitize_array_suffix(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def build_reachability_arrays(combinedPolyData, reachableLobes, lobeLabels, lobeOrder):
    """One cell array per (LobeMask, NumReachableLobes, LobeLabel, Reaches_<LOBE>
    x N), aligned 1:1 with combinedPolyData's own cells via its GroupId array -
    broadcasts each group's classification to every candidate-cell instance
    of that group."""
    groupIdArray = combinedPolyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    numCells = combinedPolyData.GetNumberOfCells()

    maskArray = vtk.vtkIntArray()
    maskArray.SetName(LOBE_MASK_ARRAY_NAME)
    maskArray.SetNumberOfValues(numCells)

    numArray = vtk.vtkIntArray()
    numArray.SetName(NUM_REACHABLE_ARRAY_NAME)
    numArray.SetNumberOfValues(numCells)

    perLobeArrays = {}
    for lobeName in lobeOrder:
        arr = vtk.vtkIntArray()
        arr.SetName(f"Reaches_{sanitize_array_suffix(lobeName)}")
        arr.SetNumberOfValues(numCells)
        perLobeArrays[lobeName] = arr

    labelArray = vtk.vtkIntArray()
    labelArray.SetName(LOBE_LABEL_ARRAY_NAME)
    labelArray.SetNumberOfValues(numCells)

    for cellId in range(numCells):
        groupId = int(groupIdArray.GetValue(cellId))
        lobes = reachableLobes.get(groupId, set())
        mask = 0
        for bit, lobeName in enumerate(lobeOrder):
            if lobeName in lobes:
                mask |= (1 << bit)
        maskArray.SetValue(cellId, mask)
        numArray.SetValue(cellId, len(lobes))
        labelArray.SetValue(cellId, lobeLabels.get(groupId, 0))
        for lobeName in lobeOrder:
            if lobeName not in lobes:
                value = 0
            elif len(lobes) == 1:
                value = 1
            else:
                value = 2
            perLobeArrays[lobeName].SetValue(cellId, value)

    return maskArray, numArray, labelArray, perLobeArrays


def build_legend_data(lobeOrder, nodes, directLobes, reachableLobes, fractions,
                       ownLobes, ownFractions, lobeLabels, threshold):
    """Assembles the same per-group legend dict save_legend_json() writes to
    disk, as a plain in-memory value - so a caller running lobe_reachability
    and vescan.stages.anatomical_segments back to back in the same
    process (see main.py) can hand this straight to
    anatomical_segments.classify_and_save() instead of writing it to JSON
    and immediately reading it back."""
    return {
        "lobe_order": lobeOrder,
        "lobe_bit": {lobeName: bit for bit, lobeName in enumerate(lobeOrder)},
        "containment_threshold": threshold,
        "lobe_label_legend": {
            "0": "none",
            **{str(i + 1): f"exclusive:{lobeName}" for i, lobeName in enumerate(lobeOrder)},
            str(len(lobeOrder) + 1): "trunk (this group's own points are outside every lobe)",
            str(len(lobeOrder) + 2): "crossing (this group's own points touch >1 lobe, or it is "
                                      "downstream of a group that does)",
        },
        "lobe_label_point_legend": {
            "0": "none (disconnected fragment)",
            **{str(i + 1): f"exclusive:{lobeName}" for i, lobeName in enumerate(lobeOrder)},
            str(len(lobeOrder) + 1): "trunk (this exact point is outside every lobe)",
            str(len(lobeOrder) + 2): "overlap (this exact point simultaneously tests inside more than "
                                      "one lobe surface - not propagated downstream like the group-level "
                                      "'crossing' above; expected to be rare, only at fissure overlaps). "
                                      "The point where a vessel actually enters/leaves a lobe instead shows "
                                      "up as a plain value change between consecutive points along the "
                                      f"'{POINT_LOBE_LABEL_ARRAY_NAME}' point array.",
        },
        "groups": {
            str(groupId): {
                "blanked": node["blanked"],
                "generation": node["generation"],
                "direct_lobes": sorted(directLobes.get(groupId, ())),
                "lobe_fractions": fractions.get(groupId, {}),
                "reachable_lobes": sorted(reachableLobes.get(groupId, ())),
                "classification": classify_group(reachableLobes.get(groupId, set())),
                "own_lobes": sorted(ownLobes.get(groupId, ())),
                "own_lobe_fractions": ownFractions.get(groupId, {}),
                "lobe_label": lobeLabels.get(groupId, 0),
            }
            for groupId, node in nodes.items()
        },
    }


def save_legend_json(output_path, lobeOrder, nodes, directLobes, reachableLobes, fractions,
                      ownLobes, ownFractions, lobeLabels, threshold):
    data = build_legend_data(lobeOrder, nodes, directLobes, reachableLobes, fractions,
                              ownLobes, ownFractions, lobeLabels, threshold)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_statistics(nodes, reachableLobes, lobeOrder):
    nUnclassified = sum(1 for groupId in nodes if not reachableLobes.get(groupId))
    nExclusive = sum(1 for groupId in nodes if len(reachableLobes.get(groupId, ())) == 1)
    nShared = sum(1 for groupId in nodes if len(reachableLobes.get(groupId, ())) > 1)

    logger.info("=== Lobe reachability statistics ===")
    logger.info("Groups: %d total", len(nodes))
    logger.info("  %d lead to exactly one lobe", nExclusive)
    logger.info("  %d are shared trunks (lead to more than one lobe)", nShared)
    logger.info("  %d reach no lobe (disconnected fragment, or genuinely extra-lobar - e.g. the main "
                "trunk before the first lobar bifurcation)", nUnclassified)
    for lobeName in lobeOrder:
        n = sum(1 for groupId in nodes if lobeName in reachableLobes.get(groupId, ()))
        logger.info("  %d group(s) lead to '%s'", n, lobeName)


def run(graph_json_path, combined_model_path, output_path, lobes,
        containment_threshold=0.1, tolerance=1e-4,
        legend_output_path=None, extract_lobe_paths=None):
    """Returns (outputPolyData, reachableLobes, legend_data, coordinateSpace) -
    legend_data (see build_legend_data()) and coordinateSpace are returned
    even when legend_output_path is None, so a caller can chain straight into
    vescan.stages.anatomical_segments.classify_and_save() without a
    JSON round-trip (see main.py)."""
    extract_lobe_paths = extract_lobe_paths or {}
    lobeOrder = list(lobes.keys())
    for lobeName in extract_lobe_paths:
        if lobeName not in lobeOrder:
            raise ValueError(f"--extract-lobe '{lobeName}' was not also passed as --lobe {lobeName}=PATH")

    nodes, edges, roots = load_graph_json(graph_json_path)
    downstream = build_downstream_adjacency(nodes, edges)

    coordinateSpace = detect_coordinate_space(combined_model_path) or "LPS"
    with Stage("Loading combined model"):
        combinedPolyData = load_surface(combined_model_path)
    if combinedPolyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME) is None:
        raise ValueError(f"{combined_model_path} has no '{GROUP_ID_ARRAY_NAME}' cell array - pass "
                          f"vescan.stages.build_graph's --split-output (05_branch_tree.vtk)")

    logger.info("Loading %d lobe surface(s)...", len(lobes))
    lobeSurfaces = load_lobe_surfaces(lobes, coordinateSpace)

    pointsByGroup, blankedByGroup = collect_group_points(combinedPolyData)

    fragmentIds = find_disconnected_fragment_ids(nodes, downstream, roots)
    if fragmentIds:
        fragmentLength = sum(nodes[gid].get("length", 0.0) for gid in fragmentIds if gid in nodes)
        logger.info("Ignoring %d group(s) (%.1fmm total) in %d disconnected fragment(s) separate from the "
                    "main tree - not eligible for lobe classification regardless of spatial containment "
                    "(see find_disconnected_fragment_ids()).", len(fragmentIds), fragmentLength, len(roots) - 1)
        for gid in fragmentIds:
            blankedByGroup[gid] = True

    with Stage("Testing branch points for lobe containment"):
        directLobes, fractions = compute_direct_lobes(pointsByGroup, blankedByGroup, lobeSurfaces,
                                                        tolerance, containment_threshold)
    nDirect = sum(1 for lobes_ in directLobes.values() if lobes_)
    logger.info("%d/%d branch group(s) directly seeded to a lobe (>= %s of points inside).",
                nDirect, len(directLobes), f"{containment_threshold:.0%}")

    reachableLobes = propagate_reachability(roots, downstream, directLobes)
    print_statistics(nodes, reachableLobes, lobeOrder)

    # LobeLabel (the coloring scheme) needs to know whether EVERY group - not
    # just branches - sits inside a lobe, including bifurcations, so it can
    # tell a bifurcation truly outside every lobe (trunk) from one already
    # deep inside a single lobe (see compute_lobe_labels()'s docstring for
    # why the old subtree-union approach painted large stretches of clearly
    # single-lobe vessel as trunk purely because of a downstream crossing).
    # fragmentIds stays blanked=True here too, so disconnected fragments are
    # excluded from this pass exactly like they are from directLobes.
    testEveryoneMap = {gid: (gid in fragmentIds) for gid in blankedByGroup}
    with Stage("Testing every group's (including bifurcations) points for lobe containment"):
        ownLobes, ownFractions = compute_direct_lobes(pointsByGroup, testEveryoneMap, lobeSurfaces,
                                                        tolerance, containment_threshold)
    lobeLabels = compute_lobe_labels(lobeOrder, ownLobes, downstream, roots, fragmentIds)

    fragmentPointIds = collect_fragment_point_ids(combinedPolyData, fragmentIds)
    with Stage("Testing every centerline point directly for lobe containment"):
        pointLabelArray = compute_point_lobe_labels(combinedPolyData, lobeSurfaces, lobeOrder,
                                                      tolerance, fragmentPointIds)

    maskArray, numArray, labelArray, perLobeArrays = build_reachability_arrays(
        combinedPolyData, reachableLobes, lobeLabels, lobeOrder)
    outputPolyData = vtk.vtkPolyData()
    outputPolyData.DeepCopy(combinedPolyData)
    outputPolyData.GetCellData().AddArray(maskArray)
    outputPolyData.GetCellData().AddArray(numArray)
    outputPolyData.GetCellData().AddArray(labelArray)
    for arr in perLobeArrays.values():
        outputPolyData.GetCellData().AddArray(arr)
    outputPolyData.GetPointData().AddArray(pointLabelArray)

    save_surface(outputPolyData, output_path, coordinate_space=coordinateSpace)
    arrayNames = [NUM_REACHABLE_ARRAY_NAME, LOBE_MASK_ARRAY_NAME, LOBE_LABEL_ARRAY_NAME] + \
        [arr.GetName() for arr in perLobeArrays.values()]
    logger.info("Saved labeled centerline (%d cells, new cell arrays %s; %d points, new point array %s) to %s",
                outputPolyData.GetNumberOfCells(), arrayNames,
                outputPolyData.GetNumberOfPoints(), POINT_LOBE_LABEL_ARRAY_NAME, output_path)

    legendData = build_legend_data(lobeOrder, nodes, directLobes, reachableLobes, fractions,
                                    ownLobes, ownFractions, lobeLabels, containment_threshold)
    if legend_output_path:
        with open(legend_output_path, "w", encoding="utf-8") as f:
            json.dump(legendData, f, indent=2)
        logger.info("Saved legend/per-group JSON to %s", legend_output_path)

    groupIdArray = outputPolyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    for lobeName, extractPath in extract_lobe_paths.items():
        keptGroupIds = {groupId for groupId in nodes if lobeName in reachableLobes.get(groupId, ())}
        keptCellIds = [cellId for cellId in range(outputPolyData.GetNumberOfCells())
                       if int(groupIdArray.GetValue(cellId)) in keptGroupIds]
        subset = _extract_cells(outputPolyData, keptCellIds)
        save_surface(subset, extractPath, coordinate_space=coordinateSpace)
        logger.info("Saved '%s' feeding-artery subset centerline (%d group(s), %d cell(s), own branches + "
                    "shared upstream trunks) to %s - has the same '%s' cell array as "
                    "vescan.stages.cut_graph's output, so it can be fed straight into "
                    "vescan.stages.clip_vessel's centerline arguments.",
                    lobeName, len(keptGroupIds), subset.GetNumberOfCells(), extractPath, GROUP_ID_ARRAY_NAME)

    return outputPolyData, reachableLobes, legendData, coordinateSpace


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph_json", help="Path to build_graph's --graph-output "
                                            "(05_branch_tree_topology.json) - topology: per-group "
                                            "generation/blanked and the parent->child GroupId edges")
    parser.add_argument("combined_model", help="Path to build_graph's --split-output "
                                                "(05_branch_tree.vtk) - geometry: one cell per group/candidate "
                                                "cell with a 'GroupId' cell array")
    parser.add_argument("output", help="Path to save the labeled combined centerline (.vtk/.vtp) - same cells/"
                                        "order as combined_model, with new reachability cell arrays added")
    parser.add_argument("--lobe", action="append", required=True, default=[], metavar="NAME=PATH",
                         help="One lobe surface (.vtk/.vtp), repeatable - e.g. --lobe RUL=upper_lobe_right.vtk. "
                              "NAME becomes the LobeMask bit and the Reaches_<NAME> array suffix, in first-seen "
                              "order. Needs at least 2 lobes for 'shared trunk' detection to ever fire.")
    parser.add_argument("--containment-threshold", type=float, default=0.1,
                         help="Minimum fraction (0-1) of a branch group's own points that must fall inside a lobe "
                              "surface for that group to be directly seeded to it - always requires at least 1 "
                              "point inside regardless of this value (default 0.1, deliberately low - see the "
                              "module docstring's two-stage algorithm)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                         help="vtkSelectEnclosedPoints tolerance (default 1e-4)")
    parser.add_argument("--legend-output", default=None,
                         help="Optional path to save a JSON legend: lobe bit order, plus per-group direct/"
                              "reachable lobes and classification")
    parser.add_argument("--extract-lobe", action="append", default=[], metavar="NAME=PATH",
                         help="Optional, repeatable: also save a standalone subset centerline (.vtk/.vtp, same "
                              "'GroupId' cell-array convention as cut_graph's output) containing "
                              "only the groups that reach lobe NAME (own branches + shared upstream trunks). NAME "
                              "must also have been passed via --lobe")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    lobes = parse_lobe_args(args.lobe)
    extractLobePaths = parse_lobe_args(args.extract_lobe)

    run(
        args.graph_json, args.combined_model, args.output, lobes,
        containment_threshold=args.containment_threshold,
        tolerance=args.tolerance,
        legend_output_path=args.legend_output,
        extract_lobe_paths=extractLobePaths,
    )


if __name__ == "__main__":
    sys.exit(main())