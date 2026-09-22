#!/usr/bin/env python3
"""
What a "crossing vessel" IS - the single owner of that definition, imported
by both vescan.stages.anatomical_segments (which labels them on the
centerline) and vescan.stages.statistics (which reports them). It
lived inside statistics.py until both stages needed it; nothing here writes
files.

TWO KINDS OF CROSSING
---------------------
Not every vessel carrying the crossing label actually transects a lobe
boundary. Measured against the interlobar fissure sheets
(vescan.stages.build_lobe_segments), two distinct behaviours come out
of real data, and they are different anatomical objects:

  TRANSLOBAR  goes through the fissure and stays on the other side. Its
              piercings are steep - it meets the fissure plane head-on.
  INTERLOBAR  runs ALONG the fissure, drifting in and out of the neighbouring
              lobe. Several piercings, all shallow: it grazes the sheet
              rather than transecting it.

Both count as ONE crossing vessel. The interlobar one is not several
crossings that happen to share a branch - it is one vessel travelling in the
interlobar plane, and its multiple contact points are a property worth
measuring, not an artifact to collapse.

THE RULE, and the data that fixed it (13 crossing vessels, 3 patients):

    interlobar  <=>  at least MIN_PIERCINGS_FOR_INTERLOBAR piercings
                     AND median incidence angle < INTERLOBAR_MAX_MEDIAN_ANGLE_DEG

                        piercings   median angle
      interlobar             5           15.0
      interlobar             2            4.1
      interlobar             4            1.6
      ---------------------------------------- threshold sits here
      translobar             2           39.4      angles [13.9, 64.9]
      translobar             1       18.7 .. 79.6

A single piercing is translobar by construction: the vessel went through
once and stayed. The threshold only ever decides multi-piercing cases, where
the margin is 15.0 vs 39.4 - a factor of 2.6, so 30 deg is not a tuned
number, it is the middle of a wide gap.

WHAT WAS TRIED AND REJECTED: "fraction of the crossing's length running
within 5 mm of the fissure" sounds like the natural measure of "runs along
it" and does NOT separate the two - a short translobar crossing scores 100%
simply by being short, while the longest interlobar one scores 19.6% by
wandering. Measured, discarded; the angle is the discriminator.

ILL-CONDITIONING, stated because it is real: at grazing incidence the
tangent direction is indeterminate, and the same piercing moves between 5.7
and 0.8 deg depending on the window used. Above ~15 deg the angle is stable
to within a few degrees across every window/patch radius tried. Hence the
median (not the max, not a single piercing's value) drives the decision, and
ANGLE_TANGENTIAL_DEG marks the band below which an angle should be read as
"grazing", not as a precise number.

LENGTHS - three of them, because "how long is this crossing" has three
different honest answers and collapsing them loses information:

  length_mm                 total crossing-labelled centerline: the WHOLE
                            downstream subtree, every branch summed.
  length_to_end_mm          longest single path from where the crossing label
                            starts down to a distal leaf. "How far does it
                            still go", not "how much vessel is there".
  length_after_piercing_mm  the same, but anchored at the fissure itself -
                            the deepest reach measured from a piercing point.

They diverge a lot once a crossing branches: R01-091's largest venous
crossing totals 402.6mm over 58 groups and 20 leaves, while its longest
single path is 86.1mm. The first number is not wrong, it just answers
"how much vasculature is downstream of this crossing".

The last two differ because the label and the fissure are different anchors.
Usually length_after_piercing_mm is much SMALLER: most piercings don't even
land on the run's root group, so by the time the vessel actually meets the
sheet much of the labelled run is already behind it (R01-137's largest
venous crossing: 69.9mm to the end, only 10.2mm after its piercings).

It can also come out slightly LARGER - by a few tenths of a millimetre -
and that is not an inconsistency: LobeLabelPoint flips where a point stops
testing inside the origin lobe, which can fall just after the centerline
crosses the sheet, leaving the piercing marginally upstream of the label.
Those two boundaries are the same anatomical surface reached by different
routes (lobe containment vs. the extracted fissure mesh), so they agree to
within the discretisation, not exactly.

ANGLE CONVENTION: to the fissure PLANE, 0-90 deg, unsigned. 0 = runs along
the fissure, 90 = pierces perpendicular. Unsigned deliberately: the fissure
sheets are extracted from one lobe's side, so their normals are not
consistently oriented between the four contacts, and a signed angle would
silently depend on which lobe happened to be "A" in that pair.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import vtk
from vtk.util import numpy_support as ns

from vescan.lobes import LOBE_ORDER

logger = logging.getLogger(__name__)

GROUP_ID_ARRAY_NAME = "GroupId"
GENERATION_ARRAY_NAME = "Generation"
RADIUS_ARRAY_NAME = "Radius"
POINT_ANATOMICAL_SEGMENT_ARRAY_NAME = "AnatomicalSegmentPoint"
LOBE_LABEL_POINT_ARRAY_NAME = "LobeLabelPoint"

# Written onto the centerline by anatomical_segments - see
# build_crossing_arrays(). CrossingId makes the grouping visible (every group
# of one crossing vessel shares an id, so an interlobar vessel reads as ONE
# object in Slicer instead of a string of separately-coloured branches).
CROSSING_ID_ARRAY_NAME = "CrossingId"
CROSSING_TYPE_ARRAY_NAME = "CrossingType"
CROSSING_ID_POINT_ARRAY_NAME = "CrossingIdPoint"
CROSSING_TYPE_POINT_ARRAY_NAME = "CrossingTypePoint"

CROSSING_TYPE_NONE = 0
CROSSING_TYPE_INTERLOBAR = 1
CROSSING_TYPE_TRANSLOBAR = 2
# Labelled as crossing, but its centerline never meets a fissure sheet. Real
# and expected near the periphery and the hilum, where the lobes stop
# touching and so no fissure exists to pierce - reported as its own kind
# rather than forced into one of the two above.
CROSSING_TYPE_UNCLASSIFIED = 3

CROSSING_TYPE_NAME = {
    CROSSING_TYPE_NONE: "none",
    CROSSING_TYPE_INTERLOBAR: "interlobar",
    CROSSING_TYPE_TRANSLOBAR: "translobar",
    CROSSING_TYPE_UNCLASSIFIED: "unclassified",
}

INTERLOBAR_MAX_MEDIAN_ANGLE_DEG = 30.0
MIN_PIERCINGS_FOR_INTERLOBAR = 2
# Below this, report the angle as grazing rather than as a precise value -
# see the ill-conditioning note in the module docstring.
ANGLE_TANGENTIAL_DEG = 10.0

# Arclength half-window for the local tangent. Long enough to average out
# the ~0.7mm point spacing, short enough that a curving vessel's direction at
# the piercing is not smeared: angles moved by <6 deg between 2 and 10mm on
# every non-grazing piercing measured.
TANGENT_WINDOW_MM = 5.0
# Radius of the fissure patch plane-fitted for the local normal. The GLOBAL
# plane fit of a fissure leaves 1.4-5.8mm RMS; this patch leaves 0.06-0.31mm,
# i.e. locally the sheet really is planar and the global plane is not the
# object to use.
FISSURE_PATCH_RADIUS_MM = 5.0
MIN_PATCH_POINTS = 12
# Two piercings closer than this are the same crossing reported twice, from
# the two centerline segments meeting at a shared point.
DUPLICATE_PIERCING_MM = 0.1


def cell_array_values(polyData, name):
    array = polyData.GetCellData().GetArray(name)
    if array is None:
        return None
    return [array.GetValue(i) for i in range(array.GetNumberOfTuples())]


def representative_cells(polyData):
    """group_id -> its cell with the most points, the same representative
    build_graph.build_group_data() uses.

    A group's other cells are the SAME branch traced by other centerlines;
    walking them all inflates a tree's length from 9912.7mm to 12761.4mm,
    and would report one piercing several times over."""
    groupIds = cell_array_values(polyData, GROUP_ID_ARRAY_NAME)
    if groupIds is None:
        return {}
    best = {}
    for cellId, rawGroupId in enumerate(groupIds):
        groupId = int(rawGroupId)
        nPoints = polyData.GetCell(cellId).GetNumberOfPoints()
        if groupId not in best or nPoints > best[groupId][1]:
            best[groupId] = (cellId, nPoints)
    return {groupId: cellId for groupId, (cellId, _) in best.items()}


def lobe_name(value):
    """LobeLabelPoint 1..5 -> lobe name; None for 0/6/7 (disconnected,
    outside every lobe, or inside two at a fissure)."""
    value = int(value)
    return LOBE_ORDER[value - 1] if 1 <= value <= len(LOBE_ORDER) else None


def generation_per_group(polyData):
    groupIds = cell_array_values(polyData, GROUP_ID_ARRAY_NAME) or []
    generations = cell_array_values(polyData, GENERATION_ARRAY_NAME) or []
    shallowest = {}
    for cellId, rawGroupId in enumerate(groupIds):
        if cellId >= len(generations):
            break
        generation = int(generations[cellId])
        groupId = int(rawGroupId)
        if generation >= 0 and (groupId not in shallowest or generation < shallowest[groupId]):
            shallowest[groupId] = generation
    return shallowest


def crossing_geometry(polyData, crossingValues):
    """group_id -> {length, points, lobes, origin} for every group holding at
    least one crossing-labeled point, measured on its representative cell.

    crossingValues is a SET, not one value: stage 10 refines the plain
    crossing label into an interlobar and a translobar value (73-78, see
    anatomical_segments.crossing_values_for()), so a file may carry any of
    the three and looking for only one would silently find nothing.

    Length is GEOMETRIC, never a point count: centerline.curve_sampling_distance
    is an upper bound, not a fixed step - measured spacing runs from 0 to
    1.001mm, mean 0.683mm, so one point per millimetre would overstate every
    length by ~45%. A segment counts only when BOTH endpoints carry the label."""
    pointSegments = polyData.GetPointData().GetArray(POINT_ANATOMICAL_SEGMENT_ARRAY_NAME)
    pointLobes = polyData.GetPointData().GetArray(LOBE_LABEL_POINT_ARRAY_NAME)
    if pointSegments is None:
        return {}

    geometry = {}
    for groupId, cellId in representative_cells(polyData).items():
        cell = polyData.GetCell(cellId)
        pointIds = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        flags = [int(pointSegments.GetValue(i)) in crossingValues for i in pointIds]
        if not any(flags):
            continue
        lobes = [lobe_name(pointLobes.GetValue(i)) if pointLobes is not None else None
                 for i in pointIds]

        length = 0.0
        for i in range(len(pointIds) - 1):
            if flags[i] and flags[i + 1]:
                length += math.dist(polyData.GetPoint(pointIds[i]), polyData.GetPoint(pointIds[i + 1]))

        firstCrossing = flags.index(True)
        geometry[groupId] = {
            "cell_id": cellId,
            "length": length,
            "points": sum(1 for flag in flags if flag),
            "lobes": {lobes[i] for i, flag in enumerate(flags) if flag and lobes[i]},
            # The last non-crossing point before the label starts still sits in
            # the lobe the vessel came from, so direction needs no neighbour
            # search: the label starts mid-cell, at the very point
            # LobeLabelPoint changes value.
            "origin": lobes[firstCrossing - 1] if firstCrossing > 0 else None,
        }
    return geometry


def group_crossing_runs(geometry, topology):
    """The crossing-labeled groups joined into vessels: one run per connected
    component of the GROUP GRAPH restricted to crossing groups, longest run
    first (so a run's index is stable between the stages that both call this).

    The label propagates to every group downstream of the one that actually
    straddles a boundary, so counting labeled groups over-reports badly: 75
    groups for 4 real crossings on one patient's veins. Geometry cannot do the
    joining - the centerline's cells share no point ids (not one point of
    28721 belongs to two cells), and rebuilding adjacency by position leaves
    the two traversals of a branch a few tenths of a mm apart at junctions,
    splitting those same 4 crossings into 11. Both approaches agree on total
    length to the millimetre, so the group graph is the safe choice."""
    # Only "edges" is read, never "roots": that list deliberately holds group
    # ids absent from "nodes" (build_graph.repair_orphan_roots keeps orphaned
    # sub-threshold stubs), which would read as hundreds of phantom roots.
    adjacency = {}
    for edge in topology["edges"]:
        adjacency.setdefault(edge["parent"], set()).add(edge["child"])
        adjacency.setdefault(edge["child"], set()).add(edge["parent"])

    runs, visited = [], set()
    for groupId in geometry:
        if groupId in visited:
            continue
        stack, run = [groupId], set()
        while stack:
            current = stack.pop()
            if current in run:
                continue
            run.add(current)
            visited.add(current)
            for neighbour in adjacency.get(current, ()):
                if neighbour in geometry and neighbour not in run:
                    stack.append(neighbour)
        runs.append(run)

    runs.sort(key=lambda run: -sum(geometry[g]["length"] for g in run))
    return runs


class FissureProbe:
    """Local geometry of the interlobar fissure sheets: where a segment
    pierces one, and the plane it pierces it through.

    Built once per structure - the locators are the expensive part."""

    def __init__(self, fissureSurface):
        self.surface = fissureSurface
        self.points = ns.vtk_to_numpy(fissureSurface.GetPoints().GetData())

        self.obbTree = vtk.vtkOBBTree()
        self.obbTree.SetDataSet(fissureSurface)
        self.obbTree.BuildLocator()

        self.kdTree = vtk.vtkKdTree()
        self.kdTree.BuildLocatorFromPoints(fissureSurface.GetPoints())

        contact = fissureSurface.GetCellData().GetArray("FissureContact")
        self.contactPerCell = ns.vtk_to_numpy(contact) if contact else None
        fissure = fissureSurface.GetCellData().GetArray("Fissure")
        self.fissurePerCell = ns.vtk_to_numpy(fissure) if fissure else None

    def intersect(self, pointA, pointB):
        """[(position, fissureCellId)] where segment A->B crosses a sheet."""
        points, cellIds = vtk.vtkPoints(), vtk.vtkIdList()
        if self.obbTree.IntersectWithLine(pointA, pointB, points, cellIds) == 0:
            return []
        return [(np.array(points.GetPoint(i)), cellIds.GetId(i))
                for i in range(points.GetNumberOfPoints())]

    def local_plane(self, position, radius=FISSURE_PATCH_RADIUS_MM):
        """(unitNormal, rmsDeviation, nPoints) of the least-squares plane
        through the fissure vertices within `radius` of `position`, or
        (None, None, n) when too few vertices are in reach to fit one."""
        ids = vtk.vtkIdList()
        self.kdTree.FindPointsWithinRadius(radius, position, ids)
        count = ids.GetNumberOfIds()
        if count < MIN_PATCH_POINTS:
            return None, None, count
        patch = self.points[[ids.GetId(i) for i in range(count)]]
        centroid = patch.mean(axis=0)
        _, _, vt = np.linalg.svd(patch - centroid, full_matrices=False)
        normal = vt[2]
        rms = float(np.sqrt((((patch - centroid) @ normal) ** 2).mean()))
        return normal, rms, count


def _tangent(polyline, index, window=TANGENT_WINDOW_MM):
    """Unit direction of the centerline over +/- `window` mm of arclength
    around `index`. None when the polyline is too short to define one."""
    def walk(step):
        i, travelled = index, 0.0
        while 0 <= i + step < len(polyline) and travelled < window:
            travelled += float(np.linalg.norm(polyline[i + step] - polyline[i]))
            i += step
        return i

    direction = polyline[walk(+1)] - polyline[walk(-1)]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-9 else None


def _incidence_angle_deg(tangent, normal):
    """Angle to the fissure PLANE in [0, 90] - see the module docstring's
    angle convention."""
    cosine = min(1.0, abs(float(np.dot(tangent, normal))))
    return 90.0 - math.degrees(math.acos(cosine))


def longest_downstream_path(geometry, run, children):
    """group_id -> the longest crossing-labelled path from the START of that
    group down to a distal leaf, staying inside `run`.

    "How far does the vessel still go", as opposed to crossing_geometry()'s
    total, which sums the WHOLE downstream subtree. The two answer different
    questions and differ by a lot once a crossing branches: on R01-091's
    veins the largest crossing totals 402.6mm over 58 groups and 20 leaves,
    but its longest single path is 86.1mm."""
    deepest = {}

    def walk(groupId, seen):
        if groupId in deepest:
            return deepest[groupId]
        best = 0.0
        for child in children.get(groupId, ()):
            if child in run and child not in seen:
                best = max(best, walk(child, seen | {child}))
        deepest[groupId] = geometry[groupId]["length"] + best
        return deepest[groupId]

    for groupId in run:
        walk(groupId, {groupId})
    return deepest


def find_piercings(polyData, geometry, run, probe, crossingValues, radiusArray=None):
    """Every point where one crossing vessel's centerline pierces a fissure.

    Walks only the representative cell of each group in `run` (see
    representative_cells()), so a branch traced by several centerlines
    contributes its piercings once."""
    pointSegments = polyData.GetPointData().GetArray(POINT_ANATOMICAL_SEGMENT_ARRAY_NAME)
    piercings = []
    for groupId in sorted(run):
        cellId = geometry[groupId]["cell_id"]
        cell = polyData.GetCell(cellId)
        pointIds = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        polyline = np.array([polyData.GetPoint(i) for i in pointIds])
        labelled = [pointSegments is None or int(pointSegments.GetValue(i)) in crossingValues
                    for i in pointIds]

        def distal_in_cell(fromIndex, fromPosition):
            """Crossing-labelled arclength left in THIS cell after a piercing:
            the remainder of the segment it landed on, plus every labelled
            segment after it. The start of the per-branch 'how far does it
            still go' measure - analyse() adds the downstream subtree."""
            remaining = float(np.linalg.norm(polyline[fromIndex + 1] - fromPosition))
            for k in range(fromIndex + 1, len(polyline) - 1):
                if labelled[k] and labelled[k + 1]:
                    remaining += float(np.linalg.norm(polyline[k + 1] - polyline[k]))
            return remaining

        for i in range(len(polyline) - 1):
            for position, fissureCellId in probe.intersect(polyline[i], polyline[i + 1]):
                if any(float(np.linalg.norm(position - p["position"])) < DUPLICATE_PIERCING_MM
                       for p in piercings):
                    continue
                tangent = _tangent(polyline, i)
                normal, rms, _ = probe.local_plane(position)
                if tangent is None or normal is None:
                    logger.debug("Piercing at %s skipped: no local tangent or fissure patch.", position)
                    continue

                radius = None
                if radiusArray is not None:
                    radius = float(radiusArray.GetValue(pointIds[i]))

                piercings.append({
                    "position": position,
                    "group": groupId,
                    "distal_in_cell_mm": distal_in_cell(i, position),
                    "angle_deg": round(_incidence_angle_deg(tangent, normal), 2),
                    "patch_rms_mm": round(rms, 3),
                    "grazing": _incidence_angle_deg(tangent, normal) < ANGLE_TANGENTIAL_DEG,
                    "fissure_contact": (int(probe.contactPerCell[fissureCellId])
                                        if probe.contactPerCell is not None else None),
                    "fissure": (int(probe.fissurePerCell[fissureCellId])
                                if probe.fissurePerCell is not None else None),
                    "radius_mm": round(radius, 3) if radius is not None else None,
                })
    return piercings


def classify(piercings, min_piercings_for_interlobar=MIN_PIERCINGS_FOR_INTERLOBAR,
             interlobar_max_median_angle_deg=INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """CROSSING_TYPE_* for one crossing vessel - see the module docstring for
    the rule and the data behind the threshold."""
    if not piercings:
        return CROSSING_TYPE_UNCLASSIFIED
    if len(piercings) < min_piercings_for_interlobar:
        return CROSSING_TYPE_TRANSLOBAR
    median = float(np.median([p["angle_deg"] for p in piercings]))
    return CROSSING_TYPE_INTERLOBAR if median < interlobar_max_median_angle_deg \
        else CROSSING_TYPE_TRANSLOBAR


def analyse(polyData, topology, crossingValues, fissureSurface=None,
            min_piercings_for_interlobar=MIN_PIERCINGS_FOR_INTERLOBAR,
            interlobar_max_median_angle_deg=INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """Every crossing vessel of one structure, longest first.

    Returns None when an input needed to answer the question is missing, so a
    caller can say so instead of reporting a misleading zero. Without
    `fissureSurface` the vessels come back with no piercings and type
    `unclassified` - the counts and lengths are unaffected, only the
    interlobar/translobar split needs the fissures."""
    if polyData is None or not topology or not topology.get("edges"):
        return None
    if polyData.GetPointData().GetArray(POINT_ANATOMICAL_SEGMENT_ARRAY_NAME) is None:
        return None

    geometry = crossing_geometry(polyData, crossingValues)
    if not geometry:
        return []

    probe = FissureProbe(fissureSurface) if fissureSurface is not None else None
    radiusArray = polyData.GetPointData().GetArray(RADIUS_ARRAY_NAME)
    generationOf = generation_per_group(polyData)

    # Directed, unlike group_crossing_runs()' adjacency: "downstream" needs to
    # know which end is the parent.
    childrenOf, parentOf = {}, {}
    for edge in topology["edges"]:
        childrenOf.setdefault(edge["parent"], []).append(edge["child"])
        parentOf[edge["child"]] = edge["parent"]

    vessels = []
    for index, run in enumerate(group_crossing_runs(geometry, topology), start=1):
        # Ordered from the run's own ROOT outwards. The root is topological -
        # the group whose parent is outside the run - NOT the shallowest
        # generation: two groups of a run routinely share a generation (638
        # and 752 both sit at 4 on R01-091's veins), and picking by generation
        # then depended on set iteration order, silently starting the walk
        # one group too far down and under-reporting every downstream length.
        # The (generation, id) tie-break only decides between real roots, and
        # keeps the result reproducible run to run.
        roots = [g for g in run if parentOf.get(g) not in run]
        ordered = sorted(roots or run, key=lambda g: (generationOf.get(g, 1 << 30), g)) + \
            sorted(run - set(roots), key=lambda g: (generationOf.get(g, 1 << 30), g))
        # Read the origin from the group where the label actually starts: a
        # deeper group is fully tainted and has no pre-crossing point left.
        fromLobe = next((geometry[g]["origin"] for g in ordered if geometry[g]["origin"]), None)
        toLobes = set()
        for member in run:
            toLobes |= geometry[member]["lobes"]
        entered = sorted(toLobes - {fromLobe}) if fromLobe else sorted(toLobes)

        piercings = find_piercings(polyData, geometry, run, probe, crossingValues,
                                   radiusArray) if probe else []
        crossingType = classify(piercings, min_piercings_for_interlobar=min_piercings_for_interlobar,
                                 interlobar_max_median_angle_deg=interlobar_max_median_angle_deg) \
            if probe else CROSSING_TYPE_UNCLASSIFIED
        angles = [p["angle_deg"] for p in piercings]

        deepest = longest_downstream_path(geometry, run, childrenOf)
        lengthToEnd = deepest.get(ordered[0]) if ordered else None

        # Same measure but anchored at the fissure itself rather than at the
        # label: the crossing label starts UPSTREAM of the actual piercing -
        # measured on real data, most piercings don't even fall on the run's
        # root group - so the two differ by more than rounding.
        for piercing in piercings:
            downstream = max((deepest[child] for child in childrenOf.get(piercing["group"], ())
                              if child in run), default=0.0)
            piercing["distal_length_mm"] = round(piercing.pop("distal_in_cell_mm") + downstream, 3)
        lengthAfterPiercing = max((p["distal_length_mm"] for p in piercings), default=None)

        vessels.append({
            "crossing": index,
            "crossing_type": CROSSING_TYPE_NAME[crossingType],
            "crossing_type_value": crossingType,
            "from_lobe": fromLobe,
            "to_lobe": "+".join(entered or sorted(toLobes)) or None,
            # THREE different lengths, answering three different questions -
            # see the module docstring's LENGTHS section.
            "length_mm": round(sum(geometry[g]["length"] for g in run), 3),
            "length_to_end_mm": round(lengthToEnd, 3) if lengthToEnd is not None else None,
            "length_after_piercing_mm": (round(lengthAfterPiercing, 3)
                                         if lengthAfterPiercing is not None else None),
            "points": sum(geometry[g]["points"] for g in run),
            "groups": sorted(run),
            "n_groups": len(run),
            "start_generation": generationOf.get(ordered[0]) if ordered else None,
            "n_piercings": len(piercings),
            "angle_median_deg": round(float(np.median(angles)), 2) if angles else None,
            "angle_min_deg": round(min(angles), 2) if angles else None,
            "angle_max_deg": round(max(angles), 2) if angles else None,
            "fissures": sorted({p["fissure"] for p in piercings if p["fissure"] is not None}),
            "patch_rms_max_mm": round(max(p["patch_rms_mm"] for p in piercings), 3) if piercings else None,
            "radius_mm": (round(float(np.median([p["radius_mm"] for p in piercings
                                                  if p["radius_mm"] is not None])), 3)
                          if any(p["radius_mm"] is not None for p in piercings) else None),
            "piercings": [{k: v for k, v in p.items() if k != "position"} for p in piercings],
        })
    return vessels


def build_crossing_arrays(polyData, vessels, crossingValues):
    """(cellId, cellType, pointId, pointType) vtkIntArrays tagging every cell
    and point with the crossing vessel it belongs to and that vessel's kind.

    CrossingId is what makes the grouping legible: an interlobar vessel's
    whole run shares ONE id, so it reads as a single object in Slicer instead
    of a string of separately-coloured branches. 0 = not a crossing, in every
    one of the four arrays.

    Ids match the `crossing` column of statistics' CSV, because both come
    from group_crossing_runs()' longest-first order.

    Counting the tagged points will NOT reproduce that CSV's `points` column,
    and shouldn't: these arrays tag EVERY cell of a crossing group, because a
    branch left half-uncoloured in Slicer reads as a bug, while
    crossing_geometry() deliberately measures one representative cell per
    group so that a branch traced by several centerlines is not counted
    several times. Display wants every copy; measurement wants exactly one.
    On R01-091's veins that is 1111 tagged points against 652 measured."""
    numCells, numPoints = polyData.GetNumberOfCells(), polyData.GetNumberOfPoints()
    groupIdArray = polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    pointSegments = polyData.GetPointData().GetArray(POINT_ANATOMICAL_SEGMENT_ARRAY_NAME)

    idByGroup, typeByGroup = {}, {}
    for vessel in vessels or []:
        for groupId in vessel["groups"]:
            idByGroup[groupId] = vessel["crossing"]
            typeByGroup[groupId] = vessel["crossing_type_value"]

    def make(name, size):
        array = vtk.vtkIntArray()
        array.SetName(name)
        array.SetNumberOfValues(size)
        for i in range(size):
            array.SetValue(i, 0)
        return array

    cellIdArray = make(CROSSING_ID_ARRAY_NAME, numCells)
    cellTypeArray = make(CROSSING_TYPE_ARRAY_NAME, numCells)
    pointIdArray = make(CROSSING_ID_POINT_ARRAY_NAME, numPoints)
    pointTypeArray = make(CROSSING_TYPE_POINT_ARRAY_NAME, numPoints)
    if groupIdArray is None:
        return cellIdArray, cellTypeArray, pointIdArray, pointTypeArray

    for cellId in range(numCells):
        groupId = int(groupIdArray.GetValue(cellId))
        if groupId not in idByGroup:
            continue
        cellIdArray.SetValue(cellId, idByGroup[groupId])
        cellTypeArray.SetValue(cellId, typeByGroup[groupId])

        # Points are tagged ONLY where they carry the crossing label: a
        # crossing group's cell also holds the run-up points from before the
        # label starts, and painting those would push the vessel's visible
        # extent upstream of where it actually begins - the same reason
        # crossing_geometry() measures length only between two labelled
        # endpoints.
        pointIds = polyData.GetCell(cellId).GetPointIds()
        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            if pointSegments is not None and int(pointSegments.GetValue(pointId)) not in crossingValues:
                continue
            pointIdArray.SetValue(pointId, idByGroup[groupId])
            pointTypeArray.SetValue(pointId, typeByGroup[groupId])

    return cellIdArray, cellTypeArray, pointIdArray, pointTypeArray
