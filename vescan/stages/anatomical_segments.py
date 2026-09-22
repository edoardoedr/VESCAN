#!/usr/bin/env python3
"""
Labels every branch/bifurcation group of a centerline tree with its precise
medical anatomical segment - THREE tiers, matching real anatomy, not just
left/right:
  Trachea -> Bronco principale dx/sx -> Bronco lobare superiore/medio/...
  Tronco polmonare -> Arteria polmonare dx/sx -> Arteria lobare superiore/...
  (nessun tronco) -> Tronco venoso polmonare dx/sx -> Vena lobare sup/...
sitting "above" vescan.stages.lobe_reachability's per-lobe output, but
reusing its already graph-derived results rather than re-walking the tree,
and folding everything into ONE unified numeric scale across all three
vessel types so they never collide when loaded/colored together.

No new graph traversal needed - lobe_reachability's --legend-output JSON (or,
in-process, its build_legend_data() return value - see classify_and_save())
already stores, per group:
  - `reachable_lobes`: the union, computed bottom-up over graph.json's
    parent->child edges (the actual bifurcation walk), of every lobe
    reachable from that group's downstream subtree. This alone is enough
    for the 3-tier split:
      * empty                                -> 0 (none)
      * exactly ONE lobe                     -> that lobe's own label
        (committed to a single lobe - lobare tier)
      * >1 lobes, all on the SAME side       -> that side's dx/sx label
        (bronco/arteria principale, tronco venoso - not yet committed to
        one lobe, but already committed to a lung)
      * >1 lobes spanning BOTH sides          -> the trunk label (hasn't
        even committed to a lung yet - Trachea/Tronco polmonare). Veins
        have no such tier and are placed on a side instead, per cell -
        see classify_segment()/apply_vein_side_correction().
  - `lobe_label`: lobe_reachability's OWN separate, purely local/positional
    classification (see that module) - value 7 (== len(lobes)+2) marks a
    group whose OWN points cross more than one lobe, or which is downstream
    of one that does. THIS TAKES PRIORITY over the reachable_lobes-based
    tier above: a crossing/ambiguous branch (and everything past it) must
    stay visibly flagged rather than quietly being folded back into a plain
    dx/sx/lobare color - confirmed on real patient data that this is exactly
    the distinction worth keeping (see the earlier conversation's group
    344-350 example).

CROSSINGS ARE SPLIT IN TWO. A vessel carrying the crossing label is either
INTERLOBAR (runs along the fissure, grazing it) or TRANSLOBAR (transects it
and stays on the other side) - see vescan.crossings for the rule and
the data behind it. Both kinds get their OWN value in AnatomicalSegment and
AnatomicalSegmentPoint (73-78, the refined block below), so one Active Scalar
in Slicer shows the anatomical segment AND which kind of crossing it makes,
instead of one undifferentiated colour over both.

Unified label table: each --vessel-type owns a fixed block of 9 values -
trunk, destro, sinistro, the 5 lobes (in vescan.lobes.LOBE_ORDER),
then crossing:
  - airway block starts at 1  (1 Trachea, 2 dx, 3 sx, 4-8 lobare, 9 crossing)
  - artery block starts at 10 (10 Tronco, 11 dx, 12 sx, 13-17 lobare, 18 crossing)
  - vein   block starts at 20 (20 UNUSED, 21 dx, 22 sx, 23-27 lobare, 28 crossing)
plus the REFINED CROSSING block, one pair per vessel type, placed above
everything else in use rather than widening the 9-value blocks (which would
renumber artery/vein and collide with build_lobe_segments' 30-54):
  - 73 bronco interlobar,  74 bronco translobar
  - 75 arteria interlobar, 76 arteria translobar
  - 77 vena interlobar,    78 vena translobar
The plain 9/18/28 is NOT retired - it keeps exactly its old meaning,
"crossing, kind not determined", which is what a run without the fissure
surface still produces. Old files stay readable; new ones say more.

0 is shared across every type ("none"). Veins have no real single trunk
anatomically (they originate peripherally in the interlobular septa and
converge toward the left atrium via 4 main pulmonary veins, not one
tapering trunk), so their block has no trunk tier at all: the tree divides
straight into a right and a left venous trunk (21/22) at the root, value 20
is never produced, and the shared ColorTableVessels table likewise starts
the vein block at 21.

Outputs a combined single-file .vtk/.vtp (same cells/order as the input)
with one new cell array, AnatomicalSegment, using the unified scale above.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.anatomical_segments \\
        Artery_centerlines/09_lobe_reachability.json \\
        Artery_centerlines/09_lobe_reachability.vtk \\
        Artery_centerlines/10_anatomical_segments.vtk \\
        --vessel-type artery \\
        --legend-output Artery_centerlines/10_anatomical_segments.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import vtk

from vescan import crossings as crossingsmod
from vescan.crossings import (CROSSING_ID_ARRAY_NAME, CROSSING_TYPE_ARRAY_NAME,
                                      CROSSING_TYPE_NAME)
from vescan.io import load_surface, save_surface, detect_coordinate_space
from vescan.lobes import LOBE_ORDER, RIGHT_LOBES, LEFT_LOBES

logger = logging.getLogger(__name__)

GROUP_ID_ARRAY_NAME = "GroupId"
ANATOMICAL_SEGMENT_ARRAY_NAME = "AnatomicalSegment"
POINT_ANATOMICAL_SEGMENT_ARRAY_NAME = "AnatomicalSegmentPoint"
# vescan.stages.lobe_reachability's own per-point lobe containment,
# the input this module's point-level pass is built on.
LOBE_LABEL_POINT_ARRAY_NAME = "LobeLabelPoint"

# lobe_reachability's own LobeLabel scale is 0..len(LOBE_ORDER)+2; the top
# value is its "crossing" marker (see that module's compute_lobe_labels()) -
# reused here as-is, not recomputed.
CROSSING_LOBE_LABEL = len(LOBE_ORDER) + 2

# Within-block offsets, fixed regardless of --vessel-type.
OFFSET_TRUNK = 0
OFFSET_RIGHT = 1
OFFSET_LEFT = 2
OFFSET_LOBE_BASE = 3  # +3..+7 for LOBE_ORDER[0..4]
OFFSET_CROSSING = 8
BLOCK_SIZE = 9

BLOCK_START = {"airway": 1, "artery": 10, "vein": 20}

# REFINED CROSSING VALUES - the interlobar/translobar split, INSIDE
# AnatomicalSegment/AnatomicalSegmentPoint themselves rather than only in the
# separate CrossingType array, so one Active Scalar in Slicer shows both what
# a vessel is and which kind of crossing it makes.
#
# Placed in their own block at 73 instead of widening the 9-value blocks:
# widening would renumber artery (10-18) and vein (20-28), invalidating every
# existing output, legend and color table, and would collide with
# build_lobe_segments' LobeSegment_* values at 30-54. 73 is the first free
# value above everything in use (1-28 segments, 30-54 lobe segments, 60-72
# fissures).
#
#   73 bronco interlobar   74 bronco translobar
#   75 arteria interlobar  76 arteria translobar
#   77 vena interlobar     78 vena translobar
#
# The ORIGINAL 9/18/28 is not retired: it keeps exactly the meaning it always
# had - "crossing, kind not determined" - which is what a run without the
# fissure surface, or a crossing whose centerline never pierces one, still
# produces. So an old file reads correctly and a new one is strictly more
# informative.
REFINED_CROSSING_BLOCK_START = 73
REFINED_CROSSING_ORDER = ["airway", "artery", "vein"]


def refined_crossing_value(vesselType, crossingTypeValue):
    """The 73-78 value for one vessel type and CROSSING_TYPE_* kind, or None
    for a kind that has no refined value (none/unclassified - those keep the
    plain BLOCK_START + OFFSET_CROSSING label)."""
    from vescan.crossings import CROSSING_TYPE_INTERLOBAR, CROSSING_TYPE_TRANSLOBAR
    kind = {CROSSING_TYPE_INTERLOBAR: 0, CROSSING_TYPE_TRANSLOBAR: 1}.get(crossingTypeValue)
    if kind is None:
        return None
    return REFINED_CROSSING_BLOCK_START + 2 * REFINED_CROSSING_ORDER.index(vesselType) + kind


def crossing_values_for(vesselType):
    """Every label value that means "this is a crossing vessel" for one vessel
    type: the plain one AND both refined ones.

    Anything reading crossings back off a labeled file must use this, not the
    plain value alone - stage 10 emits the refined values whenever it could
    classify, so looking only for 9/18/28 would find nothing on a current
    file and everything on an older one."""
    from vescan.crossings import CROSSING_TYPE_INTERLOBAR, CROSSING_TYPE_TRANSLOBAR
    return {BLOCK_START[vesselType] + OFFSET_CROSSING,
            refined_crossing_value(vesselType, CROSSING_TYPE_INTERLOBAR),
            refined_crossing_value(vesselType, CROSSING_TYPE_TRANSLOBAR)}

NAMES = {
    "airway": {
        "trunk": "Trachea", "right": "Bronco principale destro", "left": "Bronco principale sinistro",
        "crossing": "Bronco con attraversamento multi-lobo",
        "RUL": "Bronco lobare superiore destro", "RML": "Bronco lobare medio destro",
        "RLL": "Bronco lobare inferiore destro", "LLL": "Bronco lobare inferiore sinistro",
        "LUL": "Bronco lobare superiore sinistro",
    },
    "artery": {
        "trunk": "Tronco polmonare", "right": "Arteria polmonare destra", "left": "Arteria polmonare sinistra",
        "crossing": "Arteria con attraversamento multi-lobo",
        "RUL": "Arteria lobare superiore destra", "RML": "Arteria lobare media destra",
        "RLL": "Arteria lobare inferiore destra", "LLL": "Arteria lobare inferiore sinistra",
        "LUL": "Arteria lobare superiore sinistra",
    },
    # No "trunk" entry: veins have no single trunk to label (see
    # classify_segment()'s VEIN EXCEPTION), so value blockStart+OFFSET_TRUNK
    # (20) is never produced and deliberately absent from the shared
    # ColorTableVessels table too, whose vein block runs 21-28.
    "vein": {
        "right": "Tronco venoso polmonare destro", "left": "Tronco venoso polmonare sinistro",
        "crossing": "Vena con attraversamento multi-lobo",
        "RUL": "Vena lobare superiore destra", "RML": "Vena lobare media destra",
        "RLL": "Vena lobare inferiore destra", "LLL": "Vena lobare inferiore sinistra",
        "LUL": "Vena lobare superiore sinistra",
    },
}


def classify_segment(group, vesselType):
    """0/crossing/trunk/right/left/lobe label for one group.

    PRIORITY ORDER, and why it changed from an earlier reachable_lobes-only
    version: reachable_lobes is a SUBTREE union, so a group whose own
    points sit 100% inside one lobe could still get swallowed into a
    coarser trunk/destra/sinistra label just because some distant
    descendant's subtree also happens to reach another lobe - confirmed on
    real vein data (a comb-shaped tree: a long "spine" of groups whose own
    points are 100% inside RUL, but whose subtree reaches all 5 lobes
    because short single-lobe tributaries keep peeling off along the way)
    that this produced visibly wrong results - "Vena lobare superiore
    destra" segments showing up as a generic crossing/trunk instead of
    their own specific lobe. Fixed by making the group's OWN local
    classification (lobe_reachability's `lobe_label` - already local-first
    with downstream-of-a-crossing taint propagation, see that module) the
    PRIMARY signal:
      1. lobe_label is the crossing value -> CROSSING (this group's own
         points touch >1 lobe, or it's downstream of one that does).
      2. lobe_label is a specific lobe (1..len(LOBE_ORDER)) -> that lobe's
         own label, directly - "once you're spatially in a lobe, you're
         that lobar vessel", regardless of what any subtree does further
         down.
      3. Otherwise (lobe_label == trunk, i.e. this group's own points are
         outside every lobe - no local signal at all) -> fall back to
         reachable_lobes (the subtree union) to decide trunk vs a side.

    CROSSING is reserved for a genuinely ambiguous vessel (own points
    literally straddle >1 lobe, or downstream of one that does) - a likely
    anomaly worth flagging. It is NOT the same situation as a real
    branching hub whose own points just sit in the mediastinum/near the
    atrium with children that individually resolve cleanly on either side
    (confirmed on real vein data - gid 952: 5 children, cleanly resolving
    to LLL/RLL/RLL/LUL/none - nothing ambiguous about that vessel itself).

    VEIN EXCEPTION: veins have no real single trunk (they converge via
    separate main pulmonary veins, not one tapering trunk - see module
    docstring), so the tree divides straight into a right and a left venous
    trunk at the root, with no trunk/confluence tier above them: a "no local
    signal, spans both sides" group is never dumped into a shared bucket,
    and blockStart+OFFSET_TRUNK is never produced for a vein.

    The side such a group ends up on is decided per CELL by
    apply_vein_side_correction(), not here: these groups straddle the
    midline by definition, and confirmed on real data that a single one of
    them can have children resolving cleanly to both sides at once (gid 952,
    5 children -> LLL/RLL/RLL/LUL/none), so no single group-level answer is
    honest. The value returned here is therefore provisional, and exists
    only so every cell of the group enters that pass. An earlier version
    instead took the majority of reachable_lobes per side, which is biased
    by construction - there are 3 right lobes against 2 left, so any group
    reaching the whole tree always voted right.

    Artery/airway keep a real trunk value instead, since Tronco polmonare/
    Trachea are genuine anatomy there."""
    blockStart = BLOCK_START[vesselType]
    ownLobeLabel = group["lobe_label"]

    if ownLobeLabel == CROSSING_LOBE_LABEL:
        return blockStart + OFFSET_CROSSING

    if 1 <= ownLobeLabel <= len(LOBE_ORDER):
        lobeName = LOBE_ORDER[ownLobeLabel - 1]
        return blockStart + OFFSET_LOBE_BASE + LOBE_ORDER.index(lobeName)

    # own points outside every lobe (lobe_label == trunk) - no local lobe
    # signal, fall back to what the subtree as a whole reaches.
    lobes = set(group["reachable_lobes"])
    if not lobes:
        return 0
    if len(lobes) == 1:
        lobeName = next(iter(lobes))
        return blockStart + OFFSET_LOBE_BASE + LOBE_ORDER.index(lobeName)

    if lobes <= RIGHT_LOBES:
        return blockStart + OFFSET_RIGHT
    if lobes <= LEFT_LOBES:
        return blockStart + OFFSET_LEFT
    if vesselType == "vein":
        return blockStart + OFFSET_RIGHT  # provisional - apply_vein_side_correction() decides per cell
    return blockStart + OFFSET_TRUNK


def label_name(label, vesselType):
    if label == 0:
        return "Non classificato"
    blockStart = BLOCK_START[vesselType]
    names = NAMES[vesselType]
    if label >= REFINED_CROSSING_BLOCK_START:
        from vescan.crossings import CROSSING_TYPE_INTERLOBAR
        kind = "interlobare" if label == refined_crossing_value(vesselType, CROSSING_TYPE_INTERLOBAR) \
            else "translobare"
        return f"{names['crossing']} - {kind}"
    offset = label - blockStart
    if offset == OFFSET_TRUNK:
        # Veins have no trunk tier at all (see classify_segment()), so this
        # value is never produced for them and NAMES carries no entry.
        return names.get("trunk", f"<{vesselType} has no trunk segment>")
    if offset == OFFSET_RIGHT:
        return names["right"]
    if offset == OFFSET_LEFT:
        return names["left"]
    if offset == OFFSET_CROSSING:
        return names["crossing"]
    return names[LOBE_ORDER[offset - OFFSET_LOBE_BASE]]


def all_labels_for(vesselType):
    """Every label value this vessel type can actually produce - veins
    never get the trunk value (see classify_segment()'s VEIN EXCEPTION),
    so it's left out here too rather than listed as a permanently-empty
    row."""
    from vescan.crossings import CROSSING_TYPE_INTERLOBAR, CROSSING_TYPE_TRANSLOBAR
    blockStart = BLOCK_START[vesselType]
    trunkLabels = [] if vesselType == "vein" else [blockStart + OFFSET_TRUNK]
    return [0] + trunkLabels + [blockStart + OFFSET_RIGHT, blockStart + OFFSET_LEFT] + \
        [blockStart + OFFSET_LOBE_BASE + i for i in range(len(LOBE_ORDER))] + \
        [blockStart + OFFSET_CROSSING,
         refined_crossing_value(vesselType, CROSSING_TYPE_INTERLOBAR),
         refined_crossing_value(vesselType, CROSSING_TYPE_TRANSLOBAR)]


def build_point_segment_array(polyData, legend, vesselType, midlineX=None, coordinateSpace="LPS"):
    """Point-level analogue of build_anatomical_segment_array(): the same
    unified label scale, but decided for each centerline point on its own
    rather than painted uniformly over a whole GroupId.

    This is what puts each boundary exactly where it happens instead of at
    the nearest group edge. A trunk stops being a trunk at the bifurcation
    that ends it, a right/left main vessel becomes a lobar one at the point
    the centerline actually enters that lobe's surface (not once some
    threshold fraction of its group is inside), and a crossing begins at the
    point the vessel changes lobe. Measured on real data that 2.7% of artery
    points, 3.9% of vein points and 34.9% of airway points disagree with
    their own group's label, and that 68/64/155 of those boundaries fall
    strictly inside a cell, where a per-group array cannot express them.

    Built on vescan.stages.lobe_reachability's LobeLabelPoint (per
    point: 0 none/fragment, 1..N inside exactly that lobe, N+1 outside every
    lobe, N+2 inside more than one at once) plus, for the points outside
    every lobe, the group's own subtree - see _extra_lobar_label().

    Crossings propagate from the exact point onward: within the group that
    crosses, every later point in centerline order; and over every group
    downstream of it, whole. That downstream set is read off the group's own
    lobe_label (lobe_reachability already propagated the taint over the
    graph there), so no second graph walk is needed here - a group carrying
    the crossing value whose own points touch at most one lobe is one that
    inherited it from an ancestor, and is crossing all the way through."""
    lobeLabelPointArray = polyData.GetPointData().GetArray(LOBE_LABEL_POINT_ARRAY_NAME)
    if lobeLabelPointArray is None:
        raise ValueError(f"Input polydata has no '{LOBE_LABEL_POINT_ARRAY_NAME}' point array - pass "
                          f"vescan.stages.lobe_reachability's own output (09_lobe_reachability.vtk)")

    blockStart = BLOCK_START[vesselType]
    n = len(LOBE_ORDER)
    crossingValue = blockStart + OFFSET_CROSSING
    groupIdArray = polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    groups = legend["groups"]

    labelArray = vtk.vtkIntArray()
    labelArray.SetName(POINT_ANATOMICAL_SEGMENT_ARRAY_NAME)
    labelArray.SetNumberOfValues(polyData.GetNumberOfPoints())
    for pointId in range(polyData.GetNumberOfPoints()):
        labelArray.SetValue(pointId, 0)

    for cellId in range(polyData.GetNumberOfCells()):
        groupId = int(groupIdArray.GetValue(cellId))
        group = groups.get(str(groupId))
        pointIds = polyData.GetCell(cellId).GetPointIds()
        ids = [pointIds.GetId(i) for i in range(pointIds.GetNumberOfIds())]
        pointLobeLabels = [int(lobeLabelPointArray.GetValue(pointId)) for pointId in ids]

        inheritedCrossing = (group is not None and group["lobe_label"] == n + 2
                             and len(group["own_lobes"]) <= 1)
        if inheritedCrossing:
            crossingFrom = 0
        else:
            crossingFrom = _crossing_start_index(pointLobeLabels, n)
            if crossingFrom is None:
                crossingFrom = len(ids)

        for index, (pointId, pointLobeLabel) in enumerate(zip(ids, pointLobeLabels)):
            if index >= crossingFrom:
                labelArray.SetValue(pointId, crossingValue)
            elif pointLobeLabel == 0:
                labelArray.SetValue(pointId, 0)
            elif 1 <= pointLobeLabel <= n:
                labelArray.SetValue(pointId, blockStart + OFFSET_LOBE_BASE + pointLobeLabel - 1)
            else:
                value = _extra_lobar_label(group, blockStart, vesselType)
                if value is None:
                    value = blockStart + _geometric_side_offset(polyData.GetPoint(pointId)[0],
                                                                 midlineX, coordinateSpace)
                labelArray.SetValue(pointId, value)

    return labelArray


def _geometric_side_offset(x, midlineX, coordinateSpace):
    """OFFSET_RIGHT/OFFSET_LEFT for a point that no subtree can place (a vein
    still straddling both sides). Falls back to OFFSET_RIGHT when no midline
    could be established - see find_midline_x()."""
    if midlineX is None:
        return OFFSET_RIGHT
    isRight = (x < midlineX) if coordinateSpace == "LPS" else (x > midlineX)
    return OFFSET_RIGHT if isRight else OFFSET_LEFT


def build_anatomical_segment_array(polyData, groupLabels):
    groupIdArray = polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    numCells = polyData.GetNumberOfCells()

    arr = vtk.vtkIntArray()
    arr.SetName(ANATOMICAL_SEGMENT_ARRAY_NAME)
    arr.SetNumberOfValues(numCells)
    for cellId in range(numCells):
        groupId = int(groupIdArray.GetValue(cellId))
        arr.SetValue(cellId, groupLabels.get(groupId, 0))
    return arr


def _extra_lobar_label(group, blockStart, vesselType):
    """Which tier a point that sits OUTSIDE every lobe belongs to, read off
    its group's subtree: trunk if that subtree still feeds both lungs, the
    right or left main vessel once it feeds only one.

    Such a point is never given a lobar label, even when its subtree feeds
    exactly one lobe - that is precisely the "Right pulmonary artery until
    it reaches the lobe" boundary this point-level pass exists to draw, and
    where it parts company with the per-group classify_segment(), which
    hands the whole group the lobe its subtree happens to end in.

    Returns None for a vein still straddling both sides: that block has no
    trunk tier, so the point has to be placed geometrically instead.

    A subtree reaching NO lobe stays unclassified (0) rather than being
    placed by side. Tried that and reverted it: a childless stub off the
    trunk bifurcation can be as wide as the trunk itself (16.3mm long,
    13.9mm radius on one real patient) and look like a pruned main pulmonary
    artery by every number available here, while being a centerline artifact
    doubling back inside the trunk's own lumen - so siding it painted a
    stretch of plainly-trunk surface as "left pulmonary artery". Leaving it
    at 0 lets transfer_centerline_labels' drop_unlabeled hand that surface to
    the nearest cell that does have a label, which is the trunk."""
    lobes = set(group["reachable_lobes"]) if group else set()
    if not lobes:
        return 0
    if lobes <= RIGHT_LOBES:
        return blockStart + OFFSET_RIGHT
    if lobes <= LEFT_LOBES:
        return blockStart + OFFSET_LEFT
    if vesselType == "vein":
        return None
    return blockStart + OFFSET_TRUNK


def _crossing_start_index(pointLobeLabels, n):
    """Index of the point at which a crossing group actually crosses, given
    that group's own per-point lobe labels in centerline order (source ->
    target). That is the first point sitting in a DIFFERENT lobe from the
    first lobe the vessel was in, or any point inside two lobes at once.

    A group-level crossing (own_lobes holding more than one lobe) is not the
    same thing as LobeLabelPoint's own N+2: a vessel that leaves lobe A and
    later enters lobe B crosses a boundary without any single one of its
    points ever testing inside both at once, so N+2 alone would never fire
    for it. Points outside every lobe in between (N+1) are skipped rather
    than treated as the crossing, since the vessel has not reached the new
    lobe yet. Returns None when no crossing shows up in this order."""
    firstLobe = None
    for index, value in enumerate(pointLobeLabels):
        if value == n + 2:
            return index
        if not 1 <= value <= n:
            continue
        if firstLobe is None:
            firstLobe = value
        elif value != firstLobe:
            return index
    return None


def straddles_midline(group):
    """True for a vein group that is the venous confluence itself: no local
    lobe signal of its own (its points sit outside every lobe) AND a subtree
    feeding both lungs, so neither side can claim it topologically.

    Everything else IS decided topologically - confirmed on real data that
    every single child of such a group already resolves cleanly to one side
    or to nothing (PD-1-Lung-00043 gid 2: SX/DX/DX/NONE/SX; R01-049 gid
    1321: 11 children, 7 clean sides and 4 empty), so the straddling is
    confined to the confluence groups themselves - one or two per patient,
    8 to 27 cells. Only those need geometry to be placed on a side."""
    if group["lobe_label"] != len(LOBE_ORDER) + 1:
        return False
    lobes = set(group["reachable_lobes"])
    return bool(lobes) and not lobes <= RIGHT_LOBES and not lobes <= LEFT_LOBES


def _mean_cell_x(polyData, cellId):
    pointIds = polyData.GetCell(cellId).GetPointIds()
    cellX = [polyData.GetPoint(pointIds.GetId(i))[0] for i in range(pointIds.GetNumberOfIds())]
    return sum(cellX) / len(cellX)


def find_midline_x(polyData, arr, blockStart):
    """The patient's own left/right divide, as the midpoint between the mean
    X of the cells already committed to a right lobe and that of the cells
    already committed to a left lobe.

    Anchoring on the two sides' own centres rather than on the centreline's
    overall centroid (what this used to do) matters because that centroid
    drifts with however many vessels each side happens to have: confirmed on
    real data that a patient with 387 right-lobar cells against 952
    left-lobar ones put it 19.4mm away from the true divide - deep into the
    mediastinum, exactly where the ambiguous proximal cells this reference
    is meant to arbitrate actually sit - while a balanced patient (873 vs
    992) landed within 0.7mm of it. Averaging the two sides instead is
    immune to that imbalance.

    Returns None when either side has no committed lobar cell at all, since
    no divide can be established then."""
    rightX, leftX = [], []
    for cellId in range(polyData.GetNumberOfCells()):
        offset = arr.GetValue(cellId) - blockStart - OFFSET_LOBE_BASE
        if 0 <= offset < len(LOBE_ORDER):
            (rightX if LOBE_ORDER[offset] in RIGHT_LOBES else leftX).append(_mean_cell_x(polyData, cellId))
    if not rightX or not leftX:
        return None
    return ((sum(rightX) / len(rightX)) + (sum(leftX) / len(leftX))) / 2


def apply_vein_side_correction(polyData, arr, blockStart, coordinateSpace, straddlingGroupIds):
    """VEIN ONLY, applied AFTER the group-level classify_segment() labels
    are already on the array - places the cells of the confluence groups
    (straddles_midline(), passed in as straddlingGroupIds) on a side, and
    touches NOTHING else. A group whose subtree already feeds one lung only
    was decided by the graph and keeps that decision: an earlier version
    re-derived every right/left cell geometrically, overruling topology it
    had no business second-guessing (on R01-049 that was 28 of 55 cells).

    For the confluence groups themselves geometry is unavoidable: they
    straddle by definition, and a single one of them has children resolving
    to both sides at once, so no group-level label can be honest and this
    has to break the "one label per GroupId" convention the rest of the
    pipeline follows. Each such cell is placed by its own mean X against
    find_midline_x(). LPS: more negative X = more right. RAS: more positive
    X = more right. With no usable midline the cells are left as
    classify_segment() set them.

    The graph's own root is deliberately NOT used as that reference, natural
    though it looks: it sits in the left atrium, which is not centred, and
    where exactly it lands depends on the max-radius seed the endpoints
    stage picked inside that blob - measured 1.7mm from the true divide on
    one patient but 11.4mm off on another, while find_midline_x() stays
    within 0.7mm on both.

    Returns {group_id: label} for the groups it touched, resolved to
    whichever side most of that group's own cells ended up on - so the
    legend JSON can report what was actually written to the array instead
    of classify_segment()'s provisional guess."""
    rightValue = blockStart + OFFSET_RIGHT
    leftValue = blockStart + OFFSET_LEFT
    if not straddlingGroupIds:
        return {}

    referenceX = find_midline_x(polyData, arr, blockStart)
    if referenceX is None:
        logger.warning("No cell is committed to a right lobe, or none to a left one - cannot establish this "
                       "patient's left/right divide, leaving the venous trunk cells as first classified.")
        return {}

    groupIdArray = polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    sidesByGroup = {}
    for cellId in range(polyData.GetNumberOfCells()):
        if arr.GetValue(cellId) not in (rightValue, leftValue):
            continue
        if int(groupIdArray.GetValue(cellId)) not in straddlingGroupIds:
            continue
        meanCellX = _mean_cell_x(polyData, cellId)
        isRight = (meanCellX < referenceX) if coordinateSpace == "LPS" else (meanCellX > referenceX)
        value = rightValue if isRight else leftValue
        arr.SetValue(cellId, value)
        sidesByGroup.setdefault(int(groupIdArray.GetValue(cellId)), []).append(value)

    return {groupId: Counter(values).most_common(1)[0][0] for groupId, values in sidesByGroup.items()}


def _analyse_crossings(polyData, vesselType, topologyPath, fissuresPath,
                        min_piercings_for_interlobar=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR,
                        interlobar_max_median_angle_deg=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """The crossing vessels of this centerline, or None when the inputs for
    the question aren't there. Delegates to vescan.crossings, the same
    module statistics (stage 11) uses, so both stages agree on what one
    crossing vessel is and on which ones are interlobar."""
    if not topologyPath or not os.path.isfile(topologyPath):
        logger.info("No branch-tree topology given - skipping the crossing-vessel grouping "
                    "(%s/%s arrays not written).", CROSSING_ID_ARRAY_NAME, CROSSING_TYPE_ARRAY_NAME)
        return None
    with open(topologyPath, encoding="utf-8") as f:
        topology = json.load(f)

    fissureSurface = None
    if fissuresPath and os.path.isfile(fissuresPath):
        fissureSurface = load_surface(fissuresPath)
    elif fissuresPath:
        logger.warning("Fissure surface '%s' not found - crossings will be grouped but left "
                        "unclassified (no interlobar/translobar split).", fissuresPath)

    return crossingsmod.analyse(polyData, topology, crossing_values_for(vesselType),
                                 fissureSurface=fissureSurface,
                                 min_piercings_for_interlobar=min_piercings_for_interlobar,
                                 interlobar_max_median_angle_deg=interlobar_max_median_angle_deg)


def refine_crossing_labels(polyData, segmentArray, pointSegmentArray, vesselType, vessels):
    """Replaces the plain crossing label with its interlobar/translobar
    refinement, in BOTH AnatomicalSegment and AnatomicalSegmentPoint.

    Runs AFTER the classification, never before: the classification itself
    finds crossings by reading the plain label, so refining first would pull
    the ground out from under it.

    A crossing whose kind stays unknown (no fissure surface given, or no
    piercing found) keeps the plain 9/18/28 - that value's meaning is
    unchanged, so nothing is lost and nothing is invented. Returns
    {oldGroupLabel -> newGroupLabel} for the groups whose CELL label moved,
    so the caller's groupLabels/legend stay in step with the array."""
    groupIdArray = polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME)
    crossingValues = crossing_values_for(vesselType)

    refinedByGroup = {}
    for vessel in vessels or []:
        refined = refined_crossing_value(vesselType, vessel["crossing_type_value"])
        if refined is None:
            continue
        for groupId in vessel["groups"]:
            refinedByGroup[groupId] = refined

    changedGroups, changedCells, changedPoints = {}, 0, 0
    for cellId in range(polyData.GetNumberOfCells()):
        groupId = int(groupIdArray.GetValue(cellId))
        refined = refinedByGroup.get(groupId)
        if refined is None:
            continue

        if int(segmentArray.GetValue(cellId)) in crossingValues:
            segmentArray.SetValue(cellId, refined)
            changedGroups[groupId] = refined
            changedCells += 1

        # Points are refined ONLY where they already carry a crossing label:
        # a crossing group's cell also holds the run-up points from before the
        # crossing starts, and those are still ordinary lobar/hilar points.
        pointIds = polyData.GetCell(cellId).GetPointIds()
        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            if int(pointSegmentArray.GetValue(pointId)) in crossingValues:
                pointSegmentArray.SetValue(pointId, refined)
                changedPoints += 1

    if changedCells or changedPoints:
        logger.info("Refined the crossing label into interlobar/translobar: %d cell(s), %d point(s).",
                    changedCells, changedPoints)
    return changedGroups


def classify_and_save(legend, polyData, output_path, vessel_type, legend_output_path=None,
                       coordinate_space="LPS", topology_path=None, fissures_path=None,
                       min_piercings_for_interlobar=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR,
                       interlobar_max_median_angle_deg=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """Does the actual classification/labeling work, given an
    already-loaded legend dict (lobe_reachability's build_legend_data()
    return value, or the same shape loaded back from its --legend-output
    JSON) and an already-loaded centerline polydata (lobe_reachability's
    own output). This is the piece run() below wraps with file I/O - split
    out so a caller already holding both in memory (main.py, when stage 9
    just produced them) can skip writing the legend to JSON and reading the
    VTK back just to hand them straight back here."""
    if polyData.GetCellData().GetArray(GROUP_ID_ARRAY_NAME) is None:
        raise ValueError(f"Input polydata has no '{GROUP_ID_ARRAY_NAME}' cell array - pass "
                          f"vescan.stages.lobe_reachability's own output (09_lobe_reachability.vtk)")

    groupLabels = {
        int(groupId): classify_segment(group, vessel_type)
        for groupId, group in legend["groups"].items()
    }

    segmentArray = build_anatomical_segment_array(polyData, groupLabels)
    midlineX = None
    if vessel_type == "vein":
        straddlingGroupIds = {int(groupId) for groupId, group in legend["groups"].items()
                              if straddles_midline(group)}
        midlineX = find_midline_x(polyData, segmentArray, BLOCK_START["vein"])
        # Keeps groupLabels (and so the legend/statistics below) in step with
        # what the per-cell pass actually wrote into the array.
        groupLabels.update(apply_vein_side_correction(polyData, segmentArray, BLOCK_START["vein"],
                                                       coordinate_space, straddlingGroupIds))

    pointSegmentArray = build_point_segment_array(polyData, legend, vessel_type, midlineX=midlineX,
                                                   coordinateSpace=coordinate_space)

    outputPolyData = vtk.vtkPolyData()
    outputPolyData.DeepCopy(polyData)
    outputPolyData.GetCellData().AddArray(segmentArray)
    outputPolyData.GetPointData().AddArray(pointSegmentArray)

    # Crossing grouping runs on the OUTPUT polydata, not the input: it needs
    # the AnatomicalSegmentPoint array this function just built.
    crossingVessels = _analyse_crossings(outputPolyData, vessel_type, topology_path, fissures_path,
                                          min_piercings_for_interlobar=min_piercings_for_interlobar,
                                          interlobar_max_median_angle_deg=interlobar_max_median_angle_deg)
    if crossingVessels is not None:
        for array in crossingsmod.build_crossing_arrays(
                outputPolyData, crossingVessels, crossing_values_for(vessel_type)):
            if array.GetName().endswith("Point"):
                outputPolyData.GetPointData().AddArray(array)
            else:
                outputPolyData.GetCellData().AddArray(array)

        # Strictly after build_crossing_arrays(), which still reads the plain
        # crossing label to decide which points belong to a crossing.
        groupLabels.update(refine_crossing_labels(outputPolyData, segmentArray, pointSegmentArray,
                                                   vessel_type, crossingVessels))

    save_surface(outputPolyData, output_path, coordinate_space=coordinate_space)

    counts = Counter(groupLabels.values())
    logger.info("=== Anatomical segment statistics (%s) ===", vessel_type)
    for label in all_labels_for(vessel_type):
        logger.info("  %d group(s): [%d] %s", counts.get(label, 0), label, label_name(label, vessel_type))
    logger.info("Saved labeled centerline (%d points, new point array '%s') to %s",
                outputPolyData.GetNumberOfPoints(), POINT_ANATOMICAL_SEGMENT_ARRAY_NAME, output_path)
    logger.info("Saved labeled centerline (%d cells, new array '%s') to %s",
                outputPolyData.GetNumberOfCells(), ANATOMICAL_SEGMENT_ARRAY_NAME, output_path)
    if crossingVessels:
        counted = Counter(vessel["crossing_type"] for vessel in crossingVessels)
        logger.info("Crossing vessels grouped into '%s'/'%s': %d total (%d interlobar, %d translobar, "
                    "%d unclassified)", CROSSING_ID_ARRAY_NAME, CROSSING_TYPE_ARRAY_NAME,
                    len(crossingVessels), counted["interlobar"], counted["translobar"],
                    counted["unclassified"])
        for vessel in crossingVessels:
            logger.info("  CrossingId %d: %-12s %s -> %s  %.1fmm  %d gruppi  %d perforazioni%s",
                        vessel["crossing"], vessel["crossing_type"], vessel["from_lobe"] or "?",
                        vessel["to_lobe"] or "?", vessel["length_mm"], vessel["n_groups"],
                        vessel["n_piercings"],
                        f"  angolo mediano {vessel['angle_median_deg']:.1f} deg"
                        if vessel["angle_median_deg"] is not None else "")

    if legend_output_path:
        data = {
            "vessel_type": vessel_type,
            "block_start": BLOCK_START[vessel_type],
            "label_names": {str(label): label_name(label, vessel_type) for label in all_labels_for(vessel_type)},
            "crossing_type_names": {str(value): name for value, name in CROSSING_TYPE_NAME.items()},
            "groups": {
                str(groupId): {"label": label, "name": label_name(label, vessel_type)}
                for groupId, label in groupLabels.items()
            },
        }
        if crossingVessels is not None:
            # The full per-piercing detail stays in stage 11's own JSON; this
            # is the grouping needed to read the arrays in this file.
            data["crossings"] = [
                {key: vessel[key] for key in ("crossing", "crossing_type", "from_lobe", "to_lobe",
                                               "length_mm", "n_groups", "groups", "n_piercings",
                                               "angle_median_deg", "fissures")}
                for vessel in crossingVessels
            ]
        with open(legend_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved legend/per-group JSON to %s", legend_output_path)

    return outputPolyData


def run(lobe_reachability_json_path, lobe_reachability_vtk_path, output_path, vessel_type,
        legend_output_path=None, topology_path=None, fissures_path=None,
        min_piercings_for_interlobar=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR,
        interlobar_max_median_angle_deg=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """File-based entry point (CLI / standalone use): loads lobe_reachability's
    own legend JSON + VTK output from disk, then delegates to
    classify_and_save(). When both stages run in the same process (see
    main.py), prefer calling classify_and_save() directly with
    lobe_reachability.run()'s own return values instead - it skips this
    JSON/VTK round-trip entirely."""
    with open(lobe_reachability_json_path, encoding="utf-8") as f:
        legend = json.load(f)

    coordinateSpace = detect_coordinate_space(lobe_reachability_vtk_path) or "LPS"
    polyData = load_surface(lobe_reachability_vtk_path)

    return classify_and_save(legend, polyData, output_path, vessel_type,
                              legend_output_path=legend_output_path, coordinate_space=coordinateSpace,
                              topology_path=topology_path, fissures_path=fissures_path,
                              min_piercings_for_interlobar=min_piercings_for_interlobar,
                              interlobar_max_median_angle_deg=interlobar_max_median_angle_deg)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lobe_reachability_json", help="Path to lobe_reachability's --legend-output "
                                                         "(09_lobe_reachability.json) - per-group reachable_lobes "
                                                         "and lobe_label")
    parser.add_argument("lobe_reachability_vtk", help="Path to lobe_reachability's own output "
                                                        "(09_lobe_reachability.vtk) - same cells/GroupId as the "
                                                        "combined model, from the same run as the json above")
    parser.add_argument("output", help="Path to save the labeled centerline (.vtk/.vtp) with the new "
                                        "AnatomicalSegment cell array added")
    parser.add_argument("--vessel-type", choices=["artery", "vein", "airway"], required=True,
                         help="Selects the label block/naming: artery (block 10-18), airway (block 1-9), vein "
                              "(block 20-28) - each block is trunk, destro, sinistro, 5 lobes (RUL..LUL), crossing, "
                              "in that order")
    parser.add_argument("--legend-output", default=None,
                         help="Optional path to save a JSON legend: per-group label/name, plus the "
                              "crossing-vessel grouping when --topology is given")
    parser.add_argument("--topology", default=None,
                         help="Path to 05_branch_tree_topology.json - enables the crossing-vessel "
                              f"grouping ({CROSSING_ID_ARRAY_NAME}/{CROSSING_TYPE_ARRAY_NAME} arrays). "
                              "Without it the output is exactly as before")
    parser.add_argument("--fissures", default=None,
                         help="The patient's lobe_fissures.vtk (build_lobe_segments' output). With "
                              "--topology it adds the interlobar/translobar split; without it "
                              "crossings are still grouped but left 'unclassified'")
    parser.add_argument("--min-piercings-for-interlobar", type=int,
                         default=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR, metavar="N",
                         help="A crossing vessel with fewer fissure piercings than this is translobar by "
                              f"construction (default: {crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR}) - see "
                              "vescan.crossings' module docstring for the rule and the data behind it.")
    parser.add_argument("--interlobar-max-median-angle-deg", type=float,
                         default=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG, metavar="DEG",
                         help="Above --min-piercings-for-interlobar piercings, the crossing is interlobar "
                              "when the median incidence angle to the fissure is below this (default: "
                              f"{crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG}).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.lobe_reachability_json, args.lobe_reachability_vtk, args.output, args.vessel_type,
        legend_output_path=args.legend_output,
        topology_path=args.topology, fissures_path=args.fissures,
        min_piercings_for_interlobar=args.min_piercings_for_interlobar,
        interlobar_max_median_angle_deg=args.interlobar_max_median_angle_deg,
    )


if __name__ == "__main__":
    sys.exit(main())