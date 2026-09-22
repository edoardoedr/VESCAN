#!/usr/bin/env python3
"""
Transfers labels from a centerline model onto a surface, for visualization
(e.g. color the surface by branch/bifurcation/generation, lobe reachability,
or anatomical segment in 3D Slicer's Display > Scalars).

Despite the name, this is completely generic over WHICH cell arrays get
transferred - it's used identically for branch/bifurcation labels (stage 8),
lobe reachability arrays (stage 9) and anatomical segment labels (stage 10);
only the --arrays list and the input centerline file differ each time. It
used to be named label_surface_branches.py, which undersold what it
actually does.

For every surface point, finds the nearest point on the centerline's actual
geometry (its line segments, not just its sample points - continuous along
the whole polyline, no "staircase" bias toward wherever the centerline
happens to be more densely sampled). Whatever cell that nearest point falls
on donates its cell-data values (GroupId, IsBifurcation, Generation,
LobeMask, AnatomicalSegment, ...) to that surface point.

A requested array that lives in the centerline's POINT data instead
(LobeLabelPoint, AnatomicalSegmentPoint - the per-point labels that put each
boundary exactly where it happens rather than at the nearest group edge) is
resolved one step finer, from the nearest centerline POINT rather than the
whole cell: the same hit already reports which segment of the cell it landed
on and where along it. Cell and point arrays therefore never disagree, since
both follow the same hit. Names are looked up in cell data first, so
--arrays needs no say in which is which.

If the centerline carries a per-point Radius array, "nearest" is resolved
with vmtk's own vtkvmtkPolyBallLine - the same radius-weighted implicit
function (dist^2 - radius^2, radius linearly interpolated along each
segment) that vtkvmtkPolyDataCenterlineGroupsClipper uses for branch
clipping - instead of raw Euclidean distance. This matters most on a wide
trunk near a much thinner sibling branch (e.g. right after a bifurcation):
a surface point on the trunk wall can be geometrically closer in raw
distance to the thin sibling's centerline than to the trunk's own, even
though it's nowhere near the sibling's actual lumen - raw-distance
assignment (vtkStaticCellLocator, same technique as vescan.stages.
clip_vessel's classify_and_clip_surface()) would mislabel it as belonging
to the sibling. The poly-ball metric doesn't have that failure mode: a
point deep inside the trunk's (large-radius) tube scores more negative than
the same point relative to the sibling's (small-radius) tube, so the trunk
correctly wins regardless of which centerline happens to be nearer in raw
distance. Falls back to vtkStaticCellLocator (no radius weighting) if the
centerline has no Radius point array.

Input centerline: whichever pipeline stage's own output already carries the
cell array(s) you want transferred as cell data in one file - e.g.
vescan.stages.build_graph's --split-output (05_branch_tree.vtk) for
GroupId/IsBifurcation/Generation, or vescan.stages.
lobe_reachability's / anatomical_segments' own output for their arrays.

Works on any surface in the same coordinate space as the centerline - the
full preprocessed surface (01_preprocessed.vtk) to see the whole tree
colored by branch, or a clipped surface (vescan.stages.clip_vessel's
output) to see just the kept part.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.transfer_centerline_labels pipeline_output/01_preprocessed.vtk \\
        pipeline_output/05_branch_tree.vtk pipeline_output/08_branch_labeled_surface.vtk
"""

import argparse
import logging
import sys
from collections import Counter

import vtk

try:
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
except ImportError:
    # Standalone VMTK build: compiled modules live inside the `vmtk` package.
    from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry

from vescan.io import load_surface, save_surface, detect_coordinate_space, flip_lps_ras
from vescan.stages.network import _add_cell_id_array, CELL_ID_ARRAY_NAME
from vescan.stages.cut_graph import _extract_cells

logger = logging.getLogger(__name__)

# Same convention as vescan.stages.centerline/clip_vessel - the per-point
# tube radius, used to weight the nearest-point search (see module docstring).
RADIUS_ARRAY_NAME = "Radius"

# GroupId already gives each branch/bifurcation its own number, but a GroupId
# can span more than one candidate cell (vescan.stages.build_graph's
# module docstring: "still occasionally >1" geometrically-distinct cells
# sharing one GroupId) - CellId (added on the fly below, one integer per
# cell, 0..N-1) is the one array guaranteed to give every individual branch/
# bifurcation SEGMENT its own distinct number, which is what you want for a
# "different color per piece" look with Slicer's Random color table.
DEFAULT_ARRAY_NAMES = ("GroupId", "IsBifurcation", "Generation", CELL_ID_ARRAY_NAME)


def resolve_source_arrays(centerlinePolyData, array_names):
    """Each requested array paired with whether it lives in the centerline's
    point data rather than its cell data. Cell data is looked up first, so a
    name present in both resolves the way it always did.

    Point arrays are carried because the pipeline's finest labels are
    per-point, not per-group: vescan.stages.lobe_reachability's
    LobeLabelPoint and anatomical_segments' AnatomicalSegmentPoint put each
    boundary at the exact point it happens rather than at the nearest group
    edge, and a cell-only transfer could not bring any of that to a
    surface."""
    cellData = centerlinePolyData.GetCellData()
    pointData = centerlinePolyData.GetPointData()

    resolved = {}
    missing = []
    for name in array_names:
        array = cellData.GetArray(name)
        if array is not None:
            resolved[name] = (array, False)
            continue
        array = pointData.GetArray(name)
        if array is not None:
            resolved[name] = (array, True)
            continue
        missing.append(name)

    if missing:
        available = ([cellData.GetArrayName(i) for i in range(cellData.GetNumberOfArrays())] +
                     [pointData.GetArrayName(i) for i in range(pointData.GetNumberOfArrays())])
        raise ValueError(f"Centerline is missing array(s) {missing} - available (cell and point): "
                          f"{available}. Pass a centerline output that already carries them (e.g. "
                          f"vescan.stages.build_graph's --split-output, 05_branch_tree.vtk).")
    return resolved


def _nearest_point_id(centerlinePolyData, cellId, subId, pcoord):
    """The centerline point id nearest a hit reported as (cell, segment
    within that cell, position along that segment): whichever end of segment
    `subId` the hit landed closer to.

    Snapping to an endpoint rather than interpolating is deliberate - these
    are categorical labels (which lobe, which anatomical segment), so a value
    halfway between two of them would be meaningless."""
    pointIds = centerlinePolyData.GetCell(cellId).GetPointIds()
    index = subId + (1 if pcoord >= 0.5 else 0)
    index = min(max(index, 0), pointIds.GetNumberOfIds() - 1)
    return pointIds.GetId(index)


def labeled_cell_ids(centerlinePolyData, sourceArrays, unlabeled_value=0):
    """Cells carrying a real label in at least one of the requested arrays.
    A cell counts as labeled as soon as any one value is not
    `unlabeled_value`; for a point array that means any one of its points."""
    kept = []
    for cellId in range(centerlinePolyData.GetNumberOfCells()):
        pointIds = None
        for array, isPointArray in sourceArrays.values():
            if not isPointArray:
                if array.GetTuple1(cellId) != unlabeled_value:
                    kept.append(cellId)
                    break
                continue
            if pointIds is None:
                pointIds = centerlinePolyData.GetCell(cellId).GetPointIds()
            if any(array.GetTuple1(pointIds.GetId(i)) != unlabeled_value
                   for i in range(pointIds.GetNumberOfIds())):
                kept.append(cellId)
                break
    return kept


# A labeled region smaller than this fraction of the surface's own total area
# is treated as an island and refilled from its surroundings - see
# fill_label_islands()'s docstring for how this value was chosen.
MAX_ISLAND_AREA_FRACTION = 0.0025


def _surface_neighbours_and_areas(surfacePolyData):
    """(neighbours, pointArea, totalArea) for the surface's own vertex graph.

    neighbours[i] is every point sharing a cell with point i; pointArea[i] is
    that point's share of the surface, a third of each incident triangle."""
    numPoints = surfacePolyData.GetNumberOfPoints()
    neighbours = [set() for _ in range(numPoints)]
    pointArea = [0.0] * numPoints
    totalArea = 0.0

    pointIds = vtk.vtkIdList()
    for cellId in range(surfacePolyData.GetNumberOfCells()):
        surfacePolyData.GetCellPoints(cellId, pointIds)
        ids = [pointIds.GetId(i) for i in range(pointIds.GetNumberOfIds())]
        if len(ids) == 3:
            area = vtk.vtkTriangle.TriangleArea(surfacePolyData.GetPoint(ids[0]),
                                                 surfacePolyData.GetPoint(ids[1]),
                                                 surfacePolyData.GetPoint(ids[2]))
            totalArea += area
            for pointId in ids:
                pointArea[pointId] += area / 3.0
        for a in ids:
            for b in ids:
                if a != b:
                    neighbours[a].add(b)
    return neighbours, pointArea, totalArea


def fill_label_islands(surfacePolyData, array_names,
                        max_island_area_fraction=MAX_ISLAND_AREA_FRACTION,
                        protected_values=None, max_passes=5):
    """Refills tiny isolated patches of a label from the surface around them,
    in place, for each of array_names independently.

    The transfer above gives every surface point the value of the nearest
    centerline, with no notion of which vessel the point actually belongs to.
    Near a lobar boundary that goes wrong: a point on one artery's wall can be
    nearest (even by the radius-weighted metric) to a centerline running just
    across the fissure, and comes out the neighbouring lobe's colour. The
    result is small speckles of a foreign label - measured at 0.86% of a real
    artery surface and 1.39% of the matching venous one, all of it at lobar
    boundaries. A patch like that has no reference of its own, so the honest
    value to give it is the one its surroundings agree on.

    A region is refilled when its AREA is at most max_island_area_fraction of
    the surface's total, and it has at least one neighbouring point of a
    different label to take a value from. Area rather than point count because
    mesh density varies hugely between structures - 12k points for an airway
    tree against 91k for an artery one - so a point-count threshold would mean
    something different for each.

    0.25% was chosen by measuring, not guessed: on both patients' artery trees
    the set of regions it absorbs is IDENTICAL to the one 0.5% and 1% absorb
    (16 and 17 regions, 867 and 890mm2), because the threshold falls inside a
    natural gap - islands top out at 206 and 319mm2 while the smallest real
    region is 3273 and 804mm2. At 1% that gap is crossed and an 821mm2 genuine
    region gets eaten.

    protected_values maps an array name to values that are never absorbed.
    Crossing vessels need this: a vessel that crosses a lobar boundary shows
    up precisely as a small patch of its own label surrounded by another, so
    it looks exactly like the artifact this removes while being the real
    finding the pipeline exists to report. Measured on real data that those
    patches sit right in the absorbed size range - 109 points on one artery
    tree, 160/111/109/51 on a venous one.

    A region with no differently-labeled neighbour is left alone: it is a
    detached piece of the mesh (6 of them on one airway surface) with no
    surroundings to take a colour from.

    Repeats until nothing changes, up to max_passes, so two adjacent islands
    resolve instead of one blocking the other."""
    protected_values = protected_values or {}
    neighbours, pointArea, totalArea = _surface_neighbours_and_areas(surfacePolyData)
    if totalArea <= 0:
        return 0
    maxIslandArea = totalArea * max_island_area_fraction
    numPoints = surfacePolyData.GetNumberOfPoints()

    totalFilled = 0
    for name in array_names:
        array = surfacePolyData.GetPointData().GetArray(name)
        if array is None:
            continue
        if array.GetNumberOfComponents() != 1:
            # Every label this runs on is a single number; silently rewriting
            # only component 0 of a vector would corrupt it instead.
            logger.warning("Not filling islands in '%s': it has %d components, and this only makes sense "
                            "for a single-valued label.", name, array.GetNumberOfComponents())
            continue
        protected = protected_values.get(name, set())
        values = [array.GetTuple1(i) for i in range(numPoints)]

        for _pass in range(max_passes):
            visited = [False] * numPoints
            filledThisPass = 0
            for seed in range(numPoints):
                if visited[seed]:
                    continue
                value = values[seed]
                stack, region = [seed], []
                visited[seed] = True
                while stack:
                    current = stack.pop()
                    region.append(current)
                    for neighbour in neighbours[current]:
                        if not visited[neighbour] and values[neighbour] == value:
                            visited[neighbour] = True
                            stack.append(neighbour)
                if value in protected:
                    continue
                if sum(pointArea[i] for i in region) > maxIslandArea:
                    continue
                ring = Counter(values[n] for i in region for n in neighbours[i] if values[n] != value)
                if not ring:
                    continue  # detached mesh fragment - nothing around it to copy
                replacement = ring.most_common(1)[0][0]
                for i in region:
                    values[i] = replacement
                filledThisPass += len(region)
            totalFilled += filledThisPass
            if not filledThisPass:
                break

        for i in range(numPoints):
            array.SetTuple1(i, values[i])

    return totalFilled


def transfer_labels(surfacePolyData, centerlinePolyData, array_names=DEFAULT_ARRAY_NAMES,
                     drop_unlabeled=False, unlabeled_value=0, fill_islands=False,
                     max_island_area_fraction=MAX_ISLAND_AREA_FRACTION, protected_values=None):
    """Copies each named centerline array onto surfacePolyData as point data,
    valued from whichever centerline cell is closest to each surface point -
    radius-weighted (vtkvmtkPolyBallLine) if the centerline carries a Radius
    point array, otherwise raw distance (vtkStaticCellLocator). See module
    docstring for why that distinction matters.

    A name found in the centerline's POINT data (see resolve_source_arrays())
    is resolved one step finer: the same closest-cell hit also says which
    segment of that cell, and where along it, so the value comes from the
    nearest centerline point instead of being constant over the whole cell.
    The two kinds of array therefore agree by construction - both follow the
    same hit - rather than being resolved by two different searches.

    drop_unlabeled removes the centerline cells that carry no label at all
    (value `unlabeled_value` everywhere) BEFORE the search, so the surface
    they would have claimed goes to the nearest labeled cell instead. Off by
    default because 0 is a perfectly good GroupId; pass it only for arrays
    where 0 genuinely means "unclassified", as it does for the reachability
    and anatomical-segment ones.

    That matters more than the cell count suggests, because this is a
    radius-weighted search: an unclassified cell with an implausibly large
    Radius claims surface far out of proportion to its length. Confirmed on
    real data - an airway tree whose disconnected fragments carried 25-46mm
    inscribed-sphere radii had 204 centerline points take over 8070 surface
    points, 35.2% of the whole surface, which a colour table mapping 0 to
    alpha 0 then renders as gaping transparent holes. Dropping those cells
    took it to zero.

    fill_islands runs a cleanup pass afterwards, refilling tiny isolated
    patches of a label from the surface around them - see
    fill_label_islands(), including why protected_values has to name the
    crossing-vessel label so a real crossing is not mistaken for one of those
    patches. Off by default for the same reason drop_unlabeled is: it only
    makes sense for categorical anatomical labels, not for GroupId, where
    small distinct regions are exactly the point."""
    sourceArrays = resolve_source_arrays(centerlinePolyData, array_names)

    if drop_unlabeled:
        keptCellIds = labeled_cell_ids(centerlinePolyData, sourceArrays, unlabeled_value)
        nDropped = centerlinePolyData.GetNumberOfCells() - len(keptCellIds)
        if not keptCellIds:
            logger.warning("Every centerline cell is unlabeled - transferring from all of them anyway, since "
                           "dropping them would leave nothing to transfer from.")
        elif nDropped:
            logger.info("Ignoring %d unlabeled centerline cell(s) of %d: the surface nearest them takes its "
                        "label from the nearest labeled cell instead of coming out unclassified.",
                        nDropped, centerlinePolyData.GetNumberOfCells())
            centerlinePolyData = _extract_cells(centerlinePolyData, keptCellIds)
            sourceArrays = resolve_source_arrays(centerlinePolyData, array_names)

    wantPointValues = any(isPointArray for _array, isPointArray in sourceArrays.values())

    numPoints = surfacePolyData.GetNumberOfPoints()
    outArrays = {}
    for name, (srcArray, _isPointArray) in sourceArrays.items():
        newArray = srcArray.NewInstance()
        newArray.SetName(name)
        newArray.SetNumberOfComponents(srcArray.GetNumberOfComponents())
        newArray.SetNumberOfTuples(numPoints)
        outArrays[name] = newArray

    if centerlinePolyData.GetPointData().GetArray(RADIUS_ARRAY_NAME) is not None:
        logger.info("Centerline carries %s - resolving nearest cell with vmtk's radius-weighted poly-ball "
                    "metric, so a surface point on a wide trunk isn't stolen by a geometrically-closer but "
                    "much thinner sibling branch near a bifurcation.", RADIUS_ARRAY_NAME)
        polyBall = vtkvmtkComputationalGeometry.vtkvmtkPolyBallLine()
        polyBall.SetInput(centerlinePolyData)
        polyBall.SetPolyBallRadiusArrayName(RADIUS_ARRAY_NAME)

        def closest_hit(point):
            polyBall.EvaluateFunction(point)
            cellId = polyBall.GetLastPolyBallCellId()
            if not wantPointValues:
                return cellId, None
            return cellId, _nearest_point_id(centerlinePolyData, cellId,
                                              polyBall.GetLastPolyBallCellSubId(),
                                              polyBall.GetLastPolyBallCellPCoord())
    else:
        logger.warning("Centerline has no %s point array - falling back to raw nearest-point distance, which "
                       "can mislabel a surface point on a wide trunk as belonging to a geometrically-closer but "
                       "much thinner sibling branch near a bifurcation.", RADIUS_ARRAY_NAME)
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(centerlinePolyData)
        locator.BuildLocator()
        closestPoint = [0.0, 0.0, 0.0]
        cellId, subId, dist2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)

        def closest_hit(point):
            locator.FindClosestPoint(point, closestPoint, cellId, subId, dist2)
            hitCellId = cellId.get()
            if not wantPointValues:
                return hitCellId, None
            # No parametric coordinate here, so place the hit along its
            # segment by comparing it with that segment's own endpoints.
            pointIds = centerlinePolyData.GetCell(hitCellId).GetPointIds()
            start = centerlinePolyData.GetPoint(pointIds.GetId(subId.get()))
            end = centerlinePolyData.GetPoint(pointIds.GetId(min(subId.get() + 1,
                                                                 pointIds.GetNumberOfIds() - 1)))
            toStart = vtk.vtkMath.Distance2BetweenPoints(closestPoint, start)
            toEnd = vtk.vtkMath.Distance2BetweenPoints(closestPoint, end)
            return hitCellId, _nearest_point_id(centerlinePolyData, hitCellId, subId.get(),
                                                 0.0 if toStart <= toEnd else 1.0)

    point = [0.0, 0.0, 0.0]
    for i in range(numPoints):
        surfacePolyData.GetPoint(i, point)
        cid, pid = closest_hit(point)
        for name, (srcArray, isPointArray) in sourceArrays.items():
            outArrays[name].SetTuple(i, srcArray.GetTuple(pid if isPointArray else cid))

    labeledSurface = vtk.vtkPolyData()
    labeledSurface.DeepCopy(surfacePolyData)
    for array in outArrays.values():
        labeledSurface.GetPointData().AddArray(array)

    if fill_islands:
        filled = fill_label_islands(labeledSurface, list(sourceArrays),
                                     max_island_area_fraction=max_island_area_fraction,
                                     protected_values=protected_values)
        if filled:
            logger.info("Refilled %d surface point(s) sitting in label islands smaller than %.2f%% of the "
                        "surface, from the labels around them - these are points whose nearest centerline "
                        "belongs to a different vessel across a lobar boundary.",
                        filled, 100.0 * max_island_area_fraction)
    return labeledSurface


def run(surface_path, centerline_path, output_path, array_names=DEFAULT_ARRAY_NAMES,
        drop_unlabeled=False, fill_islands=False,
        max_island_area_fraction=MAX_ISLAND_AREA_FRACTION, protected_values=None):
    logger.info("Loading surface %s...", surface_path)
    surfaceSpace = detect_coordinate_space(surface_path) or "LPS"
    surfacePolyData = load_surface(surface_path)
    logger.info("  -> %d points, %d cells (%s).",
                surfacePolyData.GetNumberOfPoints(), surfacePolyData.GetNumberOfCells(), surfaceSpace)

    logger.info("Loading centerline %s...", centerline_path)
    centerlineSpace = detect_coordinate_space(centerline_path) or "LPS"
    centerlinePolyData = load_surface(centerline_path)
    if centerlineSpace != surfaceSpace:
        logger.info("Centerline is tagged %s, surface is %s - converting to match.",
                    centerlineSpace, surfaceSpace)
        centerlinePolyData = flip_lps_ras(centerlinePolyData)
    logger.info("  -> %d points, %d cells.",
                centerlinePolyData.GetNumberOfPoints(), centerlinePolyData.GetNumberOfCells())
    if CELL_ID_ARRAY_NAME in array_names and centerlinePolyData.GetCellData().GetArray(CELL_ID_ARRAY_NAME) is None:
        _add_cell_id_array(centerlinePolyData)

    logger.info("Transferring %s...", list(array_names))
    labeledSurface = transfer_labels(surfacePolyData, centerlinePolyData, array_names=array_names,
                                      drop_unlabeled=drop_unlabeled, fill_islands=fill_islands,
                                      max_island_area_fraction=max_island_area_fraction,
                                      protected_values=protected_values)

    save_surface(labeledSurface, output_path, coordinate_space=surfaceSpace)
    logger.info("Saved labeled surface to %s. In Slicer: Display > Scalars > Active Scalar = one of %s. "
                "Use a 'Random'/discrete color table for %s (a distinct color per individual branch/bifurcation "
                "segment) or GroupId/Generation; IsBifurcation is a 0/1 flag.",
                output_path, list(array_names), CELL_ID_ARRAY_NAME)
    return labeledSurface


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_surface", help="Path to the surface to label (.vtk/.vtp) - e.g. 01_preprocessed.vtk "
                                               "(full tree) or vescan.stages.clip_vessel's output "
                                               "(clipped tree)")
    parser.add_argument("centerline", help="Path to a centerline output that carries the cell array(s) to "
                                            "transfer - e.g. build_graph's --split-output "
                                            "(05_branch_tree.vtk) for GroupId/IsBifurcation/Generation")
    parser.add_argument("output_surface", help="Path to save the labeled surface (.vtk/.vtp)")
    parser.add_argument("--arrays", default=",".join(DEFAULT_ARRAY_NAMES),
                         help=f"Comma-separated list of centerline arrays to transfer, cell or point - each "
                              f"name is looked up in cell data first, then point data, so per-point labels "
                              f"like LobeLabelPoint/AnatomicalSegmentPoint can be listed alongside the "
                              f"per-group ones (default: {','.join(DEFAULT_ARRAY_NAMES)})")
    parser.add_argument("--drop-unlabeled", action="store_true",
                         help="Ignore centerline cells whose value is 0 in every transferred array, so the "
                              "surface near them takes the nearest labeled cell's value instead of coming out "
                              "unclassified (and transparent, with a color table mapping 0 to alpha 0). Only "
                              "for arrays where 0 means 'unclassified' - NOT for the default list, where 0 is "
                              "a valid GroupId")
    parser.add_argument("--fill-islands", action="store_true",
                         help="After transferring, refill tiny isolated patches of a label from the labels "
                              "around them (see fill_label_islands()) - they are surface points whose nearest "
                              "centerline belongs to a different vessel across a lobar boundary. Same caveat "
                              "as --drop-unlabeled: only for categorical anatomical labels, NOT for GroupId. "
                              "From the CLI nothing is protected, so a crossing-vessel patch can be absorbed; "
                              "the pipeline protects those (see orchestrator._crossing_values_to_protect).")
    parser.add_argument("--max-island-area-fraction", type=float, default=MAX_ISLAND_AREA_FRACTION,
                         metavar="F", help="Largest region --fill-islands will absorb, as a fraction of the "
                                            "surface's total area (default: %(default)s).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.input_surface, args.centerline, args.output_surface,
        array_names=tuple(a.strip() for a in args.arrays.split(",") if a.strip()),
        drop_unlabeled=args.drop_unlabeled,
        fill_islands=args.fill_islands,
        max_island_area_fraction=args.max_island_area_fraction,
    )


if __name__ == "__main__":
    sys.exit(main())