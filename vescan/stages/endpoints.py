#!/usr/bin/env python3
"""
Standalone endpoint auto-detection for vascular/airway surfaces.

Replicates the "Auto-detect endpoints" logic of the Slicer ExtractCenterline
module (SlicerExtension-VMTK/ExtractCenterline/ExtractCenterline.py), without
requiring 3D Slicer. Input is a surface already preprocessed in Slicer
(cleaned/decimated), exported as .stl, .vtk or .vtp (.vtk/.vtp preferred -
see load_surface() docstring for why); output is a Slicer Markups fiducial
JSON (.mrk.json) file that can be loaded back into Slicer.

Works both as a standalone VMTK build (`from vmtk import vtkvmtk`) and inside
the 3D Slicer Python console/environment (bare `import vtkvmtkMiscPython`,
the way SlicerExtension-VMTK itself imports it) - see the import fallback
below.

Depends on the sibling module vescan/io.py (surface load/save,
coordinate-space detection/flip, largest-component filtering). Run stage 1
(vescan.stages.preprocess) first if you want Slicer's decimate/clean/
triangulate preprocessing applied before endpoint detection.

To match Slicer's result, the input surface must have the same preprocessing
(clean/decimate) as whatever Slicer used, and must end up processed in the
same coordinate system as Slicer's live scene (RAS) - see flip_lps_ras() and
detect_coordinate_space() in vescan/io.py, applied automatically here.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.endpoints input.vtp output_endpoints.mrk.json
    python -m vescan.stages.endpoints input.vtp output_endpoints.mrk.json --coordinate-system RAS
    python -m vescan.stages.endpoints input.vtp output_endpoints.mrk.json --start-point 1.2 3.4 5.6

Usage (from Python - main.py, a notebook, or the Slicer Python console):
    import sys
    sys.path.append("/path/to/vmtk_building")
    from vescan.stages import endpoints
    endpoints.run(
        "input.vtp",
        "endpoints_from_slicer.mrk.json",
    )
"""

import argparse
import json
import logging
import sys

import vtk

try:
    import vtkvmtkMiscPython as vtkvmtkMisc
except ImportError:
    # Standalone VMTK build: compiled modules live inside the `vmtk` package.
    from vmtk import vtkvmtkMiscPython as vtkvmtkMisc

from vescan.io import (
    load_surface,
    detect_coordinate_space,
    flip_lps_ras,
    keep_largest_connected_component,
)

logger = logging.getLogger(__name__)

RADIUS_ARRAY_NAME = "Radius"
TOPOLOGY_ARRAY_NAME = "Topology"
MARKS_ARRAY_NAME = "Marks"

# How far a detected endpoint may sit from the input surface before it gets
# dropped, as a multiple of the local vessel radius there. An endpoint is a
# medial-axis point, so its distance to the surface IS roughly the local
# radius - see _filter_endpoints_near_surface()'s docstring.
ENDPOINT_DISTANCE_RADIUS_FACTOR = 1.5

# Minimum fraction of the input surface's own bounding-box diagonal that a
# vtkvmtkPolyDataNetworkExtraction result must span to be trusted as a real
# trace of the whole vessel tree, rather than a degenerate/aborted one - see
# extract_network_robust()'s docstring.
MIN_HEALTHY_NETWORK_BBOX_COVERAGE = 0.9

# vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio: how far the
# traversal steps forward along the medial axis (as a multiple of the local
# radius) before re-testing where the vessel boundary is - matches VMTK's own
# default.
NETWORK_ADVANCEMENT_RATIO = 1.05


def clean_and_triangulate(surfacePolyData):
    """Mirrors ExtractCenterlineLogic.preprocess() minus decimation/subdivision
    (already done in Slicer before STL export)."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfacePolyData)
    cleaner.Update()

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(cleaner.GetOutput())
    triangulator.PassLinesOff()
    triangulator.PassVertsOff()
    triangulator.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(triangulator.GetOutput())
    normals.SetAutoOrientNormals(1)
    normals.SetFlipNormals(0)
    normals.SetConsistency(1)
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


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


def extract_network(surfacePolyData, hole_position=None, advancement_ratio=NETWORK_ADVANCEMENT_RATIO):
    """Mirrors ExtractCenterlineLogic.extractNetwork() with no pre-existing
    endpoints markup (start position = closest point to bounds corner).

    Segmented vessel surfaces (marching-cubes/flying-edges output - see
    vescan.stages.convert_segmentations) are fully closed/watertight
    2-manifold meshes with no natural opening anywhere, at the trunk or at
    any branch tip - so vtkvmtkPolyDataNetworkExtraction needs exactly ONE
    opening punched somewhere to seed its traversal from.

    CONFIRMED ON REAL DATA that which corner gets used here is NOT always
    topology-irrelevant, despite that being the original assumption here:
    for one real patient's artery surface, the bounds[0]/bounds[2]/bounds[4]
    corner below happened to land the seed hole in a spot where the
    traversal died almost immediately, producing a degenerate ~1-cell
    network out of an otherwise healthy, richly-branched tree (confirmed by
    re-running with a different corner - see extract_network_robust(),
    which is what callers should actually use; this function stays a thin,
    single-attempt primitive underneath it).

    hole_position (optional): open at this surface position instead of the
    (arbitrary) closest point to a bounds corner. run() uses this for a
    second, refined extraction pass once the actual trunk/confluence
    position is known, opening the hole there directly - matching
    vescan.stages.network.extract_network()'s own later use of that
    same designated start point, so the two networks' own sampled points
    end up coinciding almost exactly instead of merely being close (each
    network extraction's own point sampling is anchored to wherever its
    hole was opened, so two passes that open their hole at different,
    unrelated positions naturally land their nearby points a little apart,
    even on the same underlying surface)."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfacePolyData)
    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputConnection(cleaner.GetOutputPort())
    triangleFilter.Update()
    simplifiedPolyData = triangleFilter.GetOutput()

    if hole_position is None:
        bounds = [0.0] * 6
        simplifiedPolyData.GetBounds(bounds)
        hole_position = [bounds[0], bounds[2], bounds[4]]
    open_surface_at_point(simplifiedPolyData, hole_position)

    networkExtraction = vtkvmtkMisc.vtkvmtkPolyDataNetworkExtraction()
    networkExtraction.SetInputData(simplifiedPolyData)
    networkExtraction.SetAdvancementRatio(advancement_ratio)
    networkExtraction.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    networkExtraction.SetTopologyArrayName(TOPOLOGY_ARRAY_NAME)
    networkExtraction.SetMarksArrayName(MARKS_ARRAY_NAME)
    networkExtraction.Update()

    return networkExtraction.GetOutput()


def _bounds_diagonal(polyData):
    """Length (mm) of polyData's own axis-aligned bounding-box diagonal; 0.0
    for empty input."""
    if polyData is None or polyData.GetNumberOfPoints() == 0:
        return 0.0
    bounds = [0.0] * 6
    polyData.GetBounds(bounds)
    return vtk.vtkMath.Norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])


def extract_network_robust(surfacePolyData, min_healthy_coverage=MIN_HEALTHY_NETWORK_BBOX_COVERAGE,
                            advancement_ratio=NETWORK_ADVANCEMENT_RATIO):
    """Calls extract_network() at each of the surface's own 8 bounding-box
    corners in turn and keeps whichever result traces the most of the
    surface, instead of trusting a single arbitrary corner - see
    extract_network()'s docstring for the real case this fixes: an unlucky
    seed hole can make vtkvmtkPolyDataNetworkExtraction's traversal die
    almost immediately, silently turning a healthy, richly-branched tree
    into a handful of points.

    "Traces the most of the surface" is measured as BOUNDING-BOX COVERAGE:
    the network's own bounding-box diagonal over the input surface's. A
    network is a medial-axis trace of the whole surface, so a healthy one
    reaches the extremes on all three axes and scores ~1.0, while an
    aborted traversal stays inside whatever corner of the anatomy it died
    in and scores far lower.

    This deliberately replaces an earlier raw CELL-COUNT criterion, which a
    real patient's artery surface defeated (R01-091): its very first corner
    produced a 31-cell network that cleared the old 20-cell bar and ended
    the search, when the remaining 7 corners each traced ~1530 cells. The
    difference is invisible to a cell count but obvious in coverage (0.26
    vs 1.00), and the consequence was severe - the start point came from
    that stunted network's own maximum radius (2.8mm, on a peripheral
    vessel far right/posterior/inferior) instead of the real pulmonary
    trunk (12.9mm near the midline), so the centerline covered a small
    corner of the artery tree rather than the tree.

    Coverage also has the property the cell count lacked: it is scale-free,
    so one threshold fits structures of wildly different richness. On the
    same patient a healthy airway network is ~160 cells and a healthy
    artery network ~1530, yet both score >0.97 while the two genuinely
    degenerate extractions (one airway corner, one artery corner) score
    0.009 and 0.26.

    No anatomy is assumed here - coverage is purely a stand-in for "did
    this seed let the algorithm actually trace the whole tree", without
    needing to know in advance where the real trunk is (which would itself
    require a working network to answer, from get_end_points()'s own
    global-max-radius logic - this function exists precisely because that
    input can't be trusted yet at this point).

    Stops early, without trying the remaining corners, once a result
    reaches min_healthy_coverage - most surfaces are fine on the very first
    corner, and this only pays the extra cost of the full 8 when the
    surface is actually being difficult."""
    bounds = [0.0] * 6
    surfacePolyData.GetBounds(bounds)
    candidatePositions = [
        [x, y, z]
        for x in (bounds[0], bounds[1])
        for y in (bounds[2], bounds[3])
        for z in (bounds[4], bounds[5])
    ]
    surfaceDiagonal = _bounds_diagonal(surfacePolyData)

    bestNetwork = None
    bestScore = (-1.0, -1)
    bestCoverage = 0.0
    for i, position in enumerate(candidatePositions):
        network = extract_network(surfacePolyData, hole_position=position, advancement_ratio=advancement_ratio)
        coverage = (_bounds_diagonal(network) / surfaceDiagonal) if surfaceDiagonal > 0 else 0.0
        # Cell count only breaks ties between seeds that cover the surface
        # equally well; it never outvotes coverage.
        score = (coverage, network.GetNumberOfCells())
        logger.info("Network extraction seed candidate %d/%d at %s -> %d cells, %d points, "
                    "bounding-box coverage %.3f.",
                    i + 1, len(candidatePositions), [round(c, 1) for c in position],
                    network.GetNumberOfCells(), network.GetNumberOfPoints(), coverage)
        if score > bestScore:
            bestScore = score
            bestCoverage = coverage
            bestNetwork = network
        if coverage >= min_healthy_coverage:
            break

    if bestCoverage < min_healthy_coverage:
        logger.warning("No seed candidate reached %.2f bounding-box coverage (best: %.3f) - the surface "
                        "itself may genuinely be disconnected/atypical, or every corner happened to be "
                        "unlucky. Proceeding with the best one found; the start point it yields is worth "
                        "checking.", min_healthy_coverage, bestCoverage)

    return bestNetwork


def _local_radius_measure(networkPolyData, surfacePolyData, radiusArray):
    """pointId -> local vessel radius (mm), as a callable.

    MEASURES the distance from the point to surfacePolyData rather than
    trusting the network's own Radius array. The two are the same quantity: a
    network point lies on the medial axis, so its distance to the surface IS
    the radius of the largest sphere that fits inside the vessel there. The
    difference is only that one is computed from the geometry we already hold
    and the other is vtkvmtkPolyDataNetworkExtraction's own estimate.

    That estimate is not always reliable. On a real patient's artery tree
    (LUNGx-CT011) it reported 15.40mm on the right pulmonary artery where the
    true inscribed sphere is 9.00mm - inflated by 70%, and enough to beat the
    genuine widest point of the tree. The start point therefore landed 37.7mm
    right of the anatomical midline, on the right pulmonary artery instead of
    the trunk, and the whole tree was rooted there: the blanked bifurcation
    group covering the right pulmonary artery inherited the "Tronco
    polmonare" label (from it every lobe is reachable, if only by running
    back through the real trunk), the real trunk got folded into the group
    labelled left pulmonary artery, and NO group was left reaching the right
    lobes alone - so "Arteria polmonare destra" was absent from that patient
    entirely. Measuring instead moves the start point to 12.0mm from the
    midline, inside the trunk.

    The patient's anatomy was never unusual: the true radius profile peaks at
    the midline and tapers outward exactly like every other patient's. Only
    the estimate was wrong.

    Costs ~0.2s for a 24k-point network against a 106k-point surface, and
    vtkPointLocator over the surface's vertices picks the identical point for
    a fifth of that - the accurate filter is used anyway since neither is a
    meaningful cost here.

    Falls back to the Radius array when no surface is given, so calling
    get_end_points() with a bare network still works."""
    if surfacePolyData is None or surfacePolyData.GetNumberOfPoints() == 0:
        if radiusArray is None:
            raise ValueError("Network has no Radius array and no surface was given - cannot tell which "
                              "point of the tree is the widest.")
        logger.info("No surface given - falling back to the network's own Radius array to find the "
                    "trunk/origin.")
        return lambda pointId: radiusArray.GetValue(pointId)

    implicitDistance = vtk.vtkImplicitPolyDataDistance()
    implicitDistance.SetInput(surfacePolyData)
    points = networkPolyData.GetPoints()
    # Signed: negative inside the closed surface, which is where every medial
    # axis point sits - the magnitude is the radius either way.
    distances = [abs(implicitDistance.EvaluateFunction(points.GetPoint(i)))
                 for i in range(points.GetNumberOfPoints())]
    return lambda pointId: distances[pointId]


def get_end_points(networkPolyData, startPointPosition=None, surfacePolyData=None):
    """Returns a list of [x, y, z] positions; the first entry is the start
    point.

    If startPointPosition is given, the closest LEAF (degree-1 point) to it
    becomes the start point - unchanged from ExtractCenterlineLogic.getEndPoints().

    If not given, the trunk/origin is instead identified as the tree's
    GLOBAL widest point, checked across EVERY point (not just leaves), with
    the local radius MEASURED against surfacePolyData rather than read off
    the network's own Radius array - see _local_radius_measure() for the real
    patient whose estimate was 70% too large on one side and moved the start
    point onto the right pulmonary artery. Segmented vessel surfaces are
    fully closed/watertight (no natural boundary anywhere to anchor a
    "largest opening" heuristic to - see extract_network()'s docstring), so
    the widest point is all there is to go on. This also deliberately differs
    from Slicer's own logic, which finds the
    largest radius among LEAVES ONLY: that can land on a side branch with a
    locally elevated (possibly noisy) radius, and more importantly can't
    represent a tree whose real origin is a confluence rather than a simple
    single-direction trunk (e.g. veins converging near the septum) - such a
    confluence is an INTERIOR point of the network (multiple branches meet
    there), never a leaf, so a leaf-only search could never find it no
    matter the radius threshold.

    When the global maximum-radius point turns out to be a leaf, this is
    exactly Slicer's own heuristic (just correctly scoped) and behaves
    identically to picking that leaf as the start point. When it is an
    interior point instead, it is used directly as the start point anyway
    (being an actual point of the network's own geometry, no projection or
    synthesis needed) - the leaf nearest to it, if any, is NOT removed or
    demoted; it simply remains an ordinary target endpoint like all the
    others."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(networkPolyData)
    cleaner.Update()
    network = cleaner.GetOutput()
    network.BuildCells()
    network.BuildLinks(0)

    networkPoints = network.GetPoints()

    startPointId = -1
    minDistance2 = 0

    endpointIds = vtk.vtkIdList()
    for i in range(network.GetNumberOfCells()):
        cell = network.GetCell(i)
        numberOfCellPoints = cell.GetNumberOfPoints()
        if numberOfCellPoints < 2:
            continue

        for pointIndex in [0, numberOfCellPoints - 1]:
            pointId = cell.GetPointId(pointIndex)
            pointCells = vtk.vtkIdList()
            network.GetPointCells(pointId, pointCells)
            if pointCells.GetNumberOfIds() == 1:
                endpointIds.InsertUniqueId(pointId)
                if startPointPosition is not None:
                    position = networkPoints.GetPoint(pointId)
                    distance2 = vtk.vtkMath.Distance2BetweenPoints(position, startPointPosition)
                    if startPointId < 0 or distance2 < minDistance2:
                        minDistance2 = distance2
                        startPointId = pointId

    endpointPositions = []
    numberOfEndpointIds = endpointIds.GetNumberOfIds()
    if numberOfEndpointIds == 0:
        return endpointPositions

    if startPointPosition is not None:
        endpointPositions.append(list(networkPoints.GetPoint(startPointId)))
    else:
        radiusArray = network.GetPointData().GetArray(RADIUS_ARRAY_NAME)
        localRadius = _local_radius_measure(network, surfacePolyData, radiusArray)
        maxRadiusPointId = max(range(networkPoints.GetNumberOfPoints()), key=localRadius)
        startPointId = maxRadiusPointId
        isLeaf = endpointIds.IsId(maxRadiusPointId) >= 0
        # Both numbers, always: they should agree, and the run where they did
        # not is the reason this is measured rather than read off the array
        # (see _local_radius_measure()).
        logger.info("No start point given - using the widest point of the tree %s as the trunk/origin: "
                    "%.2fmm measured to the surface, %s by the network's own Radius array%s.",
                    [round(c, 2) for c in networkPoints.GetPoint(maxRadiusPointId)],
                    localRadius(maxRadiusPointId),
                    ("%.2fmm" % radiusArray.GetValue(maxRadiusPointId)) if radiusArray else "unavailable",
                    "" if isLeaf else " - an interior confluence point, not a leaf")
        endpointPositions.append(list(networkPoints.GetPoint(maxRadiusPointId)))

    for pointIdIndex in range(numberOfEndpointIds):
        pointId = endpointIds.GetId(pointIdIndex)
        if pointId == startPointId:
            continue
        endpointPositions.append(list(networkPoints.GetPoint(pointId)))

    return endpointPositions


def _endpoint_local_radii(networkPolyData, endpointPositions):
    """Local vessel radius (mm) at each endpoint, as the LARGEST Radius along
    the network cell(s) the endpoint belongs to - not the value at the
    endpoint's own point.

    That distinction is the whole reason this rule needs no absolute floor.
    vtkvmtkPolyDataNetworkExtraction's last profile on a terminal branch
    degenerates, so Radius collapses towards 0 exactly AT the tip: measured
    across both patients' three structures, the tip value is literally 0 on a
    handful of endpoints and far below the branch's real calibre on hundreds
    more. Scaling a tolerance by that number means "must be within nothing of
    the surface", and discards 200-276 perfectly good endpoints per tree.
    Taking the branch's own maximum instead describes the vessel the endpoint
    actually sits in, and drops none at any factor between 1.0 and 2.0.

    Returns zeros if the network carries no Radius array, which makes the
    filter reject every endpoint rather than silently pass them all - a
    missing array means the caller handed over something that is not a
    network extraction, and failing loudly beats filtering on nothing."""
    radiusArray = networkPolyData.GetPointData().GetArray(RADIUS_ARRAY_NAME) if networkPolyData else None
    if radiusArray is None:
        logger.warning("Network carries no '%s' array - endpoint distance filtering has no radius to "
                        "scale by.", RADIUS_ARRAY_NAME)
        return [0.0] * len(endpointPositions)

    networkPolyData.BuildLinks(0)
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(networkPolyData)
    pointLocator.BuildLocator()

    radii = []
    for position in endpointPositions:
        pointId = pointLocator.FindClosestPoint(position)
        cellIds = vtk.vtkIdList()
        networkPolyData.GetPointCells(pointId, cellIds)
        radius = float(radiusArray.GetValue(pointId))
        for i in range(cellIds.GetNumberOfIds()):
            cell = networkPolyData.GetCell(cellIds.GetId(i))
            for j in range(cell.GetNumberOfPoints()):
                radius = max(radius, float(radiusArray.GetValue(cell.GetPointId(j))))
        radii.append(radius)
    return radii


def _filter_endpoints_near_surface(surfacePolyData, endpointPositions, endpointRadii, radius_factor):
    """Drops any endpoint farther from surfacePolyData than radius_factor
    times the local vessel radius there, logging a warning for each one
    dropped. The first entry (the start point) can be dropped like any other -
    callers should re-check the result isn't empty afterwards.

    An endpoint that ends up far from the surface (e.g. from floating-point
    precision issues after a flip, or leftover state from a bug elsewhere)
    doesn't fail cleanly if it reaches vtkvmtkPolyDataCenterlines downstream -
    vtkvmtkSteepestDescentLineTracer instead logs a native "Seed id invalid"
    error and returns an empty centerline, which then segfaults
    vtkvmtkCenterlineBranchExtractor (see vescan.stages.centerline's
    module docstring). Simplest fix: never write such a point out in the
    first place.

    The allowance is RADIUS-PROPORTIONAL because a network endpoint is a
    MEDIAL-AXIS point, not a surface point: it sits on the vessel's own
    centre line, so its distance to the nearest surface point is by
    construction about the local inscribed-sphere radius there. A fixed
    threshold therefore doesn't ask "is this point off the surface?", it asks
    "is this vessel narrow?" - and it discards precisely the endpoints at the
    WIDEST parts of the tree, which are the ones that matter most.

    Measured on two real patients, every endpoint the earlier fixed 1.0mm
    threshold discarded was a legitimate medial-axis tip sitting at 0.7-1.06
    times its own local radius - 30 of them across 6 structures, including
    the tip of LUNGx-CT011's trachea (6.3mm out, local radius 8.0mm) and both
    patients' pulmonary-artery trunk tips (9.6mm/12.2mm and 10.2mm/13.2mm).
    All 30 are kept here, and none of the remaining ~3300 endpoints is
    dropped - at any factor between 1.0 and 2.0, so 1.5 is not a value the
    result balances on.

    There is deliberately NO absolute floor alongside this: the only reason
    one seemed necessary is that Radius collapses towards 0 at a terminal
    tip, which _endpoint_local_radii() fixes at the source by measuring the
    branch instead of the tip.

    This stays a real guard despite dropping nothing on current data: the
    failure it exists for is an endpoint in the wrong coordinate space, which
    lands hundreds of mm away while its local radius stays ~1mm - a ratio two
    orders of magnitude past the factor, not a near miss."""
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(surfacePolyData)
    pointLocator.BuildLocator()

    kept = []
    for position, radius in zip(endpointPositions, endpointRadii):
        pointId = pointLocator.FindClosestPoint(position)
        closestPosition = surfacePolyData.GetPoint(pointId)
        distance = vtk.vtkMath.Distance2BetweenPoints(position, closestPosition) ** 0.5
        allowed = radius_factor * radius
        if distance > allowed:
            logger.warning("Dropping endpoint at %s - %.2fmm from the nearest surface point %s "
                            "(max allowed here: %.2fmm = %.2f x local radius %.2fmm); "
                            "it does not lie on/near the surface.",
                            list(position), distance, list(closestPosition), allowed,
                            radius_factor, radius)
            continue
        kept.append(position)
    return kept


def save_as_slicer_markups(positions, output_path, coordinate_system="LPS"):
    """Writes endpoint positions as a Slicer Markups Fiducial .mrk.json file.
    The first position (the start point) is saved unselected, matching
    ExtractCenterline's convention for marking the start point."""
    controlPoints = []
    for i, position in enumerate(positions):
        controlPoints.append({
            "id": str(i + 1),
            "label": f"F-{i + 1}",
            "description": "",
            "associatedNodeID": "",
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": (i != 0),
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        })

    markupsFile = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": coordinate_system,
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(controlPoints),
                "controlPoints": controlPoints,
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 0.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(markupsFile, f, indent=4)


def run(stl_path, output_path, start_point=None, coordinate_system=None,
        preprocess=False, keep_largest_component=True, flip_input_lps_to_ras=None,
        endpoint_distance_radius_factor=ENDPOINT_DISTANCE_RADIUS_FACTOR,
        min_healthy_network_bbox_coverage=MIN_HEALTHY_NETWORK_BBOX_COVERAGE,
        network_advancement_ratio=NETWORK_ADVANCEMENT_RATIO):
    """Detects endpoints on stl_path and writes them as a Slicer Markups
    fiducial file to output_path. Returns the detected endpoint positions
    (first entry is the start point)."""
    detectedSpace = detect_coordinate_space(stl_path)
    if flip_input_lps_to_ras is None:
        if detectedSpace == "RAS":
            flip_input_lps_to_ras = False
            logger.info("Detected SPACE=RAS in file header - no flip needed.")
        else:
            # detectedSpace is "LPS" or None (.vtp/.stl carry no header; Slicer's
            # model exporter is confirmed LPS, so assume LPS as the safer default).
            flip_input_lps_to_ras = True
            logger.info("%s - assuming LPS and auto-flipping to RAS to match Slicer's live scene. "
                        "Pass flip_input_lps_to_ras=False to override.",
                        "Detected SPACE=LPS in file header" if detectedSpace
                        else "No coordinate-space header found (.vtp/.stl)")

    surfacePolyData = load_surface(stl_path)
    if flip_input_lps_to_ras:
        surfacePolyData = flip_lps_ras(surfacePolyData)
        if start_point is not None:
            start_point = [-start_point[0], -start_point[1], start_point[2]]
    if keep_largest_component:
        surfacePolyData = keep_largest_connected_component(surfacePolyData)
    if preprocess:
        surfacePolyData = clean_and_triangulate(surfacePolyData)

    networkPolyData = extract_network_robust(surfacePolyData, min_healthy_coverage=min_healthy_network_bbox_coverage,
                                              advancement_ratio=network_advancement_ratio)
    endpointPositions = get_end_points(networkPolyData, startPointPosition=start_point,
                                        surfacePolyData=surfacePolyData)

    if not endpointPositions:
        raise RuntimeError("No endpoints detected. Check that the input surface is a single open/tubular network.")

    if start_point is None:
        # The start point just found (an interior confluence point, in the
        # common case - see get_end_points()'s docstring) came from a
        # network extracted with an arbitrary bounds-corner seed hole -
        # vescan.stages.network's own (separate) extraction will
        # later open ITS seed hole exactly at this designated start point
        # instead, so its own sampled points near there won't quite
        # coincide with this pass's (each extraction's point sampling is
        # anchored to wherever its own hole was opened - see
        # extract_network()'s docstring). Re-extract once more, this time
        # opening the hole at the same position that other extraction will
        # use, and re-snap the start point onto THIS network - matching the
        # other one closely enough that both effectively agree on the same
        # physical point.
        refinedNetworkPolyData = extract_network(surfacePolyData, hole_position=endpointPositions[0],
                                                  advancement_ratio=network_advancement_ratio)
        refinedPointLocator = vtk.vtkPointLocator()
        refinedPointLocator.SetDataSet(refinedNetworkPolyData)
        refinedPointLocator.BuildLocator()
        refinedPointId = refinedPointLocator.FindClosestPoint(endpointPositions[0])
        endpointPositions[0] = list(refinedNetworkPolyData.GetPoints().GetPoint(refinedPointId))

    # The start point (first entry) stays exempt from this filter. The filter
    # itself now understands why such a point sits away from the surface -
    # every endpoint is a medial-axis point, so its distance to the surface
    # is about the local radius, which is what the radius-proportional
    # allowance is for (see _filter_endpoints_near_surface()'s docstring) -
    # and the start point would comfortably pass it (0.40-0.92 times its own
    # radius, measured across both patients' three structures). The exemption
    # is kept anyway because losing the start point is fatal rather than
    # merely lossy: run() has nothing to fall back on and raises below.
    endpointRadii = _endpoint_local_radii(networkPolyData, endpointPositions)
    startPoint, *otherPoints = endpointPositions
    otherPoints = _filter_endpoints_near_surface(surfacePolyData, otherPoints, endpointRadii[1:],
                                                  endpoint_distance_radius_factor)
    endpointPositions = [startPoint] + otherPoints
    if len(endpointPositions) < 2:
        raise RuntimeError(f"Fewer than 2 endpoints remain after filtering (need a start point plus at "
                            f"least one target) - every other detected endpoint was farther from the "
                            f"surface than {endpoint_distance_radius_factor} x its local radius. Check "
                            f"the input surface/coordinate space.")

    # endpointPositions are currently in RAS if we flipped, otherwise in whatever
    # space the (unflipped) input surface was in (falling back to the assumed LPS
    # default when undetermined, same assumption used for the flip decision above).
    currentSpace = "RAS" if flip_input_lps_to_ras else (detectedSpace or "LPS")

    # Slicer itself always stores markups files in LPS - even though the live scene
    # is RAS - for interchange-format compatibility (see any Slicer-exported
    # .mrk.json: "coordinateSystem": "LPS"). Match that convention by default so
    # raw position values compare directly against Slicer-exported files.
    if coordinate_system is None:
        coordinate_system = "LPS"
    if coordinate_system != currentSpace:
        endpointPositions = [[-p[0], -p[1], p[2]] for p in endpointPositions]

    save_as_slicer_markups(endpointPositions, output_path, coordinate_system=coordinate_system)
    return endpointPositions


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_stl", help="Path to the input surface exported from Slicer (.stl, .vtk or .vtp; "
                                           "prefer .vtk/.vtp to avoid STL float32/connectivity round-trip issues)")
    parser.add_argument("output_mrk_json", help="Path to the output Slicer Markups fiducial file (.mrk.json)")
    parser.add_argument("--start-point", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                         help="Optional position used to pick the start point among detected endpoints "
                              "(closest endpoint to this position becomes the start point). If omitted, "
                              "the network's own center of mass is used instead (a proxy for the main "
                              "trunk - see get_end_points()'s docstring).")
    parser.add_argument("--coordinate-system", choices=["LPS", "RAS"], default=None,
                         help="Coordinate system to store/tag in the output markups file. Default: LPS, "
                              "converting back from RAS if the input got flipped (see --flip-input-lps-to-ras) "
                              "- this matches Slicer's own convention of always storing .mrk.json in LPS even "
                              "though the live scene is RAS. Pass RAS to skip that back-conversion instead.")
    parser.add_argument("--preprocess", action="store_true",
                         help="Run clean/triangulate/normals before network extraction, matching Slicer's "
                              "'Preprocess input surface model' checkbox when it is CHECKED. Leave unset "
                              "(default) if that checkbox was OFF when you ran auto-detect in Slicer, since "
                              "in that case Slicer uses the raw surface unmodified (extractNetwork() still "
                              "does its own clean+triangulate internally either way, already replicated below).")
    parser.add_argument("--keep-all-components", action="store_true",
                         help="Do not filter out disconnected surface fragments before extraction. By default "
                              "only the largest connected component is kept, since stray debris/islands from "
                              "segmentation can otherwise attract the auto start point and produce endpoints "
                              "sitting on a detached piece instead of the main vessel tree.")
    parser.add_argument("--flip-input-lps-to-ras", dest="flip_input_lps_to_ras", action="store_true", default=None,
                         help="Force-negate X/Y of the input surface on load (LPS -> RAS). Default: auto-detect "
                              "from the .vtk header (Slicer writes 'SPACE=LPS'/'SPACE=RAS' on line 2); for "
                              ".vtp/.stl, which carry no such metadata, assume LPS and flip (Slicer's Models "
                              "module exporter is confirmed to always write LPS while the live scene is RAS).")
    parser.add_argument("--no-flip-input-lps-to-ras", dest="flip_input_lps_to_ras", action="store_false",
                         help="Force-disable the LPS->RAS flip, overriding auto-detection.")
    parser.add_argument("--endpoint-distance-radius-factor", type=float,
                         default=ENDPOINT_DISTANCE_RADIUS_FACTOR, metavar="F",
                         help="Drop any detected endpoint farther from the input surface than F times the "
                              "local vessel radius there (default: %(default)s). See "
                              "_filter_endpoints_near_surface()'s docstring for why an endpoint is "
                              "legitimately about one radius away from the surface - so this allowance "
                              "scales rather than being a fixed distance - and for why this matters "
                              "downstream.")
    parser.add_argument("--min-healthy-network-bbox-coverage", type=float,
                         default=MIN_HEALTHY_NETWORK_BBOX_COVERAGE, metavar="F",
                         help="Minimum bounding-box-diagonal coverage (network's own diagonal over the "
                              f"input surface's) for a network extraction to be trusted (default: %(default)s) "
                              "- see extract_network_robust()'s docstring.")
    parser.add_argument("--network-advancement-ratio", type=float,
                         default=NETWORK_ADVANCEMENT_RATIO, metavar="F",
                         help="vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio parameter "
                              "(default: %(default)s).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    endpointPositions = run(
        args.input_stl,
        args.output_mrk_json,
        start_point=args.start_point,
        coordinate_system=args.coordinate_system,
        preprocess=args.preprocess,
        keep_largest_component=not args.keep_all_components,
        flip_input_lps_to_ras=args.flip_input_lps_to_ras,
        endpoint_distance_radius_factor=args.endpoint_distance_radius_factor,
        min_healthy_network_bbox_coverage=args.min_healthy_network_bbox_coverage,
        network_advancement_ratio=args.network_advancement_ratio,
    )

    logger.info("Detected %d endpoints (first is start point):", len(endpointPositions))
    for i, position in enumerate(endpointPositions):
        marker = " (start)" if i == 0 else ""
        logger.info("  %d: %s%s", i + 1, position, marker)
    logger.info("Saved to %s", args.output_mrk_json)


if __name__ == "__main__":
    sys.exit(main())