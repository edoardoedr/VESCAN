#!/usr/bin/env python3
"""
Standalone centerline extraction for vascular/airway surfaces.

Replicates ExtractCenterlineLogic.extractCenterline() +
createCurveTreeFromCenterline() (SlicerExtension-VMTK/ExtractCenterline/
ExtractCenterline.py) - the "Apply" button's CenterlineModel/VoronoiDiagram/
CenterlineCurve/CenterlineProperties outputs - without requiring 3D Slicer.
This is the accurate, slower centerline computation (vtkvmtkPolyDataCenterlines),
as opposed to the fast approximate network extraction in
vescan.stages.network.

Input: a preprocessed surface (see vescan.stages.preprocess) and a
Slicer Markups fiducial file with >= 2 endpoints (see
vescan.stages.endpoints). Outputs:
  - centerline model (.vtk/.vtp): raw vtkvmtkPolyDataCenterlines output,
    matching Slicer's "CenterlineModel" node - NOT saved on a normal
    successful run (nothing downstream reads it; the merged/curve outputs
    below are what's actually used), only written out as a diagnostic
    artifact if extraction fails (produces 0 points) - see run()'s own
    docstring.
  - Voronoi diagram (.vtk/.vtp, optional): matching "VoronoiDiagram".
  - merged/branch-split centerline (.vtk/.vtp, optional): per-branch
    GroupIds/CenterlineIds/TractIds/Blanking/Radius, resampled at
    curve-sampling-distance - the raw polydata Slicer builds internally to
    generate CenterlineCurve/CenterlineProperties (not saved as its own
    node in the GUI). vtkvmtkMergeCenterlines can emit more than one cell
    for the same anatomical branch (duplicated/overlapping tracts kept only
    for its own internal Blanking/TractIds bookkeeping) - loading this file
    directly as a plain model can show spurious straight segments fanning
    out from the start point. Kept mainly for property computation/
    debugging; use the centerline curve output below to actually view
    individual branches.
  - centerline curve (.mrk.json, optional): one open curve markup per
    branch, matching Slicer's "CenterlineCurve" node. Built by walking the
    merged centerline's branch tree exactly like
    addCenterlineCurves()/_addCenterline() (vtkThreshold by GroupIds +
    recursive walk, deduplicating repeated cells) - this is the clean,
    non-overlapping decomposition, unlike the raw merged/branch-split file.
  - per-branch properties CSV (optional): CellId, AverageRadius, Length,
    Curvature, Torsion, Tortuosity, Start/EndPointPosition - the content of
    Slicer's "CenterlineProperties" table.

Works both as a standalone VMTK build (`from vmtk import ...`) and inside the
3D Slicer Python console/environment (bare `import vtkvmtk*Python`), like the
other stages in this pipeline.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.centerline preprocessed.vtk endpoints.mrk.json centerline_debug.vtp
    python -m vescan.stages.centerline preprocessed.vtk endpoints.mrk.json centerline_debug.vtp \\
        --voronoi-output voronoi.vtp --merged-output centerline_branches_raw.vtp \\
        --properties-csv branch_properties.csv --centerline-curve branch_curves.mrk.json

Usage (from Python - main.py, a notebook, or the Slicer Python console):
    import sys
    sys.path.append("/path/to/vmtk_building")
    from vescan.stages import centerline
    centerline.run(
        "preprocessed.vtk",
        "endpoints.mrk.json",
        "centerline_debug.vtp",
    )
"""

import argparse
import csv
import json
import logging
import sys
import time

import vtk

try:
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
except ImportError:
    # Standalone VMTK build: compiled modules live inside the `vmtk` package.
    from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry

from vescan.io import load_surface, save_surface, detect_coordinate_space, flip_positions_lps_ras, Stage
from vescan.stages.network import load_endpoints_markups, build_curve_markups, CELL_ID_ARRAY_NAME

logger = logging.getLogger(__name__)

RADIUS_ARRAY_NAME = "Radius"

# A centerline lying further outside its own surface than this is not a
# centerline - see fraction_outside_surface() for the real failure it catches
# and the measurements behind both numbers.
MAX_CENTERLINE_OUTSIDE_FRACTION = 0.25
WARN_CENTERLINE_OUTSIDE_FRACTION = 0.10
# Points sampled for that check; the fractions it distinguishes are far apart
# enough that sampling costs nothing in sensitivity.
CENTERLINE_OUTSIDE_SAMPLE_SIZE = 5000

# vtkvmtkPolyDataCenterlines' own default, mirrored here because
# build_interior_tessellation() takes over building the tessellation and must
# build the same one. Note it is a FRACTION of the bounding box diagonal, not
# a distance: 0.001 is about 0.4mm on a chest-sized surface.
DELAUNAY_TOLERANCE = 0.001


BLANKING_ARRAY_NAME = "Blanking"
GROUP_IDS_ARRAY_NAME = "GroupIds"
CENTERLINE_IDS_ARRAY_NAME = "CenterlineIds"
TRACT_IDS_ARRAY_NAME = "TractIds"
LENGTH_ARRAY_NAME = "Length"
CURVATURE_ARRAY_NAME = "Curvature"
TORSION_ARRAY_NAME = "Torsion"
TORTUOSITY_ARRAY_NAME = "Tortuosity"

DEFAULT_CURVE_SAMPLING_DISTANCE = 1.0
NEIGHBOR_MATCH_TOLERANCE = 1e-3  # mm, see _walk_centerline_tree()


def fraction_outside_surface(centerlinePolyData, surfacePolyData,
                              sample_size=CENTERLINE_OUTSIDE_SAMPLE_SIZE):
    """Fraction of the centerline's points that fall outside surfacePolyData,
    estimated on an evenly-spaced sample.

    A centerline is the medial axis of its surface, so essentially all of it
    must be inside. Measured on healthy runs the figure is 0.25-4.28% (the
    upper end on airway trees, whose coarser meshes make the containment test
    noisier near the wall), never more.

    It exists because of a real failure that produced a centerline running up
    to 440mm outside the patient while still looking like a tree: a vein
    surface came out of preprocessing inside-out, and since
    vtkvmtkPolyDataCenterlines reads the surface normals to decide which
    Delaunay tetrahedra are INSIDE the vessel, the selection inverted and
    99.92% of the Voronoi diagram - and then 100% of the centerline - ended up
    outside. Nothing else caught it: the extraction returned 22584 points, so
    the existing 0-point guard stayed silent, and every downstream stage
    happily produced plausible-looking numbers from it.

    Measuring the centerline rather than the Voronoi is deliberate: it is the
    thing that matters, it is 15x smaller, and it catches the symptom whatever
    the cause. The gap between healthy and broken here is a factor of 23, so
    the thresholds are nowhere near either."""
    numPoints = centerlinePolyData.GetNumberOfPoints()
    if numPoints == 0 or surfacePolyData is None or surfacePolyData.GetNumberOfPoints() == 0:
        return 0.0

    step = max(1, numPoints // sample_size)
    sampledIds = range(0, numPoints, step)
    points = vtk.vtkPoints()
    for pointId in sampledIds:
        points.InsertNextPoint(centerlinePolyData.GetPoint(pointId))
    sampled = vtk.vtkPolyData()
    sampled.SetPoints(points)

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(sampled)
    enclosed.SetSurfaceData(surfacePolyData)
    enclosed.Update()

    total = points.GetNumberOfPoints()
    outside = sum(1 for i in range(total) if not enclosed.IsInside(i))
    return outside / total if total else 0.0


def _distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def needs_flipped_normals(cappedSurfacePolyData):
    """True when vtkvmtkPolyDataCenterlines would orient this surface's
    normals INWARD, and so must be told to flip them.

    The filter does not use the normals it is handed: it recomputes them
    itself (vtkvmtkPolyDataCenterlines.cxx, `surfaceNormals->
    AutoOrientNormalsOn()`), and then reads the result to decide which
    Delaunay tetrahedra lie INSIDE the vessel
    (vtkvmtkInternalTetrahedraExtractor's OutwardNormalsArrayName). So an
    orientation fixed anywhere upstream - preprocess included - is simply
    overwritten, and AutoOrientNormals is documented to be unreliable on a
    mesh with non-manifold edges, which decimation puts into every surface
    this pipeline produces.

    When it guesses wrong the selection inverts completely. Measured on a
    real vein surface: 99.92% of the resulting Voronoi diagram outside the
    body, inscribed-sphere radii to 734mm against a true maximum near 12mm,
    and a centerline reaching 440mm past the patient. With the flip, the same
    surface gives 0.83% outside and a 18.47mm maximum - the healthy reference
    on another patient measures 0.56% and 12.05mm.

    Detection replays exactly what the filter does - the same
    vtkPolyDataNormals settings on the same capped surface - and reads the
    SIGNED VOLUME of the result: negative means the triangles ended up facing
    inward. Exact, parameter-free, and needs no Delaunay tessellation, so it
    costs a fraction of a second. On nine real structures it fires on exactly
    the one that is broken."""
    replay = vtk.vtkPolyDataNormals()
    replay.SetInputData(cappedSurfacePolyData)
    replay.SplittingOff()
    replay.AutoOrientNormalsOn()
    replay.SetFlipNormals(0)
    replay.ComputePointNormalsOn()
    replay.ConsistencyOn()
    replay.Update()

    from vescan.stages.preprocess import signed_volume
    volume = signed_volume(replay.GetOutput())
    return volume < 0.0, volume


def build_interior_tessellation(cappedSurfacePolyData, flipNormals=False,
                                 delaunayTolerance=DELAUNAY_TOLERANCE):
    """The Delaunay tessellation vtkvmtkPolyDataCenterlines would build for
    this surface, with the tetrahedra whose circumcentre falls OUTSIDE the
    surface removed.

    Every step up to the removal replicates the filter's own code exactly
    (vtkvmtkPolyDataCenterlines.cxx: vtkPolyDataNormals with
    AutoOrientNormalsOn -> vtkDelaunay3D at DelaunayTolerance ->
    vtkvmtkInternalTetrahedraExtractor keyed on the normals array), which was
    verified by checking that the unfiltered result reproduces the filter's
    own Voronoi diagram to the digit on real data.

    The removal is the point. A Voronoi vertex IS a tetrahedron's
    circumcentre - vtkvmtkVoronoiDiagram3D writes one per cell, in order - so
    a tetrahedron whose circumcentre lies outside the vessel contributes a
    Voronoi vertex outside the vessel, and the centerline tracer, which walks
    the Voronoi, can then wander out of the patient. VMTK's own interior test
    cannot catch these: it is local, judging a tetrahedron by the normals at
    its four corners, which sit in a perfectly ordinary patch of mesh even
    when the circumcentre they imply is half a metre away. A containment test
    is global and catches exactly what the local one misses.

    Measured on the surface that motivated this: 2255 of 266801 tetrahedra
    removed, 0.85%, taking the Voronoi's largest inscribed-sphere radius from
    365mm to 18.54mm - a plausible venous confluence. The centerline is a path
    through Voronoi vertices, so once they are all inside the surface it is
    inside by construction.

    ALL POINTS ARE KEPT, including any left unused: the tessellation's point
    ids are the surface's own point ids, and vtkvmtkVoronoiDiagram3D builds
    PoleIds against them to translate the source/target seed ids the caller
    passes. Renumbering the points - which vtkExtractCells would do - would
    silently repoint every seed."""
    surfaceNormals = vtk.vtkPolyDataNormals()
    surfaceNormals.SetInputData(cappedSurfacePolyData)
    surfaceNormals.SplittingOff()
    surfaceNormals.AutoOrientNormalsOn()
    surfaceNormals.SetFlipNormals(1 if flipNormals else 0)
    surfaceNormals.ComputePointNormalsOn()
    surfaceNormals.ConsistencyOn()
    surfaceNormals.Update()
    orientedSurface = surfaceNormals.GetOutput()

    delaunay = vtk.vtkDelaunay3D()
    delaunay.CreateDefaultLocator()
    delaunay.SetInputData(orientedSurface)
    delaunay.SetTolerance(delaunayTolerance)
    delaunay.Update()

    interiorExtractor = vtkvmtkComputationalGeometry.vtkvmtkInternalTetrahedraExtractor()
    interiorExtractor.SetInputConnection(delaunay.GetOutputPort())
    interiorExtractor.SetOutwardNormalsArrayName(orientedSurface.GetPointData().GetNormals().GetName())
    interiorExtractor.Update()
    tessellation = interiorExtractor.GetOutput()

    numberOfCells = tessellation.GetNumberOfCells()
    circumcentres = vtk.vtkPoints()
    circumcentres.SetNumberOfPoints(numberOfCells)
    centre = [0.0, 0.0, 0.0]
    cellPointIds = vtk.vtkIdList()
    for cellId in range(numberOfCells):
        tessellation.GetCellPoints(cellId, cellPointIds)
        vtk.vtkTetra.Circumsphere(tessellation.GetPoint(cellPointIds.GetId(0)),
                                   tessellation.GetPoint(cellPointIds.GetId(1)),
                                   tessellation.GetPoint(cellPointIds.GetId(2)),
                                   tessellation.GetPoint(cellPointIds.GetId(3)), centre)
        circumcentres.SetPoint(cellId, centre)

    probe = vtk.vtkPolyData()
    probe.SetPoints(circumcentres)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(probe)
    enclosed.SetSurfaceData(cappedSurfacePolyData)
    enclosed.Update()

    filtered = vtk.vtkUnstructuredGrid()
    filtered.SetPoints(tessellation.GetPoints())
    filtered.GetPointData().ShallowCopy(tessellation.GetPointData())
    filtered.Allocate(numberOfCells)
    for cellId in range(numberOfCells):
        if enclosed.IsInside(cellId):
            tessellation.GetCellPoints(cellId, cellPointIds)
            filtered.InsertNextCell(vtk.VTK_TETRA, cellPointIds)

    removed = numberOfCells - filtered.GetNumberOfCells()
    if removed:
        logger.info("Dropped %d of %d Delaunay tetrahedra (%.2f%%) whose circumcentre lies outside the "
                    "surface - each one would have put a Voronoi vertex outside the vessel for the "
                    "centerline tracer to wander into.", removed, numberOfCells, 100.0 * removed / numberOfCells)
    return filtered


def _run_centerline_filter(tubePolyData, sourceIdList, targetIds, curveSamplingDistance, simplifyVoronoi,
                            resampleBeforeSplit, flipNormals=False, delaunayTessellation=None):
    """Builds and runs a single vtkvmtkPolyDataCenterlines call for the given
    target point ids (plain ints, not a vtkIdList - see _find_pathological_
    target_ids() below, which needs to slice/recombine target subsets).
    Returns the filter itself (still holding its output/Voronoi diagram)."""
    targetIdList = vtk.vtkIdList()
    for targetId in targetIds:
        targetIdList.InsertNextId(targetId)

    centerlineFilter = vtkvmtkComputationalGeometry.vtkvmtkPolyDataCenterlines()
    centerlineFilter.SetInputData(tubePolyData)
    centerlineFilter.SetSourceSeedIds(sourceIdList)
    centerlineFilter.SetTargetSeedIds(targetIdList)
    centerlineFilter.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    centerlineFilter.SetCostFunction('1/R')  # prefer paths through points with large radius
    centerlineFilter.SetFlipNormals(bool(flipNormals))
    if delaunayTessellation is not None:
        # Use the tessellation we filtered ourselves rather than letting the
        # filter build its own - see build_interior_tessellation().
        centerlineFilter.GenerateDelaunayTessellationOff()
        centerlineFilter.SetDelaunayTessellation(delaunayTessellation)
    centerlineFilter.SetAppendEndPointsToCenterlines(0)
    centerlineFilter.SetSimplifyVoronoi(simplifyVoronoi)
    if resampleBeforeSplit:
        centerlineFilter.SetCenterlineResampling(1)
        centerlineFilter.SetResamplingStepLength(resampleBeforeSplit)
    else:
        centerlineFilter.SetCenterlineResampling(0)
        centerlineFilter.SetResamplingStepLength(curveSamplingDistance)
    centerlineFilter.Update()
    return centerlineFilter


def _centerline_succeeds(tubePolyData, sourceIdList, targetIds, curveSamplingDistance, simplifyVoronoi,
                          resampleBeforeSplit, flipNormals=False, delaunayTessellation=None):
    output = _run_centerline_filter(tubePolyData, sourceIdList, targetIds, curveSamplingDistance,
                                     simplifyVoronoi, resampleBeforeSplit, flipNormals,
                                     delaunayTessellation).GetOutput()
    return output is not None and output.GetNumberOfPoints() > 0


def _find_pathological_target_ids(tubePolyData, sourceIdList, targetIds, curveSamplingDistance,
                                   simplifyVoronoi, resampleBeforeSplit, flipNormals=False,
                                   delaunayTessellation=None):
    """Bisects targetIds (a plain list of point ids) to find the ones
    individually responsible for a 0-point centerline result - see
    extract_centerline()'s own retry logic for why this is needed: a single
    pathological seed (empirically, an extreme/near-degenerate branch tip -
    e.g. right where vtkvmtkCapPolyData's cap over that tip's own tiny
    opening comes out numerically degenerate) can corrupt
    vtkvmtkPolyDataCenterlines' entire internal Voronoi computation, not
    just fail to reach that one target - so the WHOLE run silently returns
    0 points instead of failing only for the offending target(s).

    Assumes contamination behaves as confirmed empirically: a subset
    succeeds iff it contains NONE of the pathological ids; the whole set
    failing means at least one of its ids is pathological. This does not
    catch the (unobserved so far) case of two individually-fine targets
    that only fail when combined - each recursive half is tested entirely
    on its own, not against ids already found "good" elsewhere."""
    if len(targetIds) == 1:
        return list(targetIds)
    mid = len(targetIds) // 2
    left, right = targetIds[:mid], targetIds[mid:]
    badIds = []
    for half in (left, right):
        if not _centerline_succeeds(tubePolyData, sourceIdList, half, curveSamplingDistance,
                                     simplifyVoronoi, resampleBeforeSplit, flipNormals,
                                     delaunayTessellation):
            badIds.extend(_find_pathological_target_ids(tubePolyData, sourceIdList, half,
                                                          curveSamplingDistance, simplifyVoronoi,
                                                          resampleBeforeSplit, flipNormals,
                                                          delaunayTessellation))
    return badIds


def extract_centerline(surfacePolyData, endpoints, curveSamplingDistance=DEFAULT_CURVE_SAMPLING_DISTANCE,
                        simplifyVoronoi=False, resampleBeforeSplit=None, delaunayTolerance=DELAUNAY_TOLERANCE):
    """Mirrors ExtractCenterlineLogic.extractCenterline().
    endpoints: list of (position, selected) tuples in the SAME coordinate
    space as surfacePolyData (unselected point(s) = source/start, selected
    = targets - same convention as Slicer's markups). At least 2 required.
    simplifyVoronoi: Slicer computes this from its own version
    (`majorVersion*100+minorVersion < 413`), which is False for every
    current Slicer release (5.x) - Voronoi smoothing is broken under VTK9
    (https://github.com/vmtk/SlicerExtension-VMTK/issues/34). Default False
    to match modern Slicer; exposed here only for completeness.
    resampleBeforeSplit: None (default) keeps Slicer's own behavior
    (CenterlineResampling off - see CenterlineConfig.resample_before_split's
    docstring for why). A number (mm) turns on vtkvmtkPolyDataCenterlines'
    own internal resampling at that step length, applied to the raw output
    BEFORE split_centerline() ever sees it - cuts its points-per-cell, the
    other factor (besides cell count) driving vtkvmtkCenterlineBranchExtractor's
    cost. Changes what output_centerline_path contains (no longer an exact
    match of Slicer's raw CenterlineModel).

    If the computation comes back with 0 points (a real, empirically
    observed vtkvmtkPolyDataCenterlines/vtkvmtkSteepestDescentLineTracer
    failure mode: a single pathological target seed - typically an
    extreme/near-degenerate branch tip whose own cap comes out numerically
    degenerate - can corrupt the ENTIRE internal Voronoi computation,
    logging a generic native "Seed id invalid" error instead of merely
    failing to reach that one target), this bisects the target list (see
    _find_pathological_target_ids()) to find and exclude the offending
    id(s), then retries once with the rest - instead of returning an empty
    centerline (see run()'s own 0-point guard) for what is usually still a
    perfectly good structure once that one bad target is dropped.
    Returns (centerlinePolyData, voronoiDiagramPolyData)."""
    if endpoints is None or len(endpoints) < 2:
        raise ValueError("At least two endpoints are needed for centerline extraction")

    # Cap all holes in the mesh that are not marked as endpoints.
    capDisplacement = 0.0
    surfaceCapper = vtkvmtkComputationalGeometry.vtkvmtkCapPolyData()
    surfaceCapper.SetInputData(surfacePolyData)
    surfaceCapper.SetDisplacement(capDisplacement)
    surfaceCapper.SetInPlaneDisplacement(capDisplacement)
    with Stage("Capping surface holes"):
        surfaceCapper.Update()
    tubePolyData = surfaceCapper.GetOutput()

    flipNormals, replayedVolume = needs_flipped_normals(tubePolyData)
    if flipNormals:
        logger.warning("vtkvmtkPolyDataCenterlines would orient this surface's normals INWARD (replayed "
                        "signed volume %.1fmm3) - passing FlipNormals=1. Left alone it would keep the "
                        "Delaunay tetrahedra OUTSIDE the vessel and trace the centerline out of the "
                        "patient; see needs_flipped_normals().", replayedVolume)

    with Stage("Building interior Delaunay tessellation"):
        delaunayTessellation = build_interior_tessellation(tubePolyData, flipNormals=flipNormals,
                                                             delaunayTolerance=delaunayTolerance)

    foundStartPoint = any(not selected for _position, selected in endpoints)

    sourceIdList = vtk.vtkIdList()
    targetIds = []

    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(tubePolyData)
    pointLocator.BuildLocator()

    for controlPointIndex, (position, selected) in enumerate(endpoints):
        isTarget = selected
        if not foundStartPoint and controlPointIndex == 0:
            # If no start point found then use the first point as source
            isTarget = False
        pointId = pointLocator.FindClosestPoint(position)
        if isTarget:
            targetIds.append(pointId)
        else:
            sourceIdList.InsertNextId(pointId)

    with Stage("Computing centerline (vtkvmtkPolyDataCenterlines - the slow step, can take several minutes)"):
        centerlineFilter = _run_centerline_filter(tubePolyData, sourceIdList, targetIds, curveSamplingDistance,
                                                   simplifyVoronoi, resampleBeforeSplit, flipNormals,
                                                   delaunayTessellation)

    if centerlineFilter.GetOutput() is not None and centerlineFilter.GetOutput().GetNumberOfPoints() == 0 \
            and len(targetIds) > 1:
        logger.warning("Centerline computation produced 0 points with all %d target(s) - bisecting the "
                        "target list to find and exclude the pathological one(s) (see extract_centerline()'s "
                        "own docstring for why this happens).", len(targetIds))
        with Stage("Bisecting targets to isolate pathological seed(s)"):
            badTargetIds = _find_pathological_target_ids(tubePolyData, sourceIdList, targetIds,
                                                           curveSamplingDistance, simplifyVoronoi,
                                                           resampleBeforeSplit, flipNormals,
                                                           delaunayTessellation)
        if not badTargetIds or len(badTargetIds) >= len(targetIds):
            raise RuntimeError("Centerline extraction produced 0 points and bisection could not isolate a "
                                "specific pathological target (every target may be affected, or the source "
                                "itself is the problem) - inspect the endpoints manually.")
        logger.warning("Excluding %d pathological target endpoint(s) (surface point id(s): %s) and "
                        "retrying.", len(badTargetIds), badTargetIds)
        goodTargetIds = [targetId for targetId in targetIds if targetId not in badTargetIds]
        with Stage("Retrying centerline computation with pathological target(s) excluded"):
            centerlineFilter = _run_centerline_filter(tubePolyData, sourceIdList, goodTargetIds,
                                                       curveSamplingDistance, simplifyVoronoi, resampleBeforeSplit,
                                                       flipNormals, delaunayTessellation)

    if not centerlineFilter.GetOutput():
        raise ValueError("Failed to compute centerline (no output was generated)")
    centerlinePolyData = vtk.vtkPolyData()
    centerlinePolyData.DeepCopy(centerlineFilter.GetOutput())

    if not centerlineFilter.GetVoronoiDiagram():
        raise ValueError("Failed to compute centerline (no Voronoi diagram was generated)")
    voronoiDiagramPolyData = vtk.vtkPolyData()
    voronoiDiagramPolyData.DeepCopy(centerlineFilter.GetVoronoiDiagram())

    return centerlinePolyData, voronoiDiagramPolyData


def split_centerline(centerlinePolyData):
    """Runs vtkvmtkCenterlineBranchExtractor - the expensive step (see module
    docstring), shared with vescan.stages.build_graph's own
    ensure_split() fallback. Splits the raw overlapping source->target paths
    into GroupIds/CenterlineIds/TractIds/Blanking-tagged tracts, BEFORE
    vtkvmtkMergeCenterlines' resampling - which destroys the per-cell
    CenterlineIds/TractIds correspondence build_graph.py's own topology
    lookup (FindAdjacentCenterlineGroupIds) needs, see that module's
    docstring, so this pre-merge output is the only one reusable there.

    Exposed here (and cacheable via run()'s split_output_path) so main.py
    can feed this straight into stage 5 (build_graph) instead of having it
    recompute the exact same ~15-20 minute filter from scratch on the raw
    centerline - this used to happen twice (once here, once in
    build_centerline_graph.py's own run_branch_extractor()) every single
    pipeline run."""
    if centerlinePolyData.GetNumberOfPoints() == 0:
        # vtkvmtkCenterlineBranchExtractor segfaults (not merely raises) on
        # 0-point/0-cell input - run() already guards its own call with this
        # same check right after extraction, but this function is also
        # reachable from build_graph.py's ensure_split() fallback, which may
        # load an empty centerline file left over from an earlier failed run
        # - guard here too so every caller gets a clean, catchable failure.
        raise RuntimeError("split_centerline() got an empty centerline (0 points) - refusing to run "
                            "vtkvmtkCenterlineBranchExtractor on it (it would segfault on empty input).")

    # vtkvmtkCenterlineBranchExtractor's own cost scales with cell count AND
    # points-per-cell (see this module's docstring) - logged right before the
    # expensive call so a slow run can be diagnosed from this number alone,
    # without cross-referencing the earlier "Saved centerline" line.
    logger.info("Extracting branches from a centerline with %d points, %d cells",
                centerlinePolyData.GetNumberOfPoints(), centerlinePolyData.GetNumberOfCells())
    branchExtractor = vtkvmtkComputationalGeometry.vtkvmtkCenterlineBranchExtractor()
    branchExtractor.SetInputData(centerlinePolyData)
    branchExtractor.SetBlankingArrayName(BLANKING_ARRAY_NAME)
    branchExtractor.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    branchExtractor.SetGroupIdsArrayName(GROUP_IDS_ARRAY_NAME)
    branchExtractor.SetCenterlineIdsArrayName(CENTERLINE_IDS_ARRAY_NAME)
    branchExtractor.SetTractIdsArrayName(TRACT_IDS_ARRAY_NAME)
    with Stage("Extracting branches (vtkvmtkCenterlineBranchExtractor)"):
        branchExtractor.Update()
    return branchExtractor.GetOutput()


def create_merged_centerline(centerlinePolyData, curveSamplingDistance=DEFAULT_CURVE_SAMPLING_DISTANCE,
                              splitCenterlines=None):
    """Mirrors the branch-extraction/merge half of
    ExtractCenterlineLogic.createCurveTreeFromCenterline(): splits the raw
    centerline into per-branch groups (see split_centerline(), skipped if
    splitCenterlines is already given) and resamples/merges them. This is
    the polydata Slicer builds internally to generate CenterlineCurve/
    CenterlineProperties (not exposed as its own node in the GUI)."""
    if splitCenterlines is None:
        splitCenterlines = split_centerline(centerlinePolyData)

    mergeCenterlines = vtkvmtkComputationalGeometry.vtkvmtkMergeCenterlines()
    mergeCenterlines.SetInputData(splitCenterlines)
    mergeCenterlines.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    mergeCenterlines.SetGroupIdsArrayName(GROUP_IDS_ARRAY_NAME)
    mergeCenterlines.SetCenterlineIdsArrayName(CENTERLINE_IDS_ARRAY_NAME)
    mergeCenterlines.SetTractIdsArrayName(TRACT_IDS_ARRAY_NAME)
    mergeCenterlines.SetBlankingArrayName(BLANKING_ARRAY_NAME)
    mergeCenterlines.SetResamplingStepLength(curveSamplingDistance)
    mergeCenterlines.SetMergeBlanked(True)
    with Stage("Merging centerlines (vtkvmtkMergeCenterlines)"):
        mergeCenterlines.Update()

    return mergeCenterlines.GetOutput()


def _threshold_by_group_id(mergedCenterlines, groupId):
    """Extracts the polydata subset for a single GroupIds value, mirroring the
    vtkAssignAttribute + vtkThreshold combo in ExtractCenterlineLogic._addCenterline()."""
    assignAttribute = vtk.vtkAssignAttribute()
    assignAttribute.SetInputData(mergedCenterlines)
    assignAttribute.Assign(GROUP_IDS_ARRAY_NAME, vtk.vtkDataSetAttributes.SCALARS, vtk.vtkAssignAttribute.CELL_DATA)

    thresholder = vtk.vtkThreshold()
    thresholder.SetInputConnection(assignAttribute.GetOutputPort())
    thresholder.SetLowerThreshold(groupId - 0.5)
    thresholder.SetUpperThreshold(groupId + 0.5)
    thresholder.Update()
    return thresholder.GetOutput()


def _walk_centerline_tree(mergedCenterlines, cellId, processedCellIds, branches, cellIds, reverse=False):
    """Mirrors ExtractCenterlineLogic._addCenterline(), with two deliberate
    deviations, both confirmed necessary empirically (see
    NEIGHBOR_MATCH_TOLERANCE and the reverse handling below - without them
    this walk silently stopped after 1-2 cells out of ~600 on a real
    centerline, even though a plain spatial connectivity check showed ~99%
    of the cells were genuinely part of one connected tree):

    1. Adjacency is decided by SPATIAL proximity (endpoint position within
       NEIGHBOR_MATCH_TOLERANCE) instead of exact point-ID equality. Slicer's
       own _addCenterline() uses ID equality (`endPointIndex != ...GetPointIds
       ().GetId(0)`), but vtkvmtkMergeCenterlines inserts every cell's points
       independently - confirmed empirically two cells meeting at the exact
       same physical location can get different point IDs there. This is the
       same point-ID-continuity issue build_centerline_graph.py's module
       docstring documents for vtkvmtkMergeCenterlines output generally.
    2. A candidate neighbor is checked in BOTH orientations - its first point
       AND its last point - against the current branch's end point, not just
       its first point. vtkvmtkMergeCenterlines does not guarantee a cell's
       point order runs "source to target" consistently for every cell;
       confirmed empirically some cells only connect at what would be their
       "wrong" end under a first-point-only test. When a cell matches at its
       LAST point instead of its first, it's walked with reverse=True and its
       own position list is reversed before being appended, so `branches`
       still reads each curve start-to-end in actual traversal order.

    Recursively walks the merged/branch-split centerline starting at cellId,
    appending one branch (list of point positions, in traversal order) per
    visited cellId to `branches` (and that cellId to `cellIds`, same index),
    skipping cells already visited - Slicer's per-visited-cellId curve
    creation, including its same-GroupId-visited-twice edge case (see below).

    This dedup-by-processedCellIds (not by GroupId) matters because
    vtkvmtkMergeCenterlines can emit more than one raw cell for the same
    anatomical branch (duplicated/overlapping tracts it keeps only for its
    own internal Blanking/TractIds bookkeeping). Serializing mergedCenterlines
    as-is (e.g. the --merged-output file) keeps those extra cells, which is
    why loading it directly as a plain model in Slicer can show spurious
    straight segments fanning out from the start point - Slicer itself never
    displays that raw polydata; it always walks it exactly like this to build
    the CenterlineCurve node."""
    groupIdsArray = mergedCenterlines.GetCellData().GetArray(GROUP_IDS_ARRAY_NAME)
    groupId = groupIdsArray.GetValue(cellId)
    groupPolyData = _threshold_by_group_id(mergedCenterlines, groupId)
    if groupPolyData.GetNumberOfCells() > 0:
        points = groupPolyData.GetCell(0).GetPoints()
        positions = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        if reverse:
            positions.reverse()
        branches.append(positions)
        cellIds.append(cellId)

    processedCellIds.append(cellId)
    cellPoints = mergedCenterlines.GetCell(cellId).GetPointIds()
    endPointId = cellPoints.GetId(0) if reverse else cellPoints.GetId(cellPoints.GetNumberOfIds() - 1)
    endPointPosition = mergedCenterlines.GetPoint(endPointId)
    numberOfCells = mergedCenterlines.GetNumberOfCells()
    for neighborCellIndex in range(numberOfCells):
        if neighborCellIndex in processedCellIds:
            continue
        neighborPointIds = mergedCenterlines.GetCell(neighborCellIndex).GetPointIds()
        neighborFirstPosition = mergedCenterlines.GetPoint(neighborPointIds.GetId(0))
        if _distance(endPointPosition, neighborFirstPosition) <= NEIGHBOR_MATCH_TOLERANCE:
            _walk_centerline_tree(mergedCenterlines, neighborCellIndex, processedCellIds, branches, cellIds,
                                   reverse=False)
            continue
        neighborLastPosition = mergedCenterlines.GetPoint(neighborPointIds.GetId(neighborPointIds.GetNumberOfIds() - 1))
        if _distance(endPointPosition, neighborLastPosition) <= NEIGHBOR_MATCH_TOLERANCE:
            _walk_centerline_tree(mergedCenterlines, neighborCellIndex, processedCellIds, branches, cellIds,
                                   reverse=True)


def extract_centerline_branches(mergedCenterlines):
    """Returns (branches, cellIds): one list of [x, y, z] positions and one
    matching mergedCenterlines cellId per visited branch, in walk order - the
    same tree walk ExtractCenterlineLogic.addCenterlineCurves() uses to build
    the CenterlineCurve node's tree of curves. cellIds lets callers name/tag
    each branch with the same cellId Slicer uses (see build_curve_markups())."""
    branches = []
    cellIds = []
    _walk_centerline_tree(mergedCenterlines, 0, [], branches, cellIds)
    return branches, cellIds


def save_centerline_curve_markups(mergedCenterlines, output_path, coordinate_space="LPS", base_name="branch"):
    """Writes the deduplicated branch decomposition from extract_centerline_branches()
    to a Slicer Markups .mrk.json file (one open curve per branch) - the
    standalone equivalent of Slicer's "CenterlineCurve" output. Curve names
    use the actual mergedCenterlines cellId (matching Slicer's
    addCenterlineCurves()/_addCenterline() naming), so a curve named
    "branch (37)" here corresponds to CellId 37 in the CenterlineProperties
    CSV (see compute_branch_properties()), exactly as in Slicer."""
    branches, cellIds = extract_centerline_branches(mergedCenterlines)
    data = build_curve_markups(branches, coordinate_space=coordinate_space, base_name=base_name, indices=cellIds)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return branches


def save_centerline_curve_model(mergedCenterlines, output_path, coordinate_space="LPS"):
    """Writes the same deduplicated branch decomposition as
    save_centerline_curve_markups(), but as a SINGLE multi-cell .vtk/.vtp
    (one polyline per branch) instead of one Markups Curve node per branch.
    Loading hundreds of separate Curve nodes in Slicer is slow and clutters
    the scene; this is one Model node instead, with a per-cell "CellId"
    array (same trick as vescan.stages.network's NetworkModel) so
    Slicer can still color each branch differently: Models module > Display >
    Scalars > Active Scalar = CellId, Cell location, "Random" color table -
    same per-branch-color effect as the separate curve nodes gave, one node
    instead of many."""
    branches, cellIds = extract_centerline_branches(mergedCenterlines)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    cellIdArray = vtk.vtkIntArray()
    cellIdArray.SetName(CELL_ID_ARRAY_NAME)
    for branchCellId, positions in zip(cellIds, branches):
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(positions))
        for i, position in enumerate(positions):
            pointId = points.InsertNextPoint(position)
            polyLine.GetPointIds().SetId(i, pointId)
        lines.InsertNextCell(polyLine)
        cellIdArray.InsertNextValue(branchCellId)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    polyData.GetCellData().AddArray(cellIdArray)
    save_surface(polyData, output_path, coordinate_space=coordinate_space)
    return branches


def compute_branch_properties(mergedCenterlines):
    """Mirrors the table-building half of
    ExtractCenterlineLogic.createCurveTreeFromCenterline(): per-branch-cell
    CellId, average radius, length/curvature/torsion/tortuosity, and
    start/end point positions. Returns a list of dicts, one per cell."""
    numberOfCells = mergedCenterlines.GetNumberOfCells()

    pointDataToCellData = vtk.vtkPointDataToCellData()
    pointDataToCellData.SetInputData(mergedCenterlines)
    pointDataToCellData.ProcessAllArraysOff()
    pointDataToCellData.AddPointDataArray(RADIUS_ARRAY_NAME)
    with Stage("Computing average branch radius"):
        pointDataToCellData.Update()
    averageRadiusArray = pointDataToCellData.GetOutput().GetCellData().GetArray(RADIUS_ARRAY_NAME)

    centerlineBranchGeometry = vtkvmtkComputationalGeometry.vtkvmtkCenterlineBranchGeometry()
    centerlineBranchGeometry.SetInputData(mergedCenterlines)
    centerlineBranchGeometry.SetRadiusArrayName(RADIUS_ARRAY_NAME)
    centerlineBranchGeometry.SetGroupIdsArrayName(GROUP_IDS_ARRAY_NAME)
    centerlineBranchGeometry.SetBlankingArrayName(BLANKING_ARRAY_NAME)
    centerlineBranchGeometry.SetLengthArrayName(LENGTH_ARRAY_NAME)
    centerlineBranchGeometry.SetCurvatureArrayName(CURVATURE_ARRAY_NAME)
    centerlineBranchGeometry.SetTorsionArrayName(TORSION_ARRAY_NAME)
    centerlineBranchGeometry.SetTortuosityArrayName(TORTUOSITY_ARRAY_NAME)
    centerlineBranchGeometry.SetLineSmoothing(False)
    with Stage("Computing branch geometry (length/curvature/torsion/tortuosity)"):
        centerlineBranchGeometry.Update()
    centerlineProperties = centerlineBranchGeometry.GetOutput()

    rows = []
    for cellIndex in range(numberOfCells):
        pointIds = mergedCenterlines.GetCell(cellIndex).GetPointIds()
        startPointPosition = [0.0, 0.0, 0.0]
        endPointPosition = [0.0, 0.0, 0.0]
        if pointIds.GetNumberOfIds() > 0:
            mergedCenterlines.GetPoint(pointIds.GetId(0), startPointPosition)
        if pointIds.GetNumberOfIds() > 1:
            mergedCenterlines.GetPoint(pointIds.GetId(pointIds.GetNumberOfIds() - 1), endPointPosition)
        else:
            endPointPosition = startPointPosition

        rows.append({
            "CellId": cellIndex,
            "AverageRadius": averageRadiusArray.GetValue(cellIndex) if averageRadiusArray else None,
            "Length": centerlineProperties.GetPointData().GetArray(LENGTH_ARRAY_NAME).GetValue(cellIndex),
            "Curvature": centerlineProperties.GetPointData().GetArray(CURVATURE_ARRAY_NAME).GetValue(cellIndex),
            "Torsion": centerlineProperties.GetPointData().GetArray(TORSION_ARRAY_NAME).GetValue(cellIndex),
            "Tortuosity": centerlineProperties.GetPointData().GetArray(TORTUOSITY_ARRAY_NAME).GetValue(cellIndex),
            "StartPointR": startPointPosition[0], "StartPointA": startPointPosition[1], "StartPointS": startPointPosition[2],
            "EndPointR": endPointPosition[0], "EndPointA": endPointPosition[1], "EndPointS": endPointPosition[2],
        })
    return rows


def save_branch_properties_csv(rows, csv_path):
    fieldnames = ["CellId", "AverageRadius", "Length", "Curvature", "Torsion", "Tortuosity",
                  "StartPointR", "StartPointA", "StartPointS", "EndPointR", "EndPointA", "EndPointS"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(surface_path, endpoints_path, output_centerline_path,
        curve_sampling_distance=DEFAULT_CURVE_SAMPLING_DISTANCE,
        voronoi_output_path=None, merged_output_path=None, properties_csv_path=None,
        centerline_curve_path=None, centerline_curve_model_path=None, simplify_voronoi=False,
        split_output_path=None, resample_before_split=None, delaunay_tolerance=DELAUNAY_TOLERANCE,
        max_centerline_outside_fraction=MAX_CENTERLINE_OUTSIDE_FRACTION,
        warn_centerline_outside_fraction=WARN_CENTERLINE_OUTSIDE_FRACTION):
    """Extracts the centerline from surface_path/endpoints_path, plus
    whichever optional outputs are requested. output_centerline_path (the
    raw vtkvmtkPolyDataCenterlines output) is NOT saved on a normal
    successful run - see module docstring - it's only written to that path
    as a diagnostic artifact if extraction fails (0 points).
    split_output_path (optional): also saves the pre-merge branch-split
    centerline (see split_centerline()) - pass this to
    vescan.stages.build_graph.run() as its own input to skip its
    otherwise-redundant re-run of the same ~15-20 minute filter.
    resample_before_split (optional, mm): see CenterlineConfig.resample_before_split's
    docstring (vescan/config.py) - None keeps output_centerline_path an
    exact match of Slicer's raw CenterlineModel; a number trades that away to
    cut vtkvmtkCenterlineBranchExtractor's points-per-cell cost.
    Returns (centerlinePolyData, voronoiDiagramPolyData)."""
    pipelineStart = time.monotonic()

    # Surface and endpoints must share the same coordinate space, same reasoning
    # as vescan.stages.network (FindClosestPoint() needs consistent coordinates).
    surfaceSpace = detect_coordinate_space(surface_path) or "LPS"
    surfacePolyData = load_surface(surface_path)

    endpoints, endpointsSpace = load_endpoints_markups(endpoints_path)
    if endpointsSpace != surfaceSpace:
        logger.info("Endpoints are tagged %s, surface is %s - converting endpoints to match the "
                    "surface's coordinate space.", endpointsSpace, surfaceSpace)
        flippedPositions = flip_positions_lps_ras([position for position, _selected in endpoints])
        endpoints = list(zip(flippedPositions, [selected for _position, selected in endpoints]))

    centerlinePolyData, voronoiDiagramPolyData = extract_centerline(
        surfacePolyData, endpoints, curveSamplingDistance=curve_sampling_distance,
        simplifyVoronoi=simplify_voronoi, resampleBeforeSplit=resample_before_split,
        delaunayTolerance=delaunay_tolerance)

    if voronoi_output_path:
        save_surface(voronoiDiagramPolyData, voronoi_output_path, coordinate_space=surfaceSpace)
        logger.info("Saved Voronoi diagram (%d points) to %s",
                    voronoiDiagramPolyData.GetNumberOfPoints(), voronoi_output_path)

    if centerlinePolyData.GetNumberOfPoints() == 0:
        # vtkvmtkPolyDataCenterlines/vtkvmtkSteepestDescentLineTracer can fail
        # "silently" from Python's point of view: on a bad seed (e.g. an
        # endpoint that doesn't sit on/near surface_path) it just logs a
        # native ERR and returns an empty polydata instead of raising. The
        # raw (empty) output is ONLY saved here, in this failure path - see
        # module docstring - it's not a usable centerline, and in the normal
        # (success) case nothing downstream reads it (centerline_merged/
        # centerline_curve below are what's actually used), so saving it
        # every run would just be clutter. Raising here turns the failure
        # into a normal, logged Python error instead of silently feeding
        # empty geometry into split_centerline()'s
        # vtkvmtkCenterlineBranchExtractor below, which segfaults (not
        # merely raises) on 0-point/0-cell input - a crash no try/except,
        # here or in any caller, could ever catch.
        save_surface(centerlinePolyData, output_centerline_path, coordinate_space=surfaceSpace)
        raise RuntimeError(
            f"Centerline extraction produced 0 points for '{surface_path}' - check that "
            f"'{endpoints_path}' contains endpoints that actually lie on/near this surface. "
            f"The (empty) raw output was saved to '{output_centerline_path}' for inspection.")

    logger.info("Extracted centerline (%d points, %d cells)",
                centerlinePolyData.GetNumberOfPoints(), centerlinePolyData.GetNumberOfCells())

    outsideFraction = fraction_outside_surface(centerlinePolyData, surfacePolyData)
    if outsideFraction > max_centerline_outside_fraction:
        save_surface(centerlinePolyData, output_centerline_path, coordinate_space=surfaceSpace)
        raise RuntimeError(
            f"Centerline extraction produced a centerline lying {100 * outsideFraction:.1f}% OUTSIDE "
            f"'{surface_path}' (at most {100 * max_centerline_outside_fraction:.0f}% is tolerated; healthy "
            f"runs measure 0.2-4.3%). The usual cause is a surface whose normals point inward, which makes "
            f"vtkvmtkPolyDataCenterlines keep the Delaunay tetrahedra OUTSIDE the vessel instead of inside "
            f"- see vescan.stages.preprocess.ensure_outward_normals(), which is meant to prevent "
            f"exactly that. The unusable centerline was saved to '{output_centerline_path}' for inspection.")
    if outsideFraction > warn_centerline_outside_fraction:
        logger.warning("Centerline lies %.1f%% outside the surface - above the 0.2-4.3%% seen on healthy "
                        "runs, though below the %.0f%% failure threshold. Worth a look in Slicer.",
                        100 * outsideFraction, 100 * max_centerline_outside_fraction)

    splitCenterlines = None
    if split_output_path or merged_output_path or properties_csv_path or centerline_curve_path or centerline_curve_model_path:
        splitCenterlines = split_centerline(centerlinePolyData)

    if split_output_path:
        save_surface(splitCenterlines, split_output_path, coordinate_space=surfaceSpace)
        logger.info("Saved branch-split centerline (%d cells) to %s - reusable as-is by build_graph.run(), "
                    "skipping its own branch-extractor re-run", splitCenterlines.GetNumberOfCells(), split_output_path)

    mergedCenterlines = None
    if merged_output_path or properties_csv_path or centerline_curve_path or centerline_curve_model_path:
        mergedCenterlines = create_merged_centerline(centerlinePolyData, curveSamplingDistance=curve_sampling_distance,
                                                       splitCenterlines=splitCenterlines)

    if merged_output_path:
        save_surface(mergedCenterlines, merged_output_path, coordinate_space=surfaceSpace)
        logger.info("Saved merged/branch centerline (%d branches) to %s (raw vtkvmtkMergeCenterlines output - "
                    "may contain overlapping internal-bookkeeping cells; use --centerline-curve for a clean, "
                    "deduplicated view).", mergedCenterlines.GetNumberOfCells(), merged_output_path)

    if properties_csv_path:
        rows = compute_branch_properties(mergedCenterlines)
        save_branch_properties_csv(rows, properties_csv_path)
        logger.info("Saved branch properties (%d branches) to %s", len(rows), properties_csv_path)

    if centerline_curve_path:
        branches = save_centerline_curve_markups(mergedCenterlines, centerline_curve_path, coordinate_space=surfaceSpace)
        logger.info("Saved centerline curve (%d branches) to %s", len(branches), centerline_curve_path)

    if centerline_curve_model_path:
        branches = save_centerline_curve_model(mergedCenterlines, centerline_curve_model_path, coordinate_space=surfaceSpace)
        logger.info("Saved centerline curve model (%d branches, 1 file) to %s - load as a single Model in Slicer "
                    "instead of %d separate Curve nodes; color per-branch via Display > Scalars > Active Scalar = "
                    "CellId (Cell location), 'Random' color table", len(branches), centerline_curve_model_path,
                    len(branches))

    logger.info("Total time: %.1fs", time.monotonic() - pipelineStart)
    return centerlinePolyData, voronoiDiagramPolyData


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_surface", help="Path to the preprocessed input surface (.vtk or .vtp)")
    parser.add_argument("endpoints_markups", help="Path to the endpoints Slicer Markups fiducial file (.mrk.json), "
                                                    "at least 2 control points required")
    parser.add_argument("output_centerline", help="Path to the output centerline model (.vtk or .vtp), "
                                                     "matching Slicer's CenterlineModel")
    parser.add_argument("--curve-sampling-distance", type=float, default=DEFAULT_CURVE_SAMPLING_DISTANCE,
                         help=f"Resampling step length in mm (Slicer's CurveSamplingDistance, "
                              f"default {DEFAULT_CURVE_SAMPLING_DISTANCE}).")
    parser.add_argument("--voronoi-output", default=None,
                         help="Optional path to save the Voronoi diagram (.vtk or .vtp), matching VoronoiDiagram.")
    parser.add_argument("--merged-output", default=None,
                         help="Optional path to save the raw merged/branch-split centerline (.vtk or .vtp). "
                              "Internal vtkvmtkMergeCenterlines output, kept for property computation/debugging - "
                              "can contain overlapping bookkeeping cells, so prefer --centerline-curve for viewing.")
    parser.add_argument("--properties-csv", default=None,
                         help="Optional path to save per-branch properties as CSV, matching CenterlineProperties.")
    parser.add_argument("--centerline-curve", default=None,
                         help="Optional path to save the centerline as a Slicer Markups .mrk.json with one open "
                              "curve per branch, matching Slicer's CenterlineCurve output (see "
                              "addCenterlineCurves()/_addCenterline()). This is the deduplicated, tree-walked "
                              "decomposition - the correct way to view individual branches, unlike --merged-output. "
                              "Loads as one Curve node per branch in Slicer (can be hundreds) - prefer "
                              "--centerline-curve-model for fast loading/an overview.")
    parser.add_argument("--centerline-curve-model", default=None,
                         help="Optional path to save the SAME deduplicated per-branch decomposition as "
                              "--centerline-curve, but as a single multi-cell .vtk/.vtp (one Model node, one cell "
                              "per branch, with a 'CellId' cell array) instead of one Markups Curve node per "
                              "branch - much faster to load, colorable per-branch in Slicer via Display > Scalars "
                              "(Active Scalar = CellId, Cell location, Random color table)")
    parser.add_argument("--simplify-voronoi", action="store_true",
                         help="Enable Voronoi diagram simplification. Off by default, matching every current "
                              "Slicer release - this VMTK feature is broken under VTK9 (see module docstring).")
    parser.add_argument("--split-output", default=None,
                         help="Optional path to also save the pre-merge branch-split centerline (see "
                              "split_centerline()) - pass this file as build_centerline_graph.py's own input "
                              "to skip its otherwise-redundant vtkvmtkCenterlineBranchExtractor re-run.")
    parser.add_argument("--resample-before-split", type=float, default=None, metavar="MM",
                         help="Resample the raw centerline to this step length (mm) BEFORE running "
                              "vtkvmtkCenterlineBranchExtractor, cutting its points-per-cell cost. Off by "
                              "default, matching Slicer (output_centerline stays an exact match of Slicer's "
                              "raw CenterlineModel) - enabling this trades that exact match away. Validate "
                              "branch/graph topology on a few patients before relying on it.")
    parser.add_argument("--delaunay-tolerance", type=float, default=DELAUNAY_TOLERANCE, metavar="F",
                         help="Delaunay tessellation tolerance, as a fraction of the surface's bounding-box "
                              f"diagonal (default: {DELAUNAY_TOLERANCE}, matching vtkvmtkPolyDataCenterlines' "
                              "own default).")
    parser.add_argument("--max-centerline-outside-fraction", type=float,
                         default=MAX_CENTERLINE_OUTSIDE_FRACTION, metavar="F",
                         help="Above this fraction of centerline points lying outside the surface, extraction "
                              f"is treated as failed (default: {MAX_CENTERLINE_OUTSIDE_FRACTION}) - see "
                              "fraction_outside_surface()'s docstring.")
    parser.add_argument("--warn-centerline-outside-fraction", type=float,
                         default=WARN_CENTERLINE_OUTSIDE_FRACTION, metavar="F",
                         help="Above this fraction (but below --max-centerline-outside-fraction), a warning "
                              f"is logged instead of failing (default: {WARN_CENTERLINE_OUTSIDE_FRACTION}).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.input_surface,
        args.endpoints_markups,
        args.output_centerline,
        curve_sampling_distance=args.curve_sampling_distance,
        voronoi_output_path=args.voronoi_output,
        merged_output_path=args.merged_output,
        properties_csv_path=args.properties_csv,
        centerline_curve_path=args.centerline_curve,
        centerline_curve_model_path=args.centerline_curve_model,
        simplify_voronoi=args.simplify_voronoi,
        split_output_path=args.split_output,
        resample_before_split=args.resample_before_split,
        delaunay_tolerance=args.delaunay_tolerance,
        max_centerline_outside_fraction=args.max_centerline_outside_fraction,
        warn_centerline_outside_fraction=args.warn_centerline_outside_fraction,
    )


if __name__ == "__main__":
    sys.exit(main())