#!/usr/bin/env python3
"""
Standalone vessel-surface clipping/trimming to a cut centerline tree.

Originally ported from SlicerExtension-VMTK's ClipVessel module
(SlicerExtension-VMTK/ClipVessel/ClipVessel.py), which clips a surface at a
handful of manually-picked points using vtkvmtkCenterlineSplitExtractor +
vtkvmtkPolyDataCenterlineGroupsClipper. That approach doesn't fit this
pipeline: vescan.stages.cut_graph can produce 100+ cut points on a
single tree, and vtkvmtkCenterlineSplitExtractor recomputes/renumbers
GroupIds from scratch on every call (confirmed empirically) while
vtkvmtkPolyDataCenterlineGroupsClipper becomes impractically slow once the
number of specified groups gets into the hundreds (confirmed empirically:
20+ minutes and still not done on real data with ~700 kept groups) - both
assumptions the original module's interactive, few-points-at-a-time
workflow never has to face.

This script instead clips geometrically, using vescan.stages.build_graph's
and vescan.stages.cut_graph's outputs directly:
  - build_dropped_centerline() takes the branch-split centerline (with
    stable GroupIds - build_graph's --split-output, 05_branch_tree.vtk)
    and the cut centerline (only the surviving groups -
    cut_graph's output_centerline) and computes their set difference: the
    groups that got trimmed away.
  - classify_and_clip_surface() then, for every surface point, compares its
    distance to the nearest point on the KEPT centerline vs. the nearest
    point on the DROPPED centerline. A point closer to the kept side is
    kept; a point closer to the dropped side is discarded. A triangle
    survives only if all three of its points do.
  - The largest connected component of what's left is kept (drops any
    stray sliver where the kept/dropped boundary was ambiguous), optionally
    capped and/or extended with flow extensions.
  - Finally, GroupId/IsBifurcation/Generation/CellId are transferred straight
    onto the clipped surface (transfer_centerline_labels.transfer_labels(),
    using splitCenterlinePolyData - already loaded here for
    build_dropped_centerline() above, no extra file to read) unless
    label_arrays=() - so the saved output is already the labeled clipped
    surface. There used to be a separate stage producing
    08_labeled_clipped_surface.vtk from this same output right afterwards;
    that was two on-disk copies of the same mesh (one plain, one +4 point
    arrays) for no reason, so it's gone - this file is now the only one.

This sidesteps GroupIds renumbering entirely (no vtkvmtkCenterlineSplitExtractor
call) and vtkvmtkPolyDataCenterlineGroupsClipper's group-count blowup (no
call to it at all) - just two point locators and a threshold, which is fast
regardless of how many branches got trimmed.

Trade-off vs. the original module: the cut edge follows the surface's own
triangulation (jagged at the boundary) rather than a precise plane through
the exact cut point - cut_graph's cut points aren't used for geometric
precision here, only indirectly (they're where the kept/dropped GroupIds
split happened). Use --cap to close the resulting boundary into a smooth
cap if a watertight surface is what you need downstream.

Requires the same vtkvmtk build as the rest of this pipeline (see
conda_env in the JSON config) - only for --cap/--add-flow-extensions
(vtkvmtkCapPolyData/vtkvmtkPolyDataFlowExtensionsFilter); the core
clipping is pure VTK.

CAVEAT: validated against synthetic geometry and against real patient data
for plausibility (correct trunk retained, correct branches trimmed,
correct point counts) - not against a Slicer/ClipVessel run (no Slicer
available in this environment) since ClipVessel.py's own algorithm turned
out not to be a good fit here. Inspect the output surface (e.g. in ParaView
or Slicer) before trusting it for anything downstream.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.clip_vessel pipeline_output/01_preprocessed.vtk \\
        pipeline_output/05_branch_tree.vtk pipeline_output/06_cut_centerline.vtk \\
        pipeline_output/07_clipped_surface.vtk --cap
"""

import argparse
import logging
import sys

import vtk

try:
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
except ImportError:
    # Standalone VMTK build: compiled modules live inside the `vmtk` package.
    from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry

from vescan.io import load_surface, save_surface, detect_coordinate_space, flip_lps_ras
from vescan.stages.cut_graph import _extract_cells
from vescan.stages.network import _add_cell_id_array, CELL_ID_ARRAY_NAME
from vescan.stages.transfer_centerline_labels import transfer_labels, DEFAULT_ARRAY_NAMES as DEFAULT_LABEL_ARRAYS

logger = logging.getLogger(__name__)

RADIUS_ARRAY_NAME = "Radius"
SPLIT_GROUP_IDS_ARRAY_NAME = "GroupIds"  # cell array on the branch-split centerline (05_branch_tree.vtk)
CUT_GROUP_ID_ARRAY_NAME = "GroupId"      # cell array on cut_graph's output (06_cut_centerline.vtk)
CLIP_SCALAR_ARRAY_NAME = "KeptDroppedSignedDistance"

DEFAULT_EXTENSION_LENGTH = 5.0  # matches ClipVesselLogic.setDefaultParameters()


def read_kept_group_ids(cutCenterlinePolyData):
    """The set of GroupIds present in cut_graph's output - every group that
    survived the --max-generations cutoff."""
    groupIdArray = cutCenterlinePolyData.GetCellData().GetArray(CUT_GROUP_ID_ARRAY_NAME)
    if groupIdArray is None:
        raise ValueError(f"Cut centerline has no '{CUT_GROUP_ID_ARRAY_NAME}' cell array - pass "
                          f"cut_graph's output_centerline (06_cut_centerline.vtk)")
    return {int(groupIdArray.GetValue(c)) for c in range(cutCenterlinePolyData.GetNumberOfCells())}


def build_dropped_centerline(splitCenterlinePolyData, keptGroupIds):
    """The complement of keptGroupIds within the full branch-split
    centerline: every cell belonging to a group that did NOT survive the
    --max-generations cutoff. Together with the (already-available) kept
    centerline, this is what classify_and_clip_surface() compares surface
    points against."""
    groupIdArray = splitCenterlinePolyData.GetCellData().GetArray(SPLIT_GROUP_IDS_ARRAY_NAME)
    if groupIdArray is None:
        raise ValueError(f"Split centerline has no '{SPLIT_GROUP_IDS_ARRAY_NAME}' cell array - pass "
                          f"build_graph's --split-output (05_branch_tree.vtk)")
    droppedCellIds = [c for c in range(splitCenterlinePolyData.GetNumberOfCells())
                       if int(groupIdArray.GetValue(c)) not in keptGroupIds]
    return _extract_cells(splitCenterlinePolyData, droppedCellIds)


def classify_and_clip_surface(surfacePolyData, keptCenterlinePolyData, droppedCenterlinePolyData):
    """Clips surfacePolyData at the boundary between 'closer to
    keptCenterlinePolyData' and 'closer to droppedCenterlinePolyData'.

    Uses vtkClipPolyData on a continuous per-point scalar (signed distance
    difference) rather than vtkThreshold on a discrete kept/dropped flag -
    vtkClipPolyData actually cuts through triangles that straddle the
    boundary (inserting new vertices at the zero-crossing), instead of
    keeping/discarding whole triangles based on their vertices. The old
    threshold approach made the cut edge follow the mesh's own
    triangulation (jagged, "staircase" boundary); this one follows the true
    geometric boundary at sub-triangle precision.

    Distances are measured to the centerlines' CELLS (vtkStaticCellLocator -
    nearest point on the actual polyline segments, continuous along their
    length), not to their POINTS (vtkStaticPointLocator would snap to the
    nearest sample point, which has a small "step" every time the nearest
    sample changes - confirmed to leave a visibly wavy cut edge even after
    switching threshold->clip, since the underlying scalar field itself
    wasn't smooth)."""
    if droppedCenterlinePolyData.GetNumberOfPoints() == 0:
        logger.info("Dropped centerline is empty (nothing to trim) - returning the surface unchanged.")
        return surfacePolyData

    keptLocator = vtk.vtkStaticCellLocator()
    keptLocator.SetDataSet(keptCenterlinePolyData)
    keptLocator.BuildLocator()
    droppedLocator = vtk.vtkStaticCellLocator()
    droppedLocator.SetDataSet(droppedCenterlinePolyData)
    droppedLocator.BuildLocator()

    numPoints = surfacePolyData.GetNumberOfPoints()
    scalarArray = vtk.vtkDoubleArray()
    scalarArray.SetName(CLIP_SCALAR_ARRAY_NAME)
    scalarArray.SetNumberOfValues(numPoints)

    nKept = 0
    point = [0.0, 0.0, 0.0]
    closestPoint = [0.0, 0.0, 0.0]
    cellId, subId, dist2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
    for i in range(numPoints):
        surfacePolyData.GetPoint(i, point)
        keptLocator.FindClosestPoint(point, closestPoint, cellId, subId, dist2)
        keptDist = dist2.get() ** 0.5
        droppedLocator.FindClosestPoint(point, closestPoint, cellId, subId, dist2)
        droppedDist = dist2.get() ** 0.5
        # negative => closer to the kept centerline, positive => closer to the dropped one
        signedDistance = keptDist - droppedDist
        scalarArray.SetValue(i, signedDistance)
        nKept += signedDistance <= 0.0
    logger.info("%d/%d surface points are on the kept side (before cutting through boundary triangles).",
                nKept, numPoints)

    scalarSurface = vtk.vtkPolyData()
    scalarSurface.DeepCopy(surfacePolyData)
    scalarSurface.GetPointData().SetScalars(scalarArray)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(scalarSurface)
    clipper.SetValue(0.0)
    clipper.GenerateClipScalarsOff()
    clipper.SetInsideOut(1)  # keep the side where the scalar is <= value (closer to the kept centerline)
    clipper.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(clipper.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()


def keep_largest_region(polyData):
    """Single-pass largest-connected-component extraction. Drops any stray
    sliver left over where the kept/dropped classification was ambiguous
    (e.g. right at a bifurcation, where the kept trunk and a dropped
    sibling branch are geometrically close)."""
    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputData(polyData)
    largest.SetExtractionModeToLargestRegion()
    largest.Update()
    return largest.GetOutput()


def cap_surface(surface):
    """Caps every open hole in `surface` - mirrors ClipVesselLogic.capSurface()."""
    surfaceCapper = vtkvmtkComputationalGeometry.vtkvmtkCapPolyData()
    surfaceCapper.SetInputData(surface)
    surfaceCapper.SetDisplacement(0.0)
    surfaceCapper.SetInPlaneDisplacement(0.0)
    surfaceCapper.Update()
    return surfaceCapper.GetOutput()


def extend_vessel(surfacePolyData, centerlinesPolyData, extension_length, extension_mode):
    """Adds flow extensions to every open boundary of `surfacePolyData` -
    mirrors ClipVesselLogic.extendVessel(), with the same fixed defaults for
    the parameters the Slicer GUI never exposed as widgets."""
    extensionsFilter = vtkvmtkComputationalGeometry.vtkvmtkPolyDataFlowExtensionsFilter()
    extensionsFilter.SetInputData(surfacePolyData)
    extensionsFilter.SetCenterlines(centerlinesPolyData)
    extensionsFilter.SetSigma(1)
    extensionsFilter.SetAdaptiveExtensionLength(0)
    extensionsFilter.SetAdaptiveExtensionRadius(1)
    extensionsFilter.SetAdaptiveNumberOfBoundaryPoints(0)
    extensionsFilter.SetExtensionLength(extension_length)
    extensionsFilter.SetExtensionRatio(2)
    extensionsFilter.SetExtensionRadius(1)
    extensionsFilter.SetTransitionRatio(0.25)
    extensionsFilter.SetCenterlineNormalEstimationDistanceRatio(1.0)
    extensionsFilter.SetNumberOfBoundaryPoints(50)
    if extension_mode == "centerlinedirection":
        extensionsFilter.SetExtensionModeToUseCenterlineDirection()
    elif extension_mode == "boundarynormal":
        extensionsFilter.SetExtensionModeToUseNormalToBoundary()
    if extension_mode == "linear":
        extensionsFilter.SetInterpolationModeToLinear()
    elif extension_mode == "thinplatespline":
        extensionsFilter.SetInterpolationModeToThinPlateSpline()
    extensionsFilter.Update()
    return extensionsFilter.GetOutput()


def clip_vessel(surfacePolyData, splitCenterlinePolyData, cutCenterlinePolyData, cap=False,
                 add_flow_extensions=False, extension_length=DEFAULT_EXTENSION_LENGTH,
                 extension_mode="boundarynormal", label_arrays=DEFAULT_LABEL_ARRAYS):
    keptGroupIds = read_kept_group_ids(cutCenterlinePolyData)
    logger.info("%d group(s) kept (from the cut centerline).", len(keptGroupIds))

    droppedCenterlinePolyData = build_dropped_centerline(splitCenterlinePolyData, keptGroupIds)
    logger.info("Dropped centerline: %d cell(s), %d point(s).",
                droppedCenterlinePolyData.GetNumberOfCells(), droppedCenterlinePolyData.GetNumberOfPoints())

    surface = classify_and_clip_surface(surfacePolyData, cutCenterlinePolyData, droppedCenterlinePolyData)
    logger.info("Clipped surface (pre-cleanup): %d points, %d cells.",
                surface.GetNumberOfPoints(), surface.GetNumberOfCells())

    surface = keep_largest_region(surface)
    logger.info("After keeping the largest connected region: %d points, %d cells.",
                surface.GetNumberOfPoints(), surface.GetNumberOfCells())

    if add_flow_extensions:
        surface = extend_vessel(surface, cutCenterlinePolyData, extension_length, extension_mode)

    if cap:
        surface = cap_surface(surface)

    if label_arrays:
        if CELL_ID_ARRAY_NAME in label_arrays and \
                splitCenterlinePolyData.GetCellData().GetArray(CELL_ID_ARRAY_NAME) is None:
            _add_cell_id_array(splitCenterlinePolyData)
        surface = transfer_labels(surface, splitCenterlinePolyData, array_names=label_arrays)
        logger.info("Transferred %s onto the clipped surface (see transfer_centerline_labels.transfer_labels()) - "
                    "no separate labeled-clipped-surface file needed.", list(label_arrays))

    result = vtk.vtkPolyData()
    result.DeepCopy(surface)
    return result


def run(surface_path, split_centerline_path, cut_centerline_path, output_surface_path,
        cap=False, add_flow_extensions=False, extension_length=DEFAULT_EXTENSION_LENGTH,
        extension_mode="boundarynormal", label_arrays=DEFAULT_LABEL_ARRAYS):
    logger.info("Loading surface %s...", surface_path)
    surfaceSpace = detect_coordinate_space(surface_path) or "LPS"
    surfacePolyData = load_surface(surface_path)
    logger.info("  -> %d points, %d cells (%s).",
                surfacePolyData.GetNumberOfPoints(), surfacePolyData.GetNumberOfCells(), surfaceSpace)

    logger.info("Loading split centerline %s...", split_centerline_path)
    splitSpace = detect_coordinate_space(split_centerline_path) or "LPS"
    splitCenterlinePolyData = load_surface(split_centerline_path)
    if splitSpace != surfaceSpace:
        logger.info("Split centerline is tagged %s, surface is %s - converting to match.",
                    splitSpace, surfaceSpace)
        splitCenterlinePolyData = flip_lps_ras(splitCenterlinePolyData)
    logger.info("  -> %d points, %d cells.",
                splitCenterlinePolyData.GetNumberOfPoints(), splitCenterlinePolyData.GetNumberOfCells())

    logger.info("Loading cut centerline %s...", cut_centerline_path)
    cutSpace = detect_coordinate_space(cut_centerline_path) or "LPS"
    cutCenterlinePolyData = load_surface(cut_centerline_path)
    if cutSpace != surfaceSpace:
        logger.info("Cut centerline is tagged %s, surface is %s - converting to match.", cutSpace, surfaceSpace)
        cutCenterlinePolyData = flip_lps_ras(cutCenterlinePolyData)
    logger.info("  -> %d points, %d cells.",
                cutCenterlinePolyData.GetNumberOfPoints(), cutCenterlinePolyData.GetNumberOfCells())

    result = clip_vessel(
        surfacePolyData, splitCenterlinePolyData, cutCenterlinePolyData, cap=cap,
        add_flow_extensions=add_flow_extensions, extension_length=extension_length, extension_mode=extension_mode,
        label_arrays=label_arrays)

    save_surface(result, output_surface_path, coordinate_space=surfaceSpace)
    logger.info("Saved clipped surface (%d points, %d cells) to %s",
                result.GetNumberOfPoints(), result.GetNumberOfCells(), output_surface_path)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_surface", help="Path to the surface to clip (.vtk/.vtp) - normally "
                                               "01_preprocessed.vtk, the same surface the centerline was built from")
    parser.add_argument("split_centerline", help="Path to the branch-split centerline (.vtk/.vtp) with a "
                                                  "'GroupIds' cell array - build_graph's --split-output "
                                                  "(05_branch_tree.vtk)")
    parser.add_argument("cut_centerline", help="Path to the cut centerline (.vtk/.vtp) with a 'GroupId' cell "
                                                "array - cut_graph's output_centerline "
                                                "(06_cut_centerline.vtk)")
    parser.add_argument("output_surface", help="Path to save the clipped surface (.vtk/.vtp)")
    parser.add_argument("--cap", action="store_true", help="Cap every open hole left by the clip")
    parser.add_argument("--add-flow-extensions", action="store_true",
                         help="Add flow extensions at every open boundary after clipping")
    parser.add_argument("--extension-length", type=float, default=DEFAULT_EXTENSION_LENGTH,
                         help=f"Flow extension length in mm (default {DEFAULT_EXTENSION_LENGTH}, only used with "
                              f"--add-flow-extensions)")
    parser.add_argument("--extension-mode", choices=["centerlinedirection", "boundarynormal", "linear",
                                                       "thinplatespline"], default="boundarynormal",
                         help="Flow extension direction/interpolation mode (default boundarynormal, matching "
                              "ClipVessel's GUI default; only used with --add-flow-extensions)")
    parser.add_argument("--label-arrays", default=",".join(DEFAULT_LABEL_ARRAYS),
                         help=f"Comma-separated split_centerline cell arrays to transfer onto the clipped surface "
                              f"before saving it (default: {','.join(DEFAULT_LABEL_ARRAYS)}) - pass an empty "
                              f"string to save a plain, unlabeled clipped surface instead")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()
    label_arrays = tuple(a.strip() for a in args.label_arrays.split(",") if a.strip())

    run(
        args.input_surface, args.split_centerline, args.cut_centerline, args.output_surface,
        cap=args.cap, add_flow_extensions=args.add_flow_extensions,
        extension_length=args.extension_length, extension_mode=args.extension_mode,
        label_arrays=label_arrays,
    )


if __name__ == "__main__":
    sys.exit(main())