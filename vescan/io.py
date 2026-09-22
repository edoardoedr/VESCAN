#!/usr/bin/env python3
"""
Shared surface I/O utilities for every stage under vescan/stages/.

Pure VTK, no vtkvmtk dependency - works identically as a plain CLI script or
inside the Slicer Python console.
"""

import logging
import os
import threading
import time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)


class Stage:
    """Context manager logging a start message and a periodic heartbeat
    (elapsed seconds) around a slow step, so long-running operations (big
    VTK/VMTK filter Update() calls, large per-point Python loops, ...) don't
    look hung. No true progress percentage is available for most of these
    steps, so this is elapsed-time feedback, not a real progress bar."""
    def __init__(self, label, heartbeat_seconds=5):
        self.label = label
        self.heartbeat_seconds = heartbeat_seconds
        self._stop_event = threading.Event()
        self._thread = None
        self._start = None

    def _heartbeat(self):
        while not self._stop_event.wait(self.heartbeat_seconds):
            elapsed = time.monotonic() - self._start
            logger.info("  ... %s still running (%.0fs elapsed)", self.label, elapsed)

    def __enter__(self):
        self._start = time.monotonic()
        logger.info("%s...", self.label)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            logger.info("  done (%.1fs)", elapsed)
        return False


def load_surface(surface_path):
    """Reads .stl, .vtk (legacy) or .vtp (XML) polydata.
    Prefer .vtk/.vtp over .stl when exporting from Slicer: STL stores
    coordinates as float32 and no shared vertices (just a triangle soup), so
    the reader has to reconstruct connectivity by merging near-coincident
    points within a tolerance. That reconstruction can fail at points that
    were exactly coincident in the source mesh, silently splitting it into
    spurious disconnected components. .vtk/.vtp store double-precision
    coordinates and explicit point/cell connectivity, so no reconstruction
    is needed and this artifact cannot occur."""
    ext = os.path.splitext(surface_path)[1].lower()
    if ext == ".stl":
        reader = vtk.vtkSTLReader()
    elif ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported surface file extension '{ext}' (expected .stl, .vtk or .vtp)")

    reader.SetFileName(surface_path)
    reader.Update()
    polyData = reader.GetOutput()
    if polyData.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read a valid surface from {surface_path}")
    return polyData


def save_surface(polyData, surface_path, coordinate_space="LPS"):
    """Writes .vtk (legacy) or .vtp (XML) polydata. For .vtk, embeds the
    coordinate space in the header the same way Slicer does ("3D Slicer
    output. SPACE=LPS"/"SPACE=RAS" on line 2), so detect_coordinate_space()
    picks it up automatically downstream. .vtp has no such header slot in
    this simple writer - callers reading a .vtp back should track/assume
    the space out of band (vescan.stages.endpoints assumes LPS for
    .vtp, matching Slicer's confirmed Models module export convention)."""
    ext = os.path.splitext(surface_path)[1].lower()
    if ext == ".vtk":
        writer = vtk.vtkPolyDataWriter()
        writer.SetHeader(f"3D Slicer output. SPACE={coordinate_space}")
    elif ext == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError(f"Unsupported surface file extension '{ext}' for writing (expected .vtk or .vtp)")

    writer.SetFileName(surface_path)
    writer.SetInputData(polyData)
    writer.Write()


def detect_coordinate_space(surface_path):
    """Best-effort detection of the coordinate space a Slicer-exported surface
    file was written in. Slicer's legacy .vtk writer embeds it directly in the
    header comment (e.g. "3D Slicer output. SPACE=LPS" on line 2) - confirmed
    to always be LPS while Slicer's live scene is RAS. .vtp/.stl carry no such
    metadata, so detection falls back to None (caller should assume LPS,
    matching Slicer's exporter, unless proven otherwise for that export path).
    Returns "LPS", "RAS" or None."""
    if os.path.splitext(surface_path)[1].lower() != ".vtk":
        return None
    with open(surface_path, "rb") as f:
        header = b"".join(f.readline() for _ in range(2)).decode("ascii", errors="ignore")
    if "SPACE=LPS" in header:
        return "LPS"
    if "SPACE=RAS" in header:
        return "RAS"
    return None


def flip_lps_ras(polyData):
    """Negates X and Y (RAS <-> LPS conversion; both are right-handed, only
    the sign of R/L and A/P flips, S is unchanged). Slicer's live scene is
    always in RAS; some exporters (observed: Models module .vtp export, at
    least in some configurations) write coordinates in LPS instead. If left
    uncorrected, any bounds-corner-based logic (e.g. ExtractCenterline's
    start-point heuristic) picks a different anatomical corner than the GUI
    does, cascading into different results even though the underlying
    geometry is the same."""
    transform = vtk.vtkTransform()
    transform.Scale(-1.0, -1.0, 1.0)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polyData)
    transformFilter.Update()
    return transformFilter.GetOutput()


def flip_positions_lps_ras(positions):
    """Same conversion as flip_lps_ras(), for plain [x, y, z] position lists
    (e.g. markups endpoints) instead of polydata."""
    return [[-p[0], -p[1], p[2]] for p in positions]


def check_point_near_surface(polyData, position, pointId, max_distance, description="point"):
    """Raises ValueError if `position` is farther than max_distance (mm) from
    polyData's pointId-th point - pointId is typically the result of a
    vtkPointLocator.FindClosestPoint(position) call, so this checks whether
    that "closest point" was actually close, i.e. whether `position` lies
    on/near the surface at all (FindClosestPoint() always returns SOME point
    id, even for a position floating arbitrarily far away, so it can't catch
    this by itself).

    This matters because an endpoint far from the surface (e.g. from a
    coordinate-space mismatch, or an endpoints file computed against a
    different/older version of the surface) doesn't fail cleanly in the
    VMTK filters that consume it - vtkvmtkPolyDataCenterlines' internal
    vtkvmtkSteepestDescentLineTracer instead logs a native "Seed id invalid"
    error and returns an empty centerline, which then segfaults
    vtkvmtkCenterlineBranchExtractor downstream (see
    vescan.stages.centerline's module docstring). Catching the bad
    endpoint here, before any of that runs, turns it into an immediate,
    clear, catchable Python error instead.

    Returns the actual distance (mm) if it's within max_distance."""
    closestPosition = polyData.GetPoint(pointId)
    distance = vtk.vtkMath.Distance2BetweenPoints(position, closestPosition) ** 0.5
    if distance > max_distance:
        raise ValueError(
            f"{description} at {list(position)} is {distance:.2f}mm from the nearest surface point "
            f"{list(closestPosition)} (max allowed: {max_distance}mm) - it does not lie on/near the "
            f"surface. Check for a coordinate-space (LPS/RAS) mismatch, or a stale endpoints file left "
            f"over from a previous run with different preprocessing/conversion parameters.")
    return distance


def keep_largest_connected_component(polyData, verbose=True):
    """Drops every connected component except the largest one. Segmentation
    exports frequently carry small disconnected debris/islands; if left in,
    a bounds-corner-based start point (or an unlucky manual one) can end up
    on a fragment instead of the main vessel tree."""
    labeler = vtk.vtkPolyDataConnectivityFilter()
    labeler.SetInputData(polyData)
    labeler.SetExtractionModeToAllRegions()
    labeler.ColorRegionsOn()
    labeler.Update()
    numberOfRegions = labeler.GetNumberOfExtractedRegions()

    if numberOfRegions > 1:
        regionIdArray = labeler.GetOutput().GetPointData().GetArray("RegionId")
        counts = {}
        for i in range(regionIdArray.GetNumberOfTuples()):
            rid = int(regionIdArray.GetValue(i))
            counts[rid] = counts.get(rid, 0) + 1
        sortedCounts = sorted(counts.values(), reverse=True)
        if verbose:
            logger.warning("input surface has %d disconnected components (point counts, largest "
                            "first): %s", numberOfRegions, sortedCounts)
            if len(sortedCounts) > 1 and sortedCounts[1] > 0.05 * sortedCounts[0]:
                logger.warning("  The second-largest component is not tiny relative to the largest one - "
                                "double check it isn't actually part of the vessel tree that got disconnected "
                                "(e.g. a segmentation gap) before trusting this filtering.")

    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputData(polyData)
    largest.SetExtractionModeToLargestRegion()
    largest.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(largest.GetOutput())
    cleaner.Update()
    result = cleaner.GetOutput()

    if verbose and numberOfRegions > 1:
        logger.info("  Kept largest component: %d / %d points.",
                    result.GetNumberOfPoints(), polyData.GetNumberOfPoints())

    return result


def sanitize_nan_arrays(polyData, replacement=0.0):
    """Replaces NaN/Inf values in every point-data and cell-data array of
    polyData IN PLACE, returning the number of values replaced.

    vtkvmtkCenterlineGeometry/vtkvmtkCenterlineBranchGeometry's Curvature/
    Torsion/FrenetNormal/FrenetBinormal arrays are mathematically undefined
    on straight or near-degenerate segments (division by a near-zero
    curvature) and come out as NaN instead of raising - a handful of NaN
    tuples then propagate into the saved .vtk file. Some readers (older VTK/
    Slicer legacy-ASCII parsers in particular) choke on the literal "nan"
    token and refuse to load the whole model, which is a much worse outcome
    than a locally undefined value, so this replaces them with `replacement`
    (0.0 by default) instead of leaving them in the output."""
    totalReplaced = 0
    for dataObject in (polyData.GetPointData(), polyData.GetCellData()):
        for i in range(dataObject.GetNumberOfArrays()):
            array = dataObject.GetArray(i)
            if array is None or array.GetDataType() not in (vtk.VTK_FLOAT, vtk.VTK_DOUBLE):
                continue
            # vtk_to_numpy() is a zero-copy view for float/double arrays (backed
            # by the same memory as `array`) - assigning into it mutates `array`
            # directly, no need to build/swap in a new vtkDataArray.
            values = vtk_to_numpy(array)
            badMask = ~np.isfinite(values)
            badCount = int(np.count_nonzero(badMask))
            if badCount == 0:
                continue
            totalReplaced += badCount
            logger.warning("array '%s' has %d non-finite (NaN/Inf) value(s) - replacing with %s.",
                            array.GetName(), badCount, replacement)
            values[badMask] = replacement
    return totalReplaced