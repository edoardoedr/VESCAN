#!/usr/bin/env python3
"""Standalone surface preprocessing for vascular/airway surfaces.

Replicates ExtractCenterlineLogic.preprocess() (SlicerExtension-VMTK/
ExtractCenterline/ExtractCenterline.py) without requiring 3D Slicer: decimate
to a target point count, clean, triangulate, optionally subdivide once, and
recompute consistent normals.

Requires the `pyfqmr` package for the decimation step (`pip install pyfqmr`)
- clean/triangulate/normals stay pure VTK. Runs as a plain CLI script or
inside the Slicer Python console (as long as pyfqmr is importable there too).
Meant to run BEFORE autofind_endpoints.py in the pipeline (its output feeds
directly into that stage's input).

DECIMATION METHOD:
Slicer's "Preprocess input surface model" does NOT call vtkQuadricDecimation.
It calls a separate compiled CLI executable (slicer.modules.decimation, with
method="FastQuadric" and a "DecimationAggressiveness" parameter) bundled with
Slicer. "FastQuadric" is the name of Sven Forstmann's Fast-Quadric-Mesh-
Simplification algorithm, which `pyfqmr` wraps directly with matching
parameter names (target_count, aggressiveness) and the same threshold formula
(threshold = alpha * (iteration+K)^aggressiveness) - strong evidence it's the
same underlying implementation Slicer vendors. This script therefore uses
pyfqmr with aggressiveness passed straight through unchanged, and
target_count = round(2 * targetNumberOfPoints) (Euler's formula for a closed
triangulated 2-manifold: faces ~= 2*vertices).

Validated empirically against Slicer's own decimated export (TargetNumberOfPoints=5000,
DecimationAggressiveness=4.0): point count 80045 (pyfqmr) vs 79269 (Slicer, ~1% off),
nearest-neighbor median ~0.13mm both directions, p95 <0.72mm. (For comparison, the
previous vtkQuadricDecimation-based approach was off by ~17x in point count and
~0.5-1.8mm median distance - not usable for fidelity with Slicer.) DEFAULT_
DECIMATION_AGGRESSIVENESS below matches this validated value - main.py's own
pipeline default was previously 2.0 for no documented reason; both now agree.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.preprocess input.vtk output_preprocessed.vtk
    python -m vescan.stages.preprocess input.vtk output_preprocessed.vtk --target-points 5000
    python -m vescan.stages.preprocess input.vtk output_preprocessed.vtk --subdivide
    python -m vescan.stages.preprocess input.vtk output_preprocessed.vtk --no-decimate

Usage (from Python - main.py, a notebook, or the Slicer Python console):
    import sys
    sys.path.append("/path/to/vmtk_building")
    from vescan.stages import preprocess
    preprocess.run(
        "input.vtk",
        "output_preprocessed.vtk",
    )
"""

import argparse
import logging
import sys

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vescan.io import load_surface, save_surface, detect_coordinate_space, keep_largest_connected_component

logger = logging.getLogger(__name__)

DEFAULT_TARGET_NUMBER_OF_POINTS = 5000.0
DEFAULT_DECIMATION_AGGRESSIVENESS = 4.0


def _polydata_to_arrays(polyData):
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(polyData)
    triangulator.PassLinesOff()
    triangulator.PassVertsOff()
    triangulator.Update()
    tri = triangulator.GetOutput()

    verts = vtk_to_numpy(tri.GetPoints().GetData()).astype("float64")
    polys = vtk_to_numpy(tri.GetPolys().GetData())
    faces = polys.reshape(-1, 4)[:, 1:4].astype("int32")
    return verts, faces


def _arrays_to_polydata(verts, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=True))

    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(face[0]))
        cells.InsertCellPoint(int(face[1]))
        cells.InsertCellPoint(int(face[2]))

    result = vtk.vtkPolyData()
    result.SetPoints(points)
    result.SetPolys(cells)
    return result


def decimate(surfacePolyData, targetNumberOfPoints, decimationAggressiveness=DEFAULT_DECIMATION_AGGRESSIVENESS,
             verbose=True):
    """Replicates ExtractCenterlineLogic.preprocess()'s decimation step using
    pyfqmr (Fast-Quadric-Mesh-Simplification), the same algorithm Slicer's
    "FastQuadric" Decimation CLI method is believed to wrap - see module
    docstring for the validation evidence. aggressiveness is passed straight
    through; target_count (triangles) is derived from targetNumberOfPoints
    via Euler's formula (faces ~= 2*vertices for a closed triangulated mesh)."""
    try:
        import pyfqmr
    except ImportError:
        raise ImportError("pyfqmr is required for decimation (pip install pyfqmr). "
                           "Use decimateEnabled=False / --no-decimate to skip this step instead.")

    numberOfInputPoints = surfacePolyData.GetNumberOfPoints()
    if numberOfInputPoints == 0:
        raise ValueError("Input surface model is empty")

    if numberOfInputPoints <= targetNumberOfPoints:
        if verbose:
            logger.info("Input already has %d points <= target %.0f - skipping decimation.",
                        numberOfInputPoints, targetNumberOfPoints)
        return surfacePolyData

    targetFaceCount = max(int(round(2 * targetNumberOfPoints)), 4)
    if verbose:
        logger.info("Decimating %d points -> target_count=%d faces (~%.0f points target), aggressiveness=%s.",
                    numberOfInputPoints, targetFaceCount, targetNumberOfPoints, decimationAggressiveness)

    verts, faces = _polydata_to_arrays(surfacePolyData)

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(verts, faces)
    simplifier.simplify_mesh(target_count=targetFaceCount, aggressiveness=decimationAggressiveness,
                              preserve_border=True, verbose=False)
    outVerts, outFaces, _outNormals = simplifier.getMesh()

    result = _arrays_to_polydata(outVerts, outFaces)

    if verbose:
        logger.info("  Decimated to %d points.", result.GetNumberOfPoints())

    return result


def signed_volume(polyData):
    """Volume enclosed by a closed triangulated surface, SIGNED by the
    triangles' winding: positive when they face outward, negative when the
    surface is inside-out.

    Exact, linear in the number of triangles and parameter-free, which is why
    it is preferred here over sampling points and testing whether a step along
    their normal lands inside. On real data it separates the two cases with
    nothing in between: -197614mm3 for the one inverted surface against
    +24155 to +195004mm3 for the eight correct ones."""
    total = 0.0
    pointIds = vtk.vtkIdList()
    for cellId in range(polyData.GetNumberOfCells()):
        polyData.GetCellPoints(cellId, pointIds)
        if pointIds.GetNumberOfIds() != 3:
            continue
        a = polyData.GetPoint(pointIds.GetId(0))
        b = polyData.GetPoint(pointIds.GetId(1))
        c = polyData.GetPoint(pointIds.GetId(2))
        total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                  - a[1] * (b[0] * c[2] - b[2] * c[0])
                  + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return total


def ensure_outward_normals(polyData, verbose=True):
    """Flips polyData if its triangles face inward, so downstream stages can
    rely on the normals meaning "outward".

    vtkPolyDataNormals' AutoOrientNormals, used just above, is documented to
    assume a closed surface with NO non-manifold edges, and that "if these
    constraints do not hold, all bets are off". Segmented vessel surfaces
    routinely have a few dozen non-manifold edges, so the bet is one this
    pipeline was silently taking on every surface.

    It loses on real data. One patient's vein surface came out of preprocess
    entirely inside-out - 1964 of 2000 sampled normals pointing inward -
    while the same patient's artery and airway surfaces, and every other
    patient's, came out correct; the vein surface had 39 non-manifold edges
    and the artery that survived had 50, so the count predicts nothing.

    Nothing about that is visible in Slicer, since a closed surface renders
    the same either way. Fixing it here makes the file on disk mean what it
    says, for every consumer that reads its orientation.

    It does NOT, however, fix centerline extraction, and it is worth being
    explicit about that because it looks like it should.
    vtkvmtkPolyDataCenterlines ignores the normals it is handed and
    recomputes them with the very same AutoOrientNormals heuristic, then uses
    the result to tell inside from outside - so on the surface that defeated
    the heuristic here, it is defeated there too, identically, whatever
    orientation this function leaves behind. Verified by replaying that
    filter's own normals step on the corrected surface: it flips it straight
    back (+197614mm3 in, -197614mm3 out) and reproduces the same broken
    Voronoi to the last digit. That failure is handled where it happens, by
    vescan.stages.centerline.needs_flipped_normals()."""
    volume = signed_volume(polyData)
    if volume >= 0.0:
        return polyData

    if verbose:
        logger.warning("Surface is inside-out (signed volume %.1fmm3) - reversing it. AutoOrientNormals "
                        "cannot be trusted on a mesh with non-manifold edges, and centerline extraction "
                        "reads these normals to tell inside from outside.", volume)
    reverse = vtk.vtkReverseSense()
    reverse.SetInputData(polyData)
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


def preprocess(surfacePolyData, targetNumberOfPoints=DEFAULT_TARGET_NUMBER_OF_POINTS,
                decimationAggressiveness=DEFAULT_DECIMATION_AGGRESSIVENESS,
                subdivide=False, decimateEnabled=True, keepLargestComponent=True, verbose=True):
    """Mirrors ExtractCenterlineLogic.preprocess(): optionally drop every
    connected component but the largest first (same filtering
    autofind_endpoints.py applies, hoisted here so every downstream stage -
    endpoints, network, centerline - sees a single-component surface), then
    decimate (optional, via pyfqmr), clean, triangulate, optional single
    linear subdivision, recompute consistent normals - same order of
    operations. See decimate() docstring for the decimation method and its
    validation against Slicer's own output."""
    if keepLargestComponent:
        surfacePolyData = keep_largest_connected_component(surfacePolyData, verbose=verbose)

    if decimateEnabled:
        surfacePolyData = decimate(surfacePolyData, targetNumberOfPoints, decimationAggressiveness, verbose=verbose)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfacePolyData)
    cleaner.Update()

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(cleaner.GetOutput())
    triangulator.PassLinesOff()
    triangulator.PassVertsOff()
    triangulator.Update()
    triangulatedOutput = triangulator.GetOutput()

    if subdivide:
        subdiv = vtk.vtkLinearSubdivisionFilter()
        subdiv.SetInputData(triangulatedOutput)
        subdiv.SetNumberOfSubdivisions(1)
        subdiv.Update()
        if subdiv.GetOutput().GetNumberOfPoints() == 0:
            if verbose:
                logger.warning("Mesh subdivision failed - skipping subdivision step.")
            subdivide = False

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(subdiv.GetOutput() if subdivide else triangulatedOutput)
    normals.SetAutoOrientNormals(1)
    normals.SetFlipNormals(0)
    normals.SetConsistency(1)
    normals.SplittingOff()
    normals.Update()

    return ensure_outward_normals(normals.GetOutput(), verbose=verbose)


def run(input_path, output_path, target_points=DEFAULT_TARGET_NUMBER_OF_POINTS,
        decimation_aggressiveness=DEFAULT_DECIMATION_AGGRESSIVENESS,
        subdivide=False, decimate_enabled=True, keep_largest_component=True):
    """Reads input_path, preprocesses it, writes output_path. This is the
    stage's single entry point - called both by main.py (in-process) and by
    this module's own CLI (main() below, via the preprocess_surface.py
    shim)."""
    # Preprocessing is a purely geometric step (no coordinate-system-dependent
    # heuristics involved, unlike network/endpoint extraction) - pass the
    # surface through in whatever space it's already in, and preserve that
    # space in the output header so downstream scripts detect it correctly.
    coordinateSpace = detect_coordinate_space(input_path) or "LPS"
    surfacePolyData = load_surface(input_path)

    preprocessedPolyData = preprocess(
        surfacePolyData,
        targetNumberOfPoints=target_points,
        decimationAggressiveness=decimation_aggressiveness,
        subdivide=subdivide,
        decimateEnabled=decimate_enabled,
        keepLargestComponent=keep_largest_component,
    )

    save_surface(preprocessedPolyData, output_path, coordinate_space=coordinateSpace)
    logger.info("Saved preprocessed surface (%d points) to %s", preprocessedPolyData.GetNumberOfPoints(), output_path)
    return preprocessedPolyData


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_surface", help="Path to input surface (.vtk or .vtp)")
    parser.add_argument("output_surface", help="Path to output preprocessed surface (.vtk or .vtp)")
    parser.add_argument("--target-points", type=float, default=DEFAULT_TARGET_NUMBER_OF_POINTS,
                         help=f"Target number of points after decimation (Slicer's TargetNumberOfPoints, "
                              f"default {DEFAULT_TARGET_NUMBER_OF_POINTS:.0f}, same default as "
                              f"ExtractCenterline's setDefaultParameters).")
    parser.add_argument("--decimation-aggressiveness", type=float, default=DEFAULT_DECIMATION_AGGRESSIVENESS,
                         help=f"Passed straight through to pyfqmr as its 'aggressiveness' parameter, matching "
                              f"Slicer's DecimationAggressiveness (default {DEFAULT_DECIMATION_AGGRESSIVENESS}) "
                              f"- see module docstring for validation against Slicer's own output.")
    parser.add_argument("--subdivide", action="store_true",
                         help="Apply one linear subdivision pass, matching Slicer's SubdivideInputSurface checkbox "
                              "(default off, same default as Slicer).")
    parser.add_argument("--no-decimate", action="store_true",
                         help="Skip decimation entirely (clean/triangulate/normals only), matching Slicer's "
                              "'Preprocess input surface model' checkbox being unchecked.")
    parser.add_argument("--keep-all-components", action="store_true",
                         help="Do not filter out disconnected surface fragments before the rest of preprocessing. "
                              "By default only the largest connected component is kept, since stray debris/islands "
                              "from segmentation can otherwise attract the auto start point and produce endpoints "
                              "on the wrong fragment (same filtering autofind_endpoints.py applies on its own "
                              "input - on by default here too, but now once, up front).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    run(
        args.input_surface,
        args.output_surface,
        target_points=args.target_points,
        decimation_aggressiveness=args.decimation_aggressiveness,
        subdivide=args.subdivide,
        decimate_enabled=not args.no_decimate,
        keep_largest_component=not args.keep_all_components,
    )


if __name__ == "__main__":
    sys.exit(main())