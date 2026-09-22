#!/usr/bin/env python3
"""
Builds ONE combined .vtk/.vtp containing all 5 pulmonary lobe surfaces
merged into a single mesh, tagged with THREE independent per-cell arrays -
LobeSegment_Vein, LobeSegment_Artery, LobeSegment_Airway - one dedicated
value per lobe per vessel type. Load once in 3D Slicer and switch Active
Scalar between the three to recolor the same 5 lobes by their venous/
arterial/bronchial drainage territory, instead of juggling three separate
lobe files.

Deliberately a NEW, separate numbering from vescan.stages.
anatomical_segments' AnatomicalSegment scale (not a reuse of its lobar
values) - this file needs
its own legend rather than inheriting the centerline one. Starts at 30 so
it never collides with AnatomicalSegment's own values (1-28 across its
airway/artery/vein blocks) even if both are ever loaded/compared together.

Value scheme, fixed lobe order everywhere else in this pipeline (RUL/RML/
RLL/LLL/LUL) - each vessel type gets its own block of 5, one per lobe:
  LobeSegment_Vein:   30 RUL, 31 RML, 32 RLL, 33 LLL, 34 LUL
  LobeSegment_Artery: 40 RUL, 41 RML, 42 RLL, 43 LLL, 44 LUL
  LobeSegment_Airway: 50 RUL, 51 RML, 52 RLL, 53 LLL, 54 LUL
The three blocks never overlap (each other, or AnatomicalSegment's).

INTERLOBAR FISSURES
-------------------
Also extracts the pulmonary fissures, which are already present in these
same lobe surfaces and need no extra input: adjacent lobes come from ONE
TotalSegmentator label map, so marching cubes emits COINCIDENT triangles on
their shared boundary - the fissure is literally the set of cells two lobe
surfaces have in common. Measured on real data, the area found from the A
side and from the B side agree to within 0.1% (e.g. 7003.3 vs 7002.8 mm2),
which is what "coincident" means here, and the selection is insensitive to
FISSURE_CONTACT_TOLERANCE (12.3% of RUL's points at 0.5 mm, 13.4% at 2.0 mm
- a plateau, not a slope), so that tolerance is a guard, not a tuning knob.

There are FOUR contact surfaces, forming THREE anatomical fissures: the
right oblique fissure separates RLL from BOTH RUL and RML, so it shows up
as two contacts. Every other lobe pair measures 0.0 mm2 across all test
patients (the left and right lungs do not touch), so those four are the
complete set. Two arrays, since both views are useful and neither is
derivable from the other at cell level:
  FissureContact: 60 RUL|RML, 61 RUL|RLL, 62 RML|RLL, 63 LUL|LLL
  Fissure:        70 orizzontale dx, 71 obliqua dx, 72 obliqua sx
0 in both means "not a fissure cell". Fissure merges 61+62 into 71.

The fissure is a gently CURVED sheet, not a plane: least-squares plane fits
leave an RMS of 1.4-5.8 mm and peak deviations of 5-16 mm. The legend JSON
therefore reports the fitted plane (centroid/normal/RMS) as a descriptor
only - use the mesh, not the plane, to decide which side of a fissure
something is on, or everything within ~1 cm of it (exactly the crossing
vessels) gets misclassified.

Caveat worth stating once: this is the SEGMENTATION's fissure, not the CT's.
Where a fissure is anatomically incomplete, TotalSegmentator still closes
the lobe boundary with an interpolated surface, so the sheet extracted here
is always complete even when the patient's own fissure is not.

Two outputs: the merged lobe mesh carries the fissure arrays alongside the
LobeSegment_* ones (fissure cells appear twice there, once per lobe, since
both copies are coincident), and a separate lobe_fissures.vtk holds the
fissure sheets ALONE, deduplicated to one copy each - that second file is
the one to use as a geometric object.

To hide ONE lobe in Slicer: this is a single merged mesh (one model node),
so the ordinary eye-icon visibility toggle hides/shows the WHOLE thing, not
one lobe within it. Use Models > Display > Scalars > enable "Threshold" on
whichever LobeSegment_*/Fissure* array is active, and set the range to
exclude (or, to isolate just it, min=max=) that lobe's/fissure's own value.

PATIENT-level, not per-vessel-type: the 5 lobe surfaces are the same
regardless of which structure (artery/vein/airway) is being processed, so
vescan.orchestrator.run_pipeline() calls this once per patient - it's
IDEMPOTENT (skips if the output already exists) the same way stage 0
(segmentation conversion) is, since it still gets invoked once per structure
in that per-patient loop.

Usage (CLI, standalone - for debugging in isolation):
    python -m vescan.stages.build_lobe_segments Outputs/patient_1/lobe_segments.vtk \\
        --lobe RUL=Data/patient_1/lung_upper_lobe_right.vtk \\
        --lobe RML=Data/patient_1/lung_middle_lobe_right.vtk \\
        --lobe RLL=Data/patient_1/lung_lower_lobe_right.vtk \\
        --lobe LLL=Data/patient_1/lung_lower_lobe_left.vtk \\
        --lobe LUL=Data/patient_1/lung_upper_lobe_left.vtk \\
        --legend-output Outputs/patient_1/lobe_segments.json \\
        --fissures-output Outputs/patient_1/lobe_fissures.vtk
"""

import argparse
import json
import logging
import sys

import vtk
from vtk.util import numpy_support as ns
import numpy as np

from vescan.io import save_surface
from vescan.lobes import LOBE_ORDER
from vescan.stages.lobe_reachability import parse_lobe_args, load_lobe_surfaces

logger = logging.getLogger(__name__)

ARRAY_NAME = {"vein": "LobeSegment_Vein", "artery": "LobeSegment_Artery", "airway": "LobeSegment_Airway"}
BLOCK_START = {"vein": 30, "artery": 40, "airway": 50}
VESSEL_TYPE_LABEL = {"vein": "Vena", "artery": "Arteria", "airway": "Bronco"}
LOBE_LABEL = {
    "RUL": "lobo superiore destro", "RML": "lobo medio destro", "RLL": "lobo inferiore destro",
    "LLL": "lobo inferiore sinistro", "LUL": "lobo superiore sinistro",
}

FISSURE_CONTACT_ARRAY = "FissureContact"
FISSURE_ARRAY = "Fissure"

# The 4 lobe pairs that actually touch - see the module docstring. Every
# other pair of the 10 possible ones measures 0.0 mm2 (the two lungs don't
# touch), so this list is exhaustive, not a shortlist.
# (lobeA, lobeB, contactValue, contactName, fissureValue)
FISSURE_CONTACTS = [
    ("RUL", "RML", 60, "Scissura orizzontale destra", 70),
    ("RUL", "RLL", 61, "Scissura obliqua destra (porzione superiore)", 71),
    ("RML", "RLL", 62, "Scissura obliqua destra (porzione inferiore)", 71),
    ("LUL", "LLL", 63, "Scissura obliqua sinistra", 72),
]

FISSURE_LABEL = {
    70: "Scissura orizzontale destra",
    71: "Scissura obliqua destra",
    72: "Scissura obliqua sinistra",
}

# A cell counts as fissure when ALL its vertices are within this distance of
# the neighbouring lobe's surface. NOT a tuning knob: the two surfaces are
# coincident there, so the selection sits on a plateau (12.3% of points at
# 0.5 mm vs 13.4% at 2.0 mm on R01-091's RUL) - 1 mm is a guard against
# sub-voxel jitter, and any value in roughly [0.5, 2] gives the same sheet.
FISSURE_CONTACT_TOLERANCE = 1.0

# Connected patches smaller than this are dropped: real fissures come out as
# ONE component of several thousand mm2, and the only extra components seen
# on real data were degenerate slivers of ~0 mm2 at the sheet's rim.
MIN_FISSURE_COMPONENT_AREA = 50.0

# Below this total area a contact is reported as absent rather than as a
# fissure - guards against a degenerate segmentation where two lobes merely
# graze each other (the largest spurious contact measured, RUL|LUL across
# the mediastinum on LUNGx-CT011, was 37.3 mm2 against 5900+ for real ones).
MIN_FISSURE_AREA = 100.0


def label_value(vesselType, lobeName):
    return BLOCK_START[vesselType] + LOBE_ORDER.index(lobeName)


def label_name(vesselType, lobeName):
    return f"{VESSEL_TYPE_LABEL[vesselType]} - {LOBE_LABEL[lobeName]}"


def _triangles(polyData):
    """(N,3) int array of the triangle connectivity. The lobe surfaces come
    out of marching cubes so they're already all triangles; vtkTriangleFilter
    upstream would only cost a copy, so this asserts instead."""
    conn = ns.vtk_to_numpy(polyData.GetPolys().GetConnectivityArray())
    numCells = polyData.GetNumberOfCells()
    if conn.size != 3 * numCells:
        raise ValueError(f"Expected an all-triangle surface, got {conn.size} connectivity "
                         f"entries for {numCells} cells")
    return conn.reshape(numCells, 3)


def _cell_areas(polyData):
    """Per-cell triangle area, vectorised - vtkMassProperties only gives the
    total, and we need areas grouped by connected region."""
    pts = ns.vtk_to_numpy(polyData.GetPoints().GetData())
    tri = _triangles(polyData)
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def contact_cell_mask(surfaceA, surfaceB, tolerance=FISSURE_CONTACT_TOLERANCE):
    """Boolean mask over surfaceA's CELLS: True where all 3 vertices lie
    within `tolerance` of surfaceB. Point-to-surface distance (not
    point-to-point) so it still behaves if a future dataset meshes the lobes
    separately instead of from one shared label map."""
    distanceFilter = vtk.vtkDistancePolyDataFilter()
    distanceFilter.SetInputData(0, surfaceA)
    distanceFilter.SetInputData(1, surfaceB)
    distanceFilter.SignedDistanceOff()
    distanceFilter.ComputeSecondDistanceOff()
    distanceFilter.Update()

    distances = ns.vtk_to_numpy(distanceFilter.GetOutput().GetPointData().GetArray("Distance"))
    nearB = distances < tolerance
    return nearB[_triangles(surfaceA)].all(axis=1)


def _extract_cells(polyData, cellMask):
    """The masked cells of polyData as a standalone polydata. Rebuilt by
    hand rather than via vtkExtractSelection so the result is plain polydata
    (no unstructured-grid round trip) and the point renumbering is ours."""
    tri = _triangles(polyData)[cellMask]
    usedPointIds = np.unique(tri)

    remap = np.full(polyData.GetNumberOfPoints(), -1, dtype=np.int64)
    remap[usedPointIds] = np.arange(usedPointIds.size)

    allPoints = ns.vtk_to_numpy(polyData.GetPoints().GetData())
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(allPoints[usedPointIds]), deep=True))

    patch = vtk.vtkPolyData()
    patch.SetPoints(points)
    patch.Allocate(tri.shape[0])
    ids = vtk.vtkIdList()
    ids.SetNumberOfIds(3)
    for triangle in remap[tri]:
        ids.SetId(0, int(triangle[0]))
        ids.SetId(1, int(triangle[1]))
        ids.SetId(2, int(triangle[2]))
        patch.InsertNextCell(vtk.VTK_TRIANGLE, ids)
    return patch


def _drop_small_components(patch, minArea=MIN_FISSURE_COMPONENT_AREA):
    """Removes connected components below minArea. Returns
    (cleanedPatch, numberOfComponentsKept, numberOfComponentsDropped)."""
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(patch)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    coloured = connectivity.GetOutput()

    # ColorRegionsOn only writes RegionId on the POINTS (checked on VTK
    # 9.5.2), so each cell takes its region from its first vertex - regions
    # are disjoint by construction, so any vertex of the cell gives the same
    # answer.
    pointRegionIds = ns.vtk_to_numpy(coloured.GetPointData().GetArray("RegionId"))
    regionIds = pointRegionIds[_triangles(coloured)[:, 0]]

    areaPerRegion = np.bincount(regionIds, weights=_cell_areas(coloured),
                                minlength=connectivity.GetNumberOfExtractedRegions())
    keptRegions = np.flatnonzero(areaPerRegion >= minArea)

    # _extract_cells() rebuilds bare geometry, so RegionId doesn't survive
    # into the result - which is what we want, it was scaffolding.
    return _extract_cells(coloured, np.isin(regionIds, keptRegions)), \
        keptRegions.size, int(areaPerRegion.size - keptRegions.size)


def fit_plane(polyData):
    """Least-squares plane through the points (smallest-singular-value
    direction of the centred coordinates). Returns
    (centroid, unitNormal, rmsDeviation, maxDeviation), all in mm.

    A DESCRIPTOR, not a model of the fissure - see the module docstring: the
    real sheet is curved and leaves 1.4-5.8 mm of RMS behind."""
    pts = ns.vtk_to_numpy(polyData.GetPoints().GetData())
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2]
    deviation = (pts - centroid) @ normal
    return centroid, normal, float(np.sqrt((deviation ** 2).mean())), float(np.abs(deviation).max())


def extract_fissures(lobeSurfaces, tolerance=FISSURE_CONTACT_TOLERANCE,
                      min_component_area=MIN_FISSURE_COMPONENT_AREA, min_fissure_area=MIN_FISSURE_AREA):
    """Finds every interlobar fissure in a {lobeName: polydata} dict.

    Returns (fissures, cellLabels):
      fissures   - one dict per FISSURE_CONTACTS entry that cleared
                   min_fissure_area, carrying the extracted sheet plus its
                   measured geometry.
      cellLabels - {lobeName: {FISSURE_CONTACT_ARRAY: int32 per-cell array,
                   FISSURE_ARRAY: ...}}, 0 where the cell isn't a fissure,
                   ready to hang on the merged lobe mesh.

    Labels BOTH sides of each contact (each lobe's own copy of the shared
    triangles), because the merged mesh contains both. Where two fissures
    meet - a cell of RUL touching RML and RLL at once - the first contact in
    FISSURE_CONTACTS order wins; the overlap is a handful of cells at the
    rim and never large enough to matter for either count."""
    cellLabels = {
        name: {
            FISSURE_CONTACT_ARRAY: np.zeros(surface.GetNumberOfCells(), dtype=np.int32),
            FISSURE_ARRAY: np.zeros(surface.GetNumberOfCells(), dtype=np.int32),
        }
        for name, surface in lobeSurfaces.items()
    }

    fissures = []
    for lobeA, lobeB, contactValue, contactName, fissureValue in FISSURE_CONTACTS:
        if lobeA not in lobeSurfaces or lobeB not in lobeSurfaces:
            logger.warning("Fissure %s: lobe surface missing (%s or %s) - skipped.",
                           contactName, lobeA, lobeB)
            continue

        maskA = contact_cell_mask(lobeSurfaces[lobeA], lobeSurfaces[lobeB], tolerance)
        maskB = contact_cell_mask(lobeSurfaces[lobeB], lobeSurfaces[lobeA], tolerance)
        if not maskA.any() or not maskB.any():
            logger.info("Fissure %s (%s|%s): no contact found - skipped.", contactName, lobeA, lobeB)
            continue

        # The sheet is taken from the A side ONLY: B's copy is coincident, so
        # keeping both would double every triangle in lobe_fissures.vtk.
        patch, numKept, numDropped = _drop_small_components(_extract_cells(lobeSurfaces[lobeA], maskA),
                                                             minArea=min_component_area)
        area = float(_cell_areas(patch).sum()) if patch.GetNumberOfCells() else 0.0
        if area < min_fissure_area:
            logger.info("Fissure %s (%s|%s): contact area %.1f mm2 below the %.1f mm2 floor - "
                        "reported as absent.", contactName, lobeA, lobeB, area, min_fissure_area)
            continue
        if numDropped:
            logger.info("Fissure %s: dropped %d connected component(s) under %.0f mm2.",
                        contactName, numDropped, min_component_area)

        for lobeName, mask in ((lobeA, maskA), (lobeB, maskB)):
            unlabelled = mask & (cellLabels[lobeName][FISSURE_CONTACT_ARRAY] == 0)
            cellLabels[lobeName][FISSURE_CONTACT_ARRAY][unlabelled] = contactValue
            cellLabels[lobeName][FISSURE_ARRAY][unlabelled] = fissureValue

        centroid, normal, rms, maxDeviation = fit_plane(patch)
        for arrayName, value in ((FISSURE_CONTACT_ARRAY, contactValue), (FISSURE_ARRAY, fissureValue)):
            values = np.full(patch.GetNumberOfCells(), value, dtype=np.int32)
            arr = ns.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_INT)
            arr.SetName(arrayName)
            patch.GetCellData().AddArray(arr)

        fissures.append({
            "contact_value": contactValue, "contact_name": contactName,
            "fissure_value": fissureValue, "fissure_name": FISSURE_LABEL[fissureValue],
            "lobe_a": lobeA, "lobe_b": lobeB, "polydata": patch,
            "area_mm2": round(area, 1),
            "n_components": int(numKept),
            "centroid": [round(float(c), 3) for c in centroid],
            "plane_normal": [round(float(n), 4) for n in normal],
            "plane_rms_mm": round(rms, 3),
            "plane_max_deviation_mm": round(maxDeviation, 3),
            "n_points": patch.GetNumberOfPoints(),
            "n_cells": patch.GetNumberOfCells(),
        })
        logger.info("Fissure %-45s %s|%s  area %8.1f mm2  piano RMS %.2f mm",
                    contactName, lobeA, lobeB, area, rms)

    return fissures, cellLabels


def build_fissure_surface(fissures):
    """The fissure sheets merged into one polydata (one copy per fissure,
    the A side), carrying both fissure arrays. None if there are no
    fissures at all."""
    if not fissures:
        return None
    append = vtk.vtkAppendPolyData()
    for fissure in fissures:
        append.AddInputData(fissure["polydata"])
    append.Update()
    return append.GetOutput()


def build_combined_lobes(lobes, target_space="LPS", extract_fissures_too=True,
                         fissure_contact_tolerance=FISSURE_CONTACT_TOLERANCE,
                         min_fissure_component_area=MIN_FISSURE_COMPONENT_AREA,
                         min_fissure_area=MIN_FISSURE_AREA):
    """lobes: {lobeName: path}, any non-empty subset of LOBE_ORDER - lobes
    left out are simply absent from the combined overview (and from any
    fissure that would have needed them, via extract_fissures()'s own
    per-contact "missing" skip). Returns (combined, fissures): the merged
    vtkPolyData with the three LobeSegment_* cell arrays plus, unless
    extract_fissures_too is off, the two fissure ones; and the fissure
    descriptors from extract_fissures() ([] when skipped)."""
    unknown = [name for name in lobes if name not in LOBE_ORDER]
    if unknown:
        raise ValueError(f"Unknown lobe name(s) {unknown} - expected a subset of {LOBE_ORDER}")
    if not lobes:
        raise ValueError(f"No lobes given - need at least one of {LOBE_ORDER}")
    excluded = [name for name in LOBE_ORDER if name not in lobes]
    if excluded:
        logger.info("Building combined lobe overview without %s (excluded via config).", excluded)

    lobeSurfaces = load_lobe_surfaces(lobes, target_space)

    fissures, cellLabels = ([], {}) if not extract_fissures_too else \
        extract_fissures(lobeSurfaces, tolerance=fissure_contact_tolerance,
                          min_component_area=min_fissure_component_area, min_fissure_area=min_fissure_area)

    append = vtk.vtkAppendPolyData()
    for lobeName in LOBE_ORDER:
        if lobeName not in lobeSurfaces:
            continue
        polyData = lobeSurfaces[lobeName]
        numCells = polyData.GetNumberOfCells()
        arrays = {ARRAY_NAME[vesselType]: np.full(numCells, label_value(vesselType, lobeName),
                                                  dtype=np.int32)
                  for vesselType in ARRAY_NAME}
        arrays.update(cellLabels.get(lobeName, {}))
        for arrayName, values in arrays.items():
            arr = ns.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_INT)
            arr.SetName(arrayName)
            polyData.GetCellData().AddArray(arr)
        append.AddInputData(polyData)
    append.Update()
    return append.GetOutput(), fissures


def build_legend_data(fissures=None):
    """fissures: the list from extract_fissures(), or None/[] to leave the
    fissure section out entirely (so a run with fissure extraction off
    doesn't advertise arrays the file doesn't carry)."""
    legend = {
        "lobe_order": LOBE_ORDER,
        "arrays": {
            ARRAY_NAME[vesselType]: {
                "vessel_type": vesselType,
                "labels": {str(label_value(vesselType, lobeName)): label_name(vesselType, lobeName)
                           for lobeName in LOBE_ORDER},
            }
            for vesselType in ARRAY_NAME
        },
    }
    if not fissures:
        return legend

    legend["arrays"][FISSURE_CONTACT_ARRAY] = {
        "description": "Superficie di contatto fra due lobi (4 contatti)",
        "labels": {str(f["contact_value"]): f"{f['contact_name']} ({f['lobe_a']}|{f['lobe_b']})"
                   for f in fissures},
    }
    legend["arrays"][FISSURE_ARRAY] = {
        "description": "Scissura anatomica (3 scissure; l'obliqua destra riunisce i contatti 61 e 62)",
        "labels": {str(value): name for value, name in FISSURE_LABEL.items()
                   if any(f["fissure_value"] == value for f in fissures)},
    }
    # The fitted plane is a DESCRIPTOR - see fit_plane()/the module docstring
    # - hence reported next to the RMS that says how far from planar the real
    # sheet is, never on its own.
    legend["fissures"] = [{k: v for k, v in f.items() if k != "polydata"} for f in fissures]
    return legend


def save_legend_json(output_path, fissures=None):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(build_legend_data(fissures), f, indent=2)


def run(output_path, lobes, coordinate_space="LPS", legend_output_path=None,
        fissures_output_path=None, extract_fissures_too=True,
        fissure_contact_tolerance=FISSURE_CONTACT_TOLERANCE,
        min_fissure_component_area=MIN_FISSURE_COMPONENT_AREA,
        min_fissure_area=MIN_FISSURE_AREA):
    """lobes: {lobeName: path} - see default_lobe_surfaces() in
    vescan/paths.py for the usual source of this dict (optionally
    filtered down to a subset - see BuildLobeSegmentsConfig.enabled_lobes),
    and LobeOverviewPaths there for output_path/legend_output_path/
    fissures_output_path. Returns (combined polydata, fissure descriptors)."""
    combined, fissures = build_combined_lobes(lobes, target_space=coordinate_space,
                                              extract_fissures_too=extract_fissures_too,
                                              fissure_contact_tolerance=fissure_contact_tolerance,
                                              min_fissure_component_area=min_fissure_component_area,
                                              min_fissure_area=min_fissure_area)

    arrayNames = list(ARRAY_NAME.values())
    if fissures:
        arrayNames += [FISSURE_CONTACT_ARRAY, FISSURE_ARRAY]
    save_surface(combined, output_path, coordinate_space=coordinate_space)
    logger.info("Saved combined lobes (%d cells, arrays %s) to %s",
                combined.GetNumberOfCells(), arrayNames, output_path)

    if fissures_output_path:
        fissureSurface = build_fissure_surface(fissures)
        if fissureSurface is None:
            logger.warning("No fissures extracted - %s not written.", fissures_output_path)
        else:
            save_surface(fissureSurface, fissures_output_path, coordinate_space=coordinate_space)
            logger.info("Saved %d fissure sheet(s) (%d cells, %.1f mm2 total) to %s",
                        len(fissures), fissureSurface.GetNumberOfCells(),
                        sum(f["area_mm2"] for f in fissures), fissures_output_path)

    if legend_output_path:
        save_legend_json(legend_output_path, fissures)
        logger.info("Saved legend JSON to %s", legend_output_path)

    return combined, fissures


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="Path to save the combined lobes surface (.vtk/.vtp)")
    parser.add_argument("--lobe", action="append", required=True, default=[], metavar="NAME=PATH",
                         help="One lobe surface (.vtk/.vtp), repeatable - needs exactly one of "
                              f"{LOBE_ORDER} (e.g. --lobe RUL=lung_upper_lobe_right.vtk)")
    parser.add_argument("--coordinate-space", default="LPS", choices=["LPS", "RAS"],
                         help="Target coordinate space for the combined output (default: LPS) - any lobe "
                              "surface tagged differently is converted to match")
    parser.add_argument("--legend-output", default=None,
                         help="Optional path to save a JSON legend: per-array label values/names, "
                              "plus each fissure's measured area/centroid/fitted plane")
    parser.add_argument("--fissures-output", default=None,
                         help="Optional path (.vtk/.vtp) to save the interlobar fissure sheets on "
                              "their own, one copy each - the merged lobe surface gets the "
                              f"{FISSURE_CONTACT_ARRAY}/{FISSURE_ARRAY} arrays either way")
    parser.add_argument("--no-fissures", action="store_true",
                         help="Skip fissure extraction entirely (the merged surface then carries "
                              "only the three LobeSegment_* arrays)")
    parser.add_argument("--fissure-contact-tolerance", type=float, default=FISSURE_CONTACT_TOLERANCE,
                         help="A cell is fissure when all 3 of its vertices are within this many mm "
                              f"of the neighbouring lobe (default: {FISSURE_CONTACT_TOLERANCE}) - a "
                              "guard against sub-voxel jitter, not a tuning knob: adjacent lobes "
                              "come from one label map so their surfaces are coincident there")
    parser.add_argument("--min-fissure-component-area", type=float, default=MIN_FISSURE_COMPONENT_AREA,
                         help="Connected patches of a candidate fissure smaller than this many mm2 are "
                              f"dropped as artifacts (default: {MIN_FISSURE_COMPONENT_AREA})")
    parser.add_argument("--min-fissure-area", type=float, default=MIN_FISSURE_AREA,
                         help="Below this total contact area (mm2), a fissure is reported as absent "
                              f"rather than extracted (default: {MIN_FISSURE_AREA})")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()
    lobes = parse_lobe_args(args.lobe)

    run(args.output, lobes, coordinate_space=args.coordinate_space,
        legend_output_path=args.legend_output, fissures_output_path=args.fissures_output,
        extract_fissures_too=not args.no_fissures,
        fissure_contact_tolerance=args.fissure_contact_tolerance,
        min_fissure_component_area=args.min_fissure_component_area,
        min_fissure_area=args.min_fissure_area)


if __name__ == "__main__":
    sys.exit(main())
