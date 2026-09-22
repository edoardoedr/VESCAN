#!/usr/bin/env python3
"""
Standalone binary-labelmap -> closed-surface conversion, replicating 3D
Slicer's vtkBinaryLabelmapToClosedSurfaceConversionRule (SegmentationCore)
without requiring 3D Slicer.

Reads a multi-label segmentation volume (.nrrd/.nii/.nii.gz/.mha/... -
anything SimpleITK can read) and, for each label value, reproduces Slicer's
default "Closed surface" representation:

  1. crop to the label's bounding box + 1 voxel margin (this is what makes
     it fast on big multi-organ volumes - Slicer itself operates on each
     segment's own, similarly-cropped binary labelmap, not the full volume);
  2. pad by 1 voxel on all sides if the label touches the crop border (avoids
     open surfaces at the volume edge - IsLabelmapPaddingNecessary in the
     source rule);
  3. iso-contour at the label value with vtkDiscreteFlyingEdges3D (default
     method) or vtkSurfaceNets3D, in identity IJK space;
  4. optional decimation (vtkDecimatePro) and smoothing
     (vtkWindowedSincPolyDataFilter), same formulas as the source rule;
  5. transform IJK -> world with the volume's IJK-to-RAS matrix (built from
     SimpleITK's origin/spacing/direction, which are in LPS - flipped to RAS
     the same way Slicer's vtkOrientedImageData::GetImageToWorldMatrix()
     represents its world space);
  6. recompute consistent normals (flying edges method only, matching the
     source rule).

NOT implemented: Slicer's "joint smoothing" segmentation-conversion parameter
(off by default) and shared-labelmap "convert once, extract per segment"
optimization - both are advanced, opt-in features; every label here is
iso-contoured independently, which is what Slicer itself does whenever joint
smoothing is off (its default).

Coordinate space: internally the surface is built in RAS (Slicer's live-scene
convention), then converted to LPS before writing (via flip_lps_ras from
vescan.io), matching the "3D Slicer output. SPACE=LPS" convention
already used by every other .vtk file in this pipeline (see Data/patient_1/).
Pass --coordinate-space RAS to skip that flip and write RAS coordinates
instead.

Two entry points:
  - convert_labelmap_file(...)  converts ONE segmentation volume into one
    surface file per label - the original, still the CLI's own behavior
    (see main() below / the labelmap_to_closed_surface.py shim at the repo
    root).
  - run(...)  converts every segmentation volume directly inside a FOLDER
    (e.g. one patient's TotalSegmentator output) into
    <folder>_vtk/<stem>.vtk (or <stem>_<label_name>.vtk for a multi-label
    file) - this is the pipeline's own stage 0 (pipeline.stages.convert /
    pipeline.convert in the JSON config), a plain importable stage like
    every other one, so a run can start from raw segmentations instead of
    an already-exported surface.

Usage (CLI, single file - for debugging in isolation):
    python -m vescan.stages.convert_segmentations segmentation.nrrd output_dir/
    python -m vescan.stages.convert_segmentations segmentation.nrrd output_dir/ --labels 1 3 5
    python -m vescan.stages.convert_segmentations segmentation.nrrd output_dir/ --names names.json
    python -m vescan.stages.convert_segmentations segmentation.nrrd output_dir/ --method surface-nets

Usage (from Python, whole folder):
    from vescan.stages import convert_segmentations
    convert_segmentations.run("Data/patient_1_raw")  # -> Data/patient_1_raw_vtk/
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from vescan.io import Stage, save_surface, flip_lps_ras

logger = logging.getLogger(__name__)

CONVERSION_METHOD_FLYING_EDGES = "flying-edges"
CONVERSION_METHOD_SURFACE_NETS = "surface-nets"

# Same defaults as vtkBinaryLabelmapToClosedSurfaceConversionRule's constructor.
DEFAULT_DECIMATION_FACTOR = 0.0
DEFAULT_SMOOTHING_FACTOR = 0.5
DEFAULT_COMPUTE_SURFACE_NORMALS = True
DEFAULT_CONVERSION_METHOD = CONVERSION_METHOD_FLYING_EDGES
DEFAULT_SURFACE_NETS_INTERNAL_SMOOTHING = False

# Recognized volume extensions for run()'s folder scan - anything else in the
# folder (e.g. .DS_Store) is skipped, same set run_pipeline_batch_data.sh's
# is_volume_file() used.
VOLUME_EXTENSIONS = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd.gz")


def _read_labelmap(input_path):
    """Reads any SimpleITK-supported volume and returns the label array
    (numpy, shape (nz, ny, nx), matching ITK's k,j,i index order) plus the
    IJK-to-RAS 4x4 matrix built from spacing/direction/origin. SimpleITK/ITK
    always represent origin/direction in LPS regardless of the source file's
    own convention (e.g. NIfTI's RAS qform/sform is converted on read), so
    RAS = diag(-1,-1,1) applied to the LPS affine - same relationship Slicer
    uses between its files and its live RAS scene."""
    image = sitk.ReadImage(input_path)
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{input_path}: expected a single-component label image, "
                          f"got {image.GetNumberOfComponentsPerPixel()} components")

    array = sitk.GetArrayFromImage(image).astype(np.int32)  # (nz, ny, nx)

    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    ijkToLPS = np.eye(4)
    ijkToLPS[:3, :3] = direction @ np.diag(spacing)
    ijkToLPS[:3, 3] = origin

    lpsToRAS = np.diag([-1.0, -1.0, 1.0, 1.0])
    ijkToRAS = lpsToRAS @ ijkToLPS

    return array, ijkToRAS


def _crop_label(array, label, margin=1):
    """Crops to the label's bounding box + margin voxels (clamped to the
    volume), returning the cropped sub-array (still multi-label - the iso-
    contour filter needs the real neighboring values, not a binarized mask)
    and the low corner (i0, j0, k0) of the crop in full-volume IJK. Padding
    on any side that got clamped (i.e. the label touches the real volume
    edge) is applied by the caller, mirroring
    vtkBinaryLabelmapToClosedSurfaceConversionRule::IsLabelmapPaddingNecessary."""
    kk, jj, ii = np.where(array == label)
    k0, k1 = int(kk.min()) - margin, int(kk.max()) + margin
    j0, j1 = int(jj.min()) - margin, int(jj.max()) + margin
    i0, i1 = int(ii.min()) - margin, int(ii.max()) + margin

    nz, ny, nx = array.shape
    touchesEdge = k0 < 0 or j0 < 0 or i0 < 0 or k1 >= nz or j1 >= ny or i1 >= nx

    k0c, k1c = max(k0, 0), min(k1, nz - 1)
    j0c, j1c = max(j0, 0), min(j1, ny - 1)
    i0c, i1c = max(i0, 0), min(i1, nx - 1)

    cropped = array[k0c:k1c + 1, j0c:j1c + 1, i0c:i1c + 1]

    if touchesEdge:
        # Uniform 1-voxel background pad on all 6 sides, same as the source
        # rule's vtkImageConstantPad call (it pads unconditionally on every
        # side once padding is deemed necessary, not just the touching side).
        cropped = np.pad(cropped, 1, mode="constant", constant_values=0)
        offset = (i0c - 1, j0c - 1, k0c - 1)
    else:
        offset = (i0c, j0c, k0c)

    return cropped, offset


def _numpy_to_vtk_image(array):
    """array is (nz, ny, nx) int32, identity IJK geometry (origin 0, spacing
    1) - the world transform is applied afterwards as a separate step, same
    separation of concerns as the source rule's
    binaryLabelmapWithIdentityGeometry."""
    nz, ny, nx = array.shape
    vtkArray = numpy_to_vtk(array.ravel(order="C"), deep=True, array_type=vtk.VTK_INT)
    vtkArray.SetName("ImageScalars")

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(vtkArray)
    return image


def convert_label(array, label, ijkToRAS, method=DEFAULT_CONVERSION_METHOD,
                   decimation_factor=DEFAULT_DECIMATION_FACTOR,
                   smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                   compute_normals=DEFAULT_COMPUTE_SURFACE_NORMALS,
                   surface_nets_internal_smoothing=DEFAULT_SURFACE_NETS_INTERNAL_SMOOTHING):
    """Converts a single label to a closed surface in RAS. Returns None if
    the label produces no polygons (matches the source rule's early-out when
    GetNumberOfPolys() == 0)."""
    cropped, (i0, j0, k0) = _crop_label(array, label)
    image = _numpy_to_vtk_image(cropped)

    if method == CONVERSION_METHOD_FLYING_EDGES:
        contour = vtk.vtkDiscreteFlyingEdges3D()
        contour.SetInputData(image)
        contour.ComputeGradientsOff()
        contour.ComputeNormalsOff()
        contour.SetValue(0, label)
        contour.Update()
    elif method == CONVERSION_METHOD_SURFACE_NETS:
        contour = vtk.vtkSurfaceNets3D()
        contour.SetInputData(image)
        contour.SmoothingOff()
        if surface_nets_internal_smoothing:
            contour.SmoothingOn()
            fCount = 15.0 * smoothing_factor * smoothing_factor + 9.0 * smoothing_factor
            contour.SetNumberOfIterations(int(np.floor(fCount)))
        contour.SetValue(0, label)
        contour.Update()
    else:
        raise ValueError(f"Unknown conversion method '{method}'")

    processingResult = contour.GetOutput()
    if processingResult.GetNumberOfPolys() == 0:
        return None

    if decimation_factor > 0.0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(processingResult)
        decimator.SetFeatureAngle(60)
        decimator.SplittingOff()
        decimator.PreserveTopologyOn()
        decimator.SetMaximumError(1)
        decimator.SetTargetReduction(decimation_factor)
        decimator.Update()
        processingResult = decimator.GetOutput()

    if smoothing_factor > 0 and not (method == CONVERSION_METHOD_SURFACE_NETS and surface_nets_internal_smoothing):
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(processingResult)
        passBand = 10.0 ** (-4.0 * smoothing_factor)
        numberOfIterations = int(20 + smoothing_factor * 40)
        smoother.SetNumberOfIterations(numberOfIterations)
        smoother.SetPassBand(passBand)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        processingResult = smoother.GetOutput()

    # IJK (crop) -> IJK (full volume) -> RAS, as a single 4x4.
    offsetMatrix = np.eye(4)
    offsetMatrix[:3, 3] = [i0, j0, k0]
    combined = ijkToRAS @ offsetMatrix

    vtkMatrix = vtk.vtkMatrix4x4()
    vtkMatrix.DeepCopy(combined.ravel().tolist())
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtkMatrix)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(processingResult)
    transformFilter.SetTransform(transform)

    if compute_normals and method == CONVERSION_METHOD_FLYING_EDGES:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(transformFilter.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        result = normals.GetOutput()
    else:
        transformFilter.Update()
        result = transformFilter.GetOutput()

    result.GetPointData().RemoveArray("ImageScalars")
    return result


def convert_labelmap_file(input_path, output_dir, labels=None, names=None,
                           method=DEFAULT_CONVERSION_METHOD,
                           decimation_factor=DEFAULT_DECIMATION_FACTOR,
                           smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                           compute_normals=DEFAULT_COMPUTE_SURFACE_NORMALS,
                           surface_nets_internal_smoothing=DEFAULT_SURFACE_NETS_INTERNAL_SMOOTHING,
                           coordinate_space="LPS", file_format="vtk", verbose=True):
    """Converts ONE segmentation volume into one surface file per label
    inside output_dir. Returns the list of written surface file paths."""
    with Stage(f"Reading labelmap {input_path}") if verbose else _noop():
        array, ijkToRAS = _read_labelmap(input_path)

    presentLabels = sorted(int(v) for v in np.unique(array) if v != 0)
    if labels is None:
        labels = presentLabels
    else:
        missing = [l for l in labels if l not in presentLabels]
        if missing:
            logger.warning("requested labels not present in the volume: %s", missing)
        labels = [l for l in labels if l in presentLabels]

    os.makedirs(output_dir, exist_ok=True)
    names = names or {}

    writtenPaths = []
    for label in labels:
        name = names.get(label, names.get(str(label), f"label_{label}"))
        outputPath = os.path.join(output_dir, f"{name}.{file_format}")

        with Stage(f"Converting label {label} ({name})") if verbose else _noop():
            surface = convert_label(
                array, label, ijkToRAS,
                method=method,
                decimation_factor=decimation_factor,
                smoothing_factor=smoothing_factor,
                compute_normals=compute_normals,
                surface_nets_internal_smoothing=surface_nets_internal_smoothing,
            )

        if surface is None:
            logger.info("  Label %d (%s): no polygons produced, skipping.", label, name)
            continue

        if coordinate_space == "LPS":
            surface = flip_lps_ras(surface)

        save_surface(surface, outputPath, coordinate_space=coordinate_space)
        if verbose:
            logger.info("  Saved %d points / %d polys -> %s",
                        surface.GetNumberOfPoints(), surface.GetNumberOfPolys(), outputPath)
        writtenPaths.append(outputPath)

    return writtenPaths


def _is_volume_file(filename):
    return filename.lower().endswith(VOLUME_EXTENSIONS)


def _find_volume_file(segmentation_dir, stem):
    """Finds the raw volume file in segmentation_dir whose own stem (before
    the first '.') matches `stem` exactly - VOLUME_EXTENSIONS has several
    options (.nii vs .nii.gz vs .nrrd...), so this can't just join a fixed
    extension like exclude_files/output naming can afford to elsewhere.
    Returns None if no such file exists."""
    for entry in sorted(os.listdir(segmentation_dir)):
        path = os.path.join(segmentation_dir, entry)
        if os.path.isfile(path) and _is_volume_file(entry) and entry.split(".")[0] == stem:
            return path
    return None


def merge_labelmaps(paths, label=1):
    """Reads N labelmap volumes defined on the SAME grid and returns
    (array, ijkToRAS) for their union: every voxel nonzero in ANY of them
    becomes `label` in the result, everything else 0. Used to combine
    independently-segmented but anatomically-overlapping structures (e.g.
    an airway lumen segmentation and a separate airway wall segmentation)
    into one complete labelmap with a single label, instead of two disjoint
    per-structure surfaces."""
    array, ijkToRAS = _read_labelmap(paths[0])
    merged = array != 0
    for path in paths[1:]:
        otherArray, otherIjkToRAS = _read_labelmap(path)
        if otherArray.shape != array.shape or not np.allclose(otherIjkToRAS, ijkToRAS):
            raise ValueError(f"{path} is not on the same grid as {paths[0]} (shape/IJK-to-RAS differ) - "
                              f"can't merge labelmaps that don't share geometry")
        merged |= otherArray != 0
    return merged.astype(np.int32) * label, ijkToRAS


def convert_merged_airway_wall(segmentation_dir, output_dir, airway_name, airway_wall_name,
                                method=DEFAULT_CONVERSION_METHOD,
                                decimation_factor=DEFAULT_DECIMATION_FACTOR,
                                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                                compute_normals=DEFAULT_COMPUTE_SURFACE_NORMALS,
                                surface_nets_internal_smoothing=DEFAULT_SURFACE_NETS_INTERNAL_SMOOTHING,
                                coordinate_space="LPS", file_format="vtk", verbose=True):
    """Merges the airway lumen (`airway_name`) and airway wall
    (`airway_wall_name`) raw segmentation volumes in segmentation_dir into
    one combined labelmap (see merge_labelmaps()) and converts THAT into a
    single complete airway surface, saved as
    output_dir/<airway_name>.<file_format> - i.e. the normal expected
    filename for this structure (see ConvertConfig.include_airway_wall),
    just built from lumen+wall instead of the lumen alone.

    IDEMPOTENT (same convention as run()): a no-op, returning the existing
    path, if that output file already exists.

    Returns [outputPath] if the merge+conversion produced a surface, or []
    if either raw volume is missing (logged as a warning - the caller's own
    per-file loop still converts whatever raw volume IS present, same as
    if this had never been called) or the merge produced no polygons."""
    outputPath = os.path.join(output_dir, f"{airway_name}.{file_format}")
    if os.path.isfile(outputPath):
        logger.info("  Skipping %s+%s merge: %s already converted - delete to force reconversion.",
                    airway_name, airway_wall_name, outputPath)
        return [outputPath]

    airwayPath = _find_volume_file(segmentation_dir, airway_name)
    wallPath = _find_volume_file(segmentation_dir, airway_wall_name)
    missing = [name for name, path in ((airway_name, airwayPath), (airway_wall_name, wallPath)) if path is None]
    if missing:
        logger.warning("  include_airway_wall is set but couldn't find volume file(s) for %s in %s - "
                        "converting whatever IS present independently instead.", missing, segmentation_dir)
        return []

    with Stage(f"Merging {airway_name} + {airway_wall_name} into one complete airway segmentation") if verbose \
            else _noop():
        array, ijkToRAS = merge_labelmaps([airwayPath, wallPath])

    with Stage("Converting merged airway segmentation") if verbose else _noop():
        surface = convert_label(
            array, 1, ijkToRAS,
            method=method,
            decimation_factor=decimation_factor,
            smoothing_factor=smoothing_factor,
            compute_normals=compute_normals,
            surface_nets_internal_smoothing=surface_nets_internal_smoothing,
        )

    if surface is None:
        logger.warning("  Merged airway segmentation (%s + %s): no polygons produced.",
                        airway_name, airway_wall_name)
        return []

    os.makedirs(output_dir, exist_ok=True)
    if coordinate_space == "LPS":
        surface = flip_lps_ras(surface)
    save_surface(surface, outputPath, coordinate_space=coordinate_space)
    if verbose:
        logger.info("  Saved merged airway+wall surface (%d points / %d polys) -> %s",
                    surface.GetNumberOfPoints(), surface.GetNumberOfPolys(), outputPath)
    return [outputPath]


def run(segmentation_dir, output_dir=None, exclude_files=(),
        method=DEFAULT_CONVERSION_METHOD,
        decimation_factor=DEFAULT_DECIMATION_FACTOR,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        compute_normals=DEFAULT_COMPUTE_SURFACE_NORMALS,
        surface_nets_internal_smoothing=DEFAULT_SURFACE_NETS_INTERNAL_SMOOTHING,
        coordinate_space="LPS", file_format="vtk", verbose=True,
        include_airway_wall=False, airway_name=None, airway_wall_name=None):
    """Converts every (non-excluded) segmentation volume file directly inside
    segmentation_dir into a closed surface - this pipeline's stage 0.

    Mirrors run_pipeline_batch_data.sh's convert_patient_segmentations(): one
    output <stem>.<file_format> per single-label file, or one
    <stem>_<label_name>.<file_format> per label for a multi-label file
    (produced via a temporary per-file directory, removed afterwards).

    output_dir defaults to segmentation_dir with '_vtk' appended (a trailing
    slash on segmentation_dir, if any, is stripped first so the suffix lands
    on the folder name itself, not after it) - e.g. "Data/patient_1_raw" ->
    "Data/patient_1_raw_vtk".

    A single file failing to convert is logged and does NOT stop the rest of
    the folder, matching the bash version's per-file error handling.

    IDEMPOTENT: a segmentation file whose expected output(s) already exist in
    output_dir (<stem>.<file_format>, or <stem>_*.<file_format> for a multi-
    label file) is skipped without re-converting - this stage runs once per
    patient but is invoked from every structure's own pipeline call (see
    vescan.batch), so the 2nd/3rd call for the same patient should see
    everything already done and pass through almost instantly. Delete the
    relevant output file(s) (or the whole output_dir) to force a
    reconversion.

    include_airway_wall (see ConvertConfig/convert_merged_airway_wall()):
    when set, `airway_name` and `airway_wall_name`'s own raw volumes are
    merged into ONE complete airway segmentation (single label, union of
    both) and saved as <airway_name>.<file_format>, instead of each being
    converted into its own separate surface - both raw volumes are then
    skipped by the normal per-file loop below (already handled).

    Returns the list of written surface file paths (pre-existing ones this
    call skipped, and newly-converted ones)."""
    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"Segmentation directory not found: {segmentation_dir}")

    segmentation_dir = os.path.normpath(segmentation_dir)
    if output_dir is None:
        output_dir = segmentation_dir + "_vtk"
    os.makedirs(output_dir, exist_ok=True)

    written = []
    consumedStems = set()
    if include_airway_wall and airway_name and airway_wall_name:
        try:
            merged = convert_merged_airway_wall(
                segmentation_dir, output_dir, airway_name, airway_wall_name,
                method=method, decimation_factor=decimation_factor, smoothing_factor=smoothing_factor,
                compute_normals=compute_normals, surface_nets_internal_smoothing=surface_nets_internal_smoothing,
                coordinate_space=coordinate_space, file_format=file_format, verbose=verbose,
            )
        except Exception:
            logger.exception("  airway+wall merge failed for %s + %s - falling back to converting whatever IS "
                              "present independently below.", airway_name, airway_wall_name)
            merged = []
        if merged:
            written.extend(merged)
            consumedStems = {airway_name, airway_wall_name}

    for entry in sorted(os.listdir(segmentation_dir)):
        seg_path = os.path.join(segmentation_dir, entry)
        if not os.path.isfile(seg_path) or not _is_volume_file(entry):
            continue
        if entry in exclude_files:
            logger.info("  Skipping excluded %s", entry)
            continue

        stem = entry.split(".")[0]
        if stem in consumedStems:
            continue  # already handled by the airway+wall merge above
        existing = (glob.glob(os.path.join(output_dir, f"{stem}.{file_format}")) +
                    glob.glob(os.path.join(output_dir, f"{stem}_*.{file_format}")))
        if existing:
            logger.info("  Skipping %s: already converted (%d file(s), e.g. %s) - delete to force reconversion.",
                        entry, len(existing), os.path.basename(existing[0]))
            written.extend(existing)
            continue

        tmp_dir = os.path.join(output_dir, f".tmp_{stem}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

        try:
            produced = convert_labelmap_file(
                seg_path, tmp_dir,
                method=method, decimation_factor=decimation_factor,
                smoothing_factor=smoothing_factor, compute_normals=compute_normals,
                surface_nets_internal_smoothing=surface_nets_internal_smoothing,
                coordinate_space=coordinate_space, file_format=file_format, verbose=verbose,
            )
        except Exception:
            logger.exception("  conversion failed for %s", seg_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        if not produced:
            logger.warning("  %s produced no surfaces (empty segmentation?)", seg_path)
        elif len(produced) == 1:
            dest = os.path.join(output_dir, f"{stem}.{file_format}")
            shutil.move(produced[0], dest)
            written.append(dest)
        else:
            # Multi-label segmentation file: keep one surface per label,
            # disambiguated with the file's own stem.
            for src in produced:
                dest = os.path.join(output_dir, f"{stem}_{os.path.basename(src)}")
                shutil.move(src, dest)
                written.append(dest)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return written


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _parse_names(names_arg):
    """Accepts either a path to a JSON file ({"1": "Artery", ...}) or an
    inline "1:Artery,2:Veins" string. Returns {int_label: name}."""
    if names_arg is None:
        return {}
    if os.path.isfile(names_arg):
        with open(names_arg, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = dict(pair.split(":", 1) for pair in names_arg.split(","))
    return {int(k): v for k, v in raw.items()}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_labelmap", help="Path to a multi-label segmentation volume "
                                                 "(.nrrd/.nii/.nii.gz/.mha/... - anything SimpleITK reads)")
    parser.add_argument("output_dir", help="Directory to write one surface file per label into")
    parser.add_argument("--labels", type=int, nargs="+", default=None,
                         help="Label values to convert (default: every nonzero label present in the volume)")
    parser.add_argument("--names", default=None,
                         help="Label name mapping: a JSON file ({\"1\": \"Artery\"}) or inline "
                              "\"1:Artery,2:Veins\" (default: label_<N>)")
    parser.add_argument("--method", choices=[CONVERSION_METHOD_FLYING_EDGES, CONVERSION_METHOD_SURFACE_NETS],
                         default=DEFAULT_CONVERSION_METHOD,
                         help=f"Iso-contouring method, matching Slicer's ConversionMethod parameter "
                              f"(default {DEFAULT_CONVERSION_METHOD}, same default as Slicer)")
    parser.add_argument("--decimation", type=float, default=DEFAULT_DECIMATION_FACTOR,
                         help=f"Decimation factor, 0.0-1.0 (default {DEFAULT_DECIMATION_FACTOR}, same as Slicer)")
    parser.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING_FACTOR,
                         help=f"Smoothing factor, 0.0-1.0 (default {DEFAULT_SMOOTHING_FACTOR}, same as Slicer)")
    parser.add_argument("--no-normals", action="store_true",
                         help="Skip surface normal computation (only applies to --method flying-edges, "
                              "which computes them by default, matching Slicer)")
    parser.add_argument("--surfacenets-internal-smoothing", action="store_true",
                         help="Use vtkSurfaceNets3D's own internal smoothing instead of "
                              "vtkWindowedSincPolyDataFilter (only applies to --method surface-nets; "
                              "off by default, matching Slicer)")
    parser.add_argument("--coordinate-space", choices=["LPS", "RAS"], default="LPS",
                         help="Coordinate space to write output surfaces in (default LPS, matching every other "
                              ".vtk file already produced by this pipeline - see vescan/io.py)")
    parser.add_argument("--format", choices=["vtk", "vtp"], default="vtk",
                         help="Output surface file format (default vtk)")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()

    names = _parse_names(args.names)

    convert_labelmap_file(
        args.input_labelmap,
        args.output_dir,
        labels=args.labels,
        names=names,
        method=args.method,
        decimation_factor=args.decimation,
        smoothing_factor=args.smoothing,
        compute_normals=not args.no_normals,
        surface_nets_internal_smoothing=args.surfacenets_internal_smoothing,
        coordinate_space=args.coordinate_space,
        file_format=args.format,
    )


if __name__ == "__main__":
    sys.exit(main())