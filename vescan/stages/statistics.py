#!/usr/bin/env python3
"""
[11/11] Crossing-vessel statistics - the pipeline's reporting stage.

Answers five questions per structure, and nothing else:
  - how many crossing vessels are there,
  - which KIND each one is - interlobar or translobar,
  - which lobe does each one go from and to,
  - how long each one is - three different lengths, see below,
  - at what angle it meets the fissure, and how many times.

THREE LENGTHS, because the question has three honest answers (the full
reasoning, and the data, are in vescan.crossings' docstring):
  length_mm                 the whole downstream subtree, every branch summed
  length_to_end_mm          longest single path from the crossing label's
                            start to a distal leaf - "how far does it go"
  length_after_piercing_mm  the same, anchored at the fissure piercing itself
On R01-091's largest venous crossing these read 402.6 / 86.1 / smaller still:
the first counts 20 leaves' worth of branches, the others follow one path.

Reads only what earlier stages wrote, so it is cheap, purely additive, and
safe to re-run on an existing output folder on its own.

INPUTS
  anatomical_segments.vtk                  AnatomicalSegmentPoint + LobeLabelPoint
  supporting_files/10_anatomical_segments.json  the label -> name legend (for block_start)
  supporting_files/05_branch_tree_topology.json the group graph
  lobe_fissures.vtk             the interlobar fissure sheets (PATIENT-level,
                                see paths.LobeOverviewPaths) - optional: without
                                it the counts and lengths are unchanged and every
                                vessel reports type "unclassified"

OUTPUTS
  statistics.json  one small object, with the per-piercing detail
  statistics.csv   ONE ROW PER CROSSING VESSEL, so concatenating the file
                      across patients and vessel types is the whole
                      aggregation step. A structure with no crossing still
                      writes one row, carrying n_crossings=0 and empty
                      crossing columns, so "none found" stays distinguishable
                      from "never processed".

WHAT COUNTS AS A CROSSING VESSEL, how the interlobar/translobar split is
decided, and the data behind both, all live in vescan.crossings - this
module is the reporting layer over it and owns no definitions of its own.
Stage 10 writes the same classification onto the centerline itself
(CrossingId/CrossingType), computed by that same module, so the `crossing`
column here and the CrossingId array there refer to the same vessel.

Usage (CLI, standalone - for debugging one stage in isolation):
    python -m vescan.stages.statistics OUTPUT_DIR --vessel-type artery \\
        --fissures ../../vtk/lobe_fissures.vtk
"""

import argparse
import csv
import json
import logging
import os
import sys

import vtk

from vescan import crossings as crossingsmod
from vescan import paths as pathsmod
from vescan.stages.anatomical_segments import crossing_values_for

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["patient", "structure", "vessel_type", "n_crossings",
                "n_interlobar", "n_translobar",
                "crossing", "crossing_type", "from_lobe", "to_lobe",
                "length_mm", "length_to_end_mm", "length_after_piercing_mm", "points",
                "n_piercings", "angle_median_deg", "angle_min_deg", "angle_max_deg",
                "fissures", "radius_mm", "patch_rms_max_mm"]


def find_crossing_vessels(polyData, anatomicalLegend, topology, fissureSurface=None,
                           min_piercings_for_interlobar=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR,
                           interlobar_max_median_angle_deg=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """The crossing vessels of one structure, longest first - see
    vescan.crossings.analyse(), which owns the definition and the
    interlobar/translobar rule. Kept as a named function here because it is
    this stage's whole question, and because it guards the inputs.

    Returns None when an input needed to answer the question is missing, so
    the caller can say so instead of reporting a misleading zero."""
    if polyData is None:
        return None
    vesselType = (anatomicalLegend or {}).get("vessel_type")
    if vesselType is None:
        return None
    # All three values that mean "crossing" - the plain one and both refined
    # ones - since stage 10 emits the refined labels whenever it could
    # classify, and older outputs still carry only the plain one.
    return crossingsmod.analyse(polyData, topology, crossing_values_for(vesselType),
                                fissureSurface=fissureSurface,
                                min_piercings_for_interlobar=min_piercings_for_interlobar,
                                interlobar_max_median_angle_deg=interlobar_max_median_angle_deg)


def _read_polydata(path):
    if not path or not os.path.isfile(path):
        return None
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    output = reader.GetOutput()
    return output if output and output.GetNumberOfPoints() else None


def _read_json(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default_names(output_dir):
    """(patient, structure) guessed from the conventional layout
    <patient>/centerline/final_export_<structure>, so the CSV identifies its
    own rows once concatenated. Either can be overridden by the caller."""
    normalized = os.path.normpath(output_dir)
    structure = os.path.basename(normalized)
    prefix = "final_export_"
    if structure.startswith(prefix):
        structure = structure[len(prefix):]
    patient = os.path.basename(os.path.dirname(os.path.dirname(normalized)))
    return patient, structure


def _fissure_names(values):
    """The `fissures` column: the anatomical fissure names a crossing meets,
    from build_lobe_segments' Fissure values, joined - e.g. "obliqua dx"."""
    from vescan.stages.build_lobe_segments import FISSURE_LABEL
    return "+".join(FISSURE_LABEL.get(value, str(value)) for value in values or [])


def save_csv(document, csv_path):
    """One row per crossing vessel; one blank-crossing row when there are
    none, so a zero stays visible after concatenation.

    Per-piercing detail deliberately stays out of the CSV and lives in the
    JSON: a vessel with five piercings must not become five rows, or every
    aggregation over "number of crossings" silently counts interlobar
    vessels several times."""
    common = {
        "patient": document["patient"],
        "structure": document["structure"],
        "vessel_type": document["vessel_type"],
        "n_crossings": document["n_crossings"],
        "n_interlobar": document["n_interlobar"],
        "n_translobar": document["n_translobar"],
    }
    blank = {column: "" for column in CSV_COLUMNS if column not in common}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        if not document["crossings"]:
            writer.writerow(dict(common, **blank))
            return
        for crossing in document["crossings"]:
            writer.writerow(dict(
                common,
                crossing=crossing["crossing"],
                crossing_type=crossing["crossing_type"],
                from_lobe=crossing["from_lobe"] or "",
                to_lobe=crossing["to_lobe"] or "",
                length_mm=crossing["length_mm"],
                length_to_end_mm="" if crossing["length_to_end_mm"] is None else crossing["length_to_end_mm"],
                length_after_piercing_mm=("" if crossing["length_after_piercing_mm"] is None
                                          else crossing["length_after_piercing_mm"]),
                points=crossing["points"],
                n_piercings=crossing["n_piercings"],
                angle_median_deg="" if crossing["angle_median_deg"] is None else crossing["angle_median_deg"],
                angle_min_deg="" if crossing["angle_min_deg"] is None else crossing["angle_min_deg"],
                angle_max_deg="" if crossing["angle_max_deg"] is None else crossing["angle_max_deg"],
                fissures=_fissure_names(crossing["fissures"]),
                radius_mm="" if crossing["radius_mm"] is None else crossing["radius_mm"],
                patch_rms_max_mm="" if crossing["patch_rms_max_mm"] is None else crossing["patch_rms_max_mm"],
            ))


def _count_type(vessels, name):
    return sum(1 for vessel in vessels or [] if vessel["crossing_type"] == name)


def run(output_dir, vessel_type, centerline_path=None, anatomical_legend_path=None,
        topology_path=None, fissures_path=None, output_path=None, csv_output_path=None,
        patient=None, structure_name=None,
        min_piercings_for_interlobar=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR,
        interlobar_max_median_angle_deg=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG):
    """Finds one structure's crossing vessels and writes them to
    OUTPUT_DIR/statistics.json and .csv. Returns the document.

    fissures_path is the patient-level lobe_fissures.vtk (see
    paths.LobeOverviewPaths). Without it everything still works except the
    interlobar/translobar split, which needs the fissure geometry - the
    document then reports every vessel as "unclassified" rather than
    guessing."""
    # Defaults for every path not explicitly passed in (standalone CLI use -
    # the orchestrator always passes all of them itself) come from
    # vescan.paths, the single source of truth for the naming convention,
    # rather than being re-derived here.
    default_paths = pathsmod.PipelinePaths(output_dir)
    centerline_path = centerline_path or default_paths.anatomical_segments
    anatomical_legend_path = anatomical_legend_path or default_paths.anatomical_segments_legend
    topology_path = topology_path or default_paths.branch_tree_topology
    output_path = output_path or default_paths.statistics
    csv_output_path = csv_output_path or default_paths.statistics_csv
    defaultPatient, defaultStructure = _default_names(output_dir)

    fissureSurface = _read_polydata(fissures_path)
    if fissures_path and fissureSurface is None:
        logger.warning("Fissure surface '%s' not found - crossing vessels will be reported without "
                        "the interlobar/translobar split.", fissures_path)

    vessels = find_crossing_vessels(
        _read_polydata(centerline_path),
        _read_json(anatomical_legend_path),
        _read_json(topology_path),
        fissureSurface=fissureSurface,
        min_piercings_for_interlobar=min_piercings_for_interlobar,
        interlobar_max_median_angle_deg=interlobar_max_median_angle_deg,
    )
    if vessels is None:
        logger.warning("Cannot report crossing vessels for %s: needs %s, %s and %s (an earlier stage is "
                        "disabled or this is an older output).", output_dir,
                        os.path.basename(centerline_path), os.path.basename(anatomical_legend_path),
                        os.path.basename(topology_path))

    document = {
        "patient": patient or defaultPatient,
        "structure": structure_name or defaultStructure,
        "vessel_type": vessel_type,
        "n_crossings": len(vessels) if vessels is not None else None,
        "n_interlobar": _count_type(vessels, "interlobar") if vessels is not None else None,
        "n_translobar": _count_type(vessels, "translobar") if vessels is not None else None,
        "n_unclassified": _count_type(vessels, "unclassified") if vessels is not None else None,
        "total_length_mm": round(sum(v["length_mm"] for v in vessels), 3) if vessels else 0.0,
        "crossings": vessels or [],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    save_csv(document, csv_output_path)

    logger.info("Crossing vessels: %s (%s interlobar, %s translobar), %.1fmm total -> %s",
                document["n_crossings"] if document["n_crossings"] is not None else "unavailable",
                document["n_interlobar"], document["n_translobar"],
                document["total_length_mm"], output_path)
    for crossing in document["crossings"]:
        logger.info("  %d. %-12s %s -> %s   %.2fmm   %d perforazion%s%s",
                    crossing["crossing"], crossing["crossing_type"],
                    crossing["from_lobe"] or "?", crossing["to_lobe"] or "?",
                    crossing["length_mm"], crossing["n_piercings"],
                    "i" if crossing["n_piercings"] != 1 else "e",
                    f"   angolo mediano {crossing['angle_median_deg']:.1f} deg"
                    if crossing["angle_median_deg"] is not None else "")
    return document


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="A finished structure's output folder")
    parser.add_argument("--vessel-type", choices=["artery", "vein", "airway"], required=True)
    parser.add_argument("--centerline", default=None,
                         help="Default: OUTPUT_DIR/anatomical_segments.vtk")
    parser.add_argument("--anatomical-legend", default=None,
                         help="Default: OUTPUT_DIR/supporting_files/10_anatomical_segments.json")
    parser.add_argument("--topology", default=None,
                         help="Default: OUTPUT_DIR/supporting_files/05_branch_tree_topology.json")
    parser.add_argument("--fissures", default=None,
                         help="The patient's lobe_fissures.vtk (build_lobe_segments' output, next to "
                              "the lobe surfaces). Without it the interlobar/translobar split cannot "
                              "be made and every crossing is reported as 'unclassified'")
    parser.add_argument("--output", default=None, help="Default: OUTPUT_DIR/statistics.json")
    parser.add_argument("--csv-output", default=None, help="Default: OUTPUT_DIR/statistics.csv")
    parser.add_argument("--patient", default=None,
                         help="Value for the CSV's patient column (default: guessed from the path)")
    parser.add_argument("--structure-name", default=None,
                         help="Value for the CSV's structure column (default: the folder's basename)")
    parser.add_argument("--min-piercings-for-interlobar", type=int,
                         default=crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR, metavar="N",
                         help="Must match whatever vescan.stages.anatomical_segments used for this "
                              "run, so the CrossingType array and this report's crossing_type column agree "
                              f"(default: {crossingsmod.MIN_PIERCINGS_FOR_INTERLOBAR}).")
    parser.add_argument("--interlobar-max-median-angle-deg", type=float,
                         default=crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG, metavar="DEG",
                         help="Must match whatever vescan.stages.anatomical_segments used for this "
                              f"run (default: {crossingsmod.INTERLOBAR_MAX_MEDIAN_ANGLE_DEG}).")
    return parser


def main():
    from vescan.logging_config import setup_console_logging
    setup_console_logging()

    args = build_arg_parser().parse_args()
    run(args.output_dir, args.vessel_type,
        centerline_path=args.centerline,
        anatomical_legend_path=args.anatomical_legend,
        topology_path=args.topology,
        fissures_path=args.fissures,
        output_path=args.output,
        csv_output_path=args.csv_output,
        patient=args.patient,
        structure_name=args.structure_name,
        min_piercings_for_interlobar=args.min_piercings_for_interlobar,
        interlobar_max_median_angle_deg=args.interlobar_max_median_angle_deg)


if __name__ == "__main__":
    sys.exit(main())
