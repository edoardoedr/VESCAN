#!/usr/bin/env python3
"""Collects every patient's statistics.json under one root into a single
.xlsx for filtering and statistics.

    python3 aggregate_statistics.py RESULTS_ROOT [RESULTS_ROOT ...] [-o out.xlsx]

Each RESULTS_ROOT is a folder holding one subfolder per patient. The layout
it expects is patient folder, then vessel folder, then the file:

    RESULTS_ROOT/<patient>/<vessel>/statistics.json
    e.g.  .../nsclc_radiogenomics_centerline/R01-006/final_export_veins/statistics.json
                                             ^patient ^vessel

so `patient` is the first directory below RESULTS_ROOT and `vessel` is the
one holding the file, with any "final_export_" prefix stripped. The search
is a plain recursive walk rather than a fixed-depth glob, so an extra level
in between (the local Outputs/ tree has <patient>/centerline/<vessel>/)
still resolves to the same patient and vessel.

SEVERAL DATASETS AT ONCE: pass more than one root and they land in the same
workbook, told apart by a `dataset` column (the root's folder name, or the
NAME in a `NAME=PATH` argument). That column is not decoration - patient ids
are only unique WITHIN a dataset, so without it two different patients that
happen to share an id would merge into one in any group-by. Every sheet is
keyed dataset-first for the same reason.

Rows are de-duplicated by the JSON file's real path, so passing the same
root twice, or two roots where one contains the other, counts each file
once instead of doubling it.

Reads only finished JSON files and writes one .xlsx - it never touches the
pipeline or its outputs, so it is safe to re-run at any time.

THREE SHEETS, because one table cannot answer every question without either
duplicating rows or hiding some:

  crossings   one row per crossing vessel - the main analysis table.
  structures  one row per patient x structure, INCLUDING the ones with zero
              crossings. Without it "how many patients have an interlobar
              vein" has no denominator: a structure with no crossing
              contributes no row to `crossings` at all, so it would silently
              drop out of every rate instead of counting as a zero.
  piercings   one row per individual fissure piercing, for angle-level work.
              Deliberately NOT merged into `crossings`: a vessel with five
              piercings must not become five rows there, or every count of
              "number of crossings" over-reports interlobar vessels.

PATIENT ID COMES FROM THE PATH, not from the JSON's own `patient` field.
That field is filled in by statistics.py by guessing from the folder shape,
and it assumes the <patient>/centerline/<vessel>/ layout: on a tree without
that intermediate level - the one described above - it walks up one
directory too far and records the DATASET name for every patient. Verified:
for .../nsclc_radiogenomics_centerline/R01-006/final_export_veins it writes
patient='nsclc_radiogenomics_centerline'. The first path component below
RESULTS_ROOT is unambiguous whatever the depth, so that is what is used, and
a disagreement is reported with --verbose rather than silently preferred.

Requires pandas and openpyxl (pip install pandas openpyxl).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Mirrors vescan.stages.build_lobe_segments.FISSURE_LABEL. Copied
# rather than imported because that module pulls in vtk, and aggregating
# finished JSON files must not require a working VMTK install. Unknown
# values fall through to the raw number, so a future fissure value shows up
# as itself instead of vanishing.
FISSURE_LABEL = {
    70: "Scissura orizzontale destra",
    71: "Scissura obliqua destra",
    72: "Scissura obliqua sinistra",
}

STATISTICS_FILENAME = "statistics.json"

# Column ORDER for each sheet - pandas would otherwise order them by first
# appearance, which puts the identity columns wherever they happen to fall.
CROSSING_COLUMNS = [
    "dataset", "patient", "structure", "vessel_type",
    "crossing", "crossing_type", "from_lobe", "to_lobe", "lobe_pair",
    "length_mm", "length_to_end_mm", "length_after_piercing_mm",
    "points", "n_groups", "start_generation",
    "n_piercings", "angle_median_deg", "angle_min_deg", "angle_max_deg",
    "fissures", "radius_mm", "patch_rms_max_mm",
    "n_crossings_in_structure", "source",
]

STRUCTURE_COLUMNS = [
    "dataset", "patient", "structure", "vessel_type",
    "n_crossings", "n_interlobar", "n_translobar", "n_unclassified",
    "total_length_mm", "source",
]

PIERCING_COLUMNS = [
    "dataset", "patient", "structure", "vessel_type",
    "crossing", "crossing_type", "piercing",
    "angle_deg", "grazing", "fissure", "fissure_contact",
    "radius_mm", "distal_length_mm", "patch_rms_mm", "group", "source",
]


def find_statistics_files(root):
    """Every statistics.json below root, at any depth, sorted."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if STATISTICS_FILENAME in filenames:
            found.append(os.path.join(dirpath, STATISTICS_FILENAME))
    return sorted(found)


def patient_and_structure(path, root):
    """(patient, structure) from the file's own path - see the module
    docstring on why the JSON's `patient` field is not trusted.

    patient   = the first directory below root
    structure = the containing folder, with any "final_export_" prefix
                stripped ("final_export_veins" -> "veins")"""
    parts = os.path.relpath(path, root).split(os.sep)
    patient = parts[0] if len(parts) > 1 else os.path.basename(os.path.abspath(root))

    structure = os.path.basename(os.path.dirname(os.path.abspath(path)))
    prefix = "final_export_"
    if structure.startswith(prefix):
        structure = structure[len(prefix):]
    return patient, structure


def _fissure_names(values):
    return " + ".join(FISSURE_LABEL.get(value, str(value)) for value in values or [])


def _lobe_pair(crossing):
    """"RUL -> RML", the single column to pivot on when counting which
    transitions occur. Empty when the origin lobe could not be read (the
    label starts at the first point of every group in the run, so no
    pre-crossing point is left - see statistics.py)."""
    origin, destination = crossing.get("from_lobe"), crossing.get("to_lobe")
    return f"{origin} -> {destination}" if origin and destination else ""


def parse_root_arg(argument):
    """"PATH" -> (basename, PATH); "NAME=PATH" -> (NAME, PATH).

    The NAME= form is for when two datasets' folders have the same basename,
    or when the folder name is not what you want to see in the column."""
    if "=" in argument:
        name, path = argument.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Atteso NAME=PATH, ricevuto '{argument}'")
        return name, path
    return os.path.basename(os.path.normpath(argument)), argument


def collect(roots, verbose=False):
    """Reads every statistics JSON under each (datasetName, root) in `roots`.

    Returns (crossings, structures, piercings, skipped) - the first three as
    DataFrames with the column order above, the last as [(path, error)]."""
    crossingRows, structureRows, piercingRows, skipped = [], [], [], []
    seenPaths = set()

    for dataset, root in roots:
        if verbose:
            print(f"[{dataset}] {root}")
        for path in find_statistics_files(root):
            # Keyed on the resolved path so overlapping or repeated roots
            # contribute each file once - see the module docstring.
            realPath = os.path.realpath(path)
            if realPath in seenPaths:
                if verbose:
                    print(f"  (gia' letto, salto) {path}")
                continue
            seenPaths.add(realPath)

            try:
                with open(path, encoding="utf-8") as f:
                    document = json.load(f)
            except (OSError, json.JSONDecodeError) as error:
                skipped.append((path, str(error)))
                continue

            patient, structure = patient_and_structure(path, root)
            source = os.path.relpath(path, root)
            identity = {"dataset": dataset, "patient": patient, "structure": structure,
                        "vessel_type": document.get("vessel_type") or ""}

            if verbose:
                print(f"  {source}")
                if document.get("patient") not in (None, patient):
                    print(f"    nota: il JSON dichiara patient={document.get('patient')!r}, "
                          f"dal percorso risulta {patient!r} - uso quello del percorso")

            _append_rows(document, identity, source, crossingRows, structureRows, piercingRows)

    def frame(rows, columns, sortBy):
        # reindex(columns=...) both orders the columns and creates any that no
        # input happened to carry, so the sheet's shape never depends on which
        # patients were in the folder.
        table = pd.DataFrame(rows).reindex(columns=columns)
        return table.sort_values(sortBy, kind="stable").reset_index(drop=True) if len(table) else table

    return (frame(crossingRows, CROSSING_COLUMNS, ["dataset", "patient", "structure", "crossing"]),
            frame(structureRows, STRUCTURE_COLUMNS, ["dataset", "patient", "structure"]),
            frame(piercingRows, PIERCING_COLUMNS,
                  ["dataset", "patient", "structure", "crossing", "piercing"]),
            skipped)


def _append_rows(document, identity, source, crossingRows, structureRows, piercingRows):
    """Flattens ONE statistics document into the three row lists."""
    structureRows.append(dict(
        identity,
        n_crossings=document.get("n_crossings"),
        n_interlobar=document.get("n_interlobar"),
        n_translobar=document.get("n_translobar"),
        n_unclassified=document.get("n_unclassified"),
        total_length_mm=document.get("total_length_mm"),
        source=source,
    ))

    for crossing in document.get("crossings") or []:
        crossingRows.append(dict(
            identity,
            crossing=crossing.get("crossing"),
            crossing_type=crossing.get("crossing_type"),
            from_lobe=crossing.get("from_lobe") or "",
            to_lobe=crossing.get("to_lobe") or "",
            lobe_pair=_lobe_pair(crossing),
            length_mm=crossing.get("length_mm"),
            length_to_end_mm=crossing.get("length_to_end_mm"),
            length_after_piercing_mm=crossing.get("length_after_piercing_mm"),
            points=crossing.get("points"),
            n_groups=crossing.get("n_groups"),
            start_generation=crossing.get("start_generation"),
            n_piercings=crossing.get("n_piercings"),
            angle_median_deg=crossing.get("angle_median_deg"),
            angle_min_deg=crossing.get("angle_min_deg"),
            angle_max_deg=crossing.get("angle_max_deg"),
            fissures=_fissure_names(crossing.get("fissures")),
            radius_mm=crossing.get("radius_mm"),
            patch_rms_max_mm=crossing.get("patch_rms_max_mm"),
            n_crossings_in_structure=document.get("n_crossings"),
            source=source,
        ))

        for index, piercing in enumerate(crossing.get("piercings") or [], start=1):
            piercingRows.append(dict(
                identity,
                crossing=crossing.get("crossing"),
                crossing_type=crossing.get("crossing_type"),
                piercing=index,
                angle_deg=piercing.get("angle_deg"),
                grazing=piercing.get("grazing"),
                fissure=_fissure_names([piercing["fissure"]] if piercing.get("fissure") else []),
                fissure_contact=piercing.get("fissure_contact"),
                radius_mm=piercing.get("radius_mm"),
                distal_length_mm=piercing.get("distal_length_mm"),
                patch_rms_mm=piercing.get("patch_rms_mm"),
                group=piercing.get("group"),
                source=source,
            ))


def write_xlsx(path, sheets):
    """sheets: [(name, DataFrame)] - each on its own sheet, with the header
    frozen and an autofilter over it so the file is usable for filtering the
    moment it opens."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets:
            table.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for index, column in enumerate(table.columns, start=1):
                width = max(len(str(column)),
                            *(len(str(v)) for v in table[column].head(200))) if len(table) else len(str(column))
                worksheet.column_dimensions[worksheet.cell(1, index).column_letter].width = min(width + 2, 45)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_roots", nargs="+", metavar="RESULTS_ROOT",
                         help="One or more folders, each with one subfolder per patient (e.g. "
                              ".../nsclc_radiogenomics_centerline). Several are merged into the "
                              "same workbook and told apart by the 'dataset' column, which takes "
                              "the folder's name - write NAME=PATH to set it yourself")
    parser.add_argument("-o", "--output", default="crossing_statistics.xlsx",
                         help="Output .xlsx (default: crossing_statistics.xlsx)")
    parser.add_argument("--csv-dir", default=None,
                         help="Also write one CSV per sheet into this folder")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="List each file read, and report any patient id in a JSON "
                              "that disagrees with its own path")
    return parser


def main():
    args = build_arg_parser().parse_args()

    try:
        roots = [parse_root_arg(argument) for argument in args.results_roots]
    except ValueError as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    missing = [path for _name, path in roots if not os.path.isdir(path)]
    if missing:
        for path in missing:
            print(f"Errore: '{path}' non e' una cartella.", file=sys.stderr)
        return 1

    duplicated = {name for name, _ in roots if sum(1 for n, _ in roots if n == name) > 1}
    if duplicated:
        print(f"Errore: nome dataset ripetuto: {', '.join(sorted(duplicated))}. "
              f"Usa la forma NOME=PERCORSO per distinguerli.", file=sys.stderr)
        return 1

    crossings, structures, piercings, skipped = collect(roots, verbose=args.verbose)
    if not len(structures):
        print(f"Nessun {STATISTICS_FILENAME} trovato sotto: "
              f"{', '.join(path for _n, path in roots)}", file=sys.stderr)
        return 1

    sheets = [("crossings", crossings), ("structures", structures), ("piercings", piercings)]
    write_xlsx(args.output, sheets)

    kinds = crossings["crossing_type"].value_counts() if len(crossings) else {}
    patients = structures.groupby("dataset")["patient"].nunique()
    print(f"{len(roots)} dataset, {int(patients.sum())} pazienti, {len(structures)} strutture "
          f"-> {args.output}")
    # Iterates the ROOTS, not the datasets present in the data: one that
    # contributed nothing (wrong path, or entirely covered by another root)
    # is worth seeing as an explicit zero rather than being absent.
    for dataset, _path in roots:
        nPatients = int(patients.get(dataset, 0))
        nCrossings = int((crossings["dataset"] == dataset).sum()) if len(crossings) else 0
        note = "   <-- nessun file trovato" if nPatients == 0 else ""
        print(f"  {dataset:34s} {nPatients:4d} pazienti, {nCrossings:4d} crossing{note}")
    print(f"  crossings  {len(crossings):5d} righe  "
          f"({kinds.get('interlobar', 0)} interlobar, {kinds.get('translobar', 0)} translobar, "
          f"{kinds.get('unclassified', 0)} unclassified)")
    print(f"  structures {len(structures):5d} righe  (incluse quelle con 0 crossing)")
    print(f"  piercings  {len(piercings):5d} righe")

    if skipped:
        print(f"  {len(skipped)} file illeggibili:", file=sys.stderr)
        for path, error in skipped:
            print(f"    {path}: {error}", file=sys.stderr)

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        for name, table in sheets:
            csvPath = os.path.join(args.csv_dir, f"{name}.csv")
            table.to_csv(csvPath, index=False)
            print(f"  {csvPath}  ({len(table)} righe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
