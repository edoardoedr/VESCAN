#!/usr/bin/env python3
"""Single entrypoint for the standalone Slicer/VMTK centerline pipeline.

    python3 main.py config.json

That is the entire CLI - everything else (single patient vs a whole batch,
which structures to process, every tuning parameter) comes from the config
file itself; see configs/pipeline_config.example.json and, in particular,
its "batch" section:
  - batch.enabled=false: processes ONE patient, whose segmentation/surfaces/
    results folders are batch.patients_root_dir/surfaces_root_dir/
    results_root_dir directly.
  - batch.enabled=true: processes EVERY patient subfolder found in
    batch.patients_root_dir, up to batch.parallel_jobs at a time.
See vescan/batch.py for that dispatch and the per-patient/per-
structure orchestration.

Every stage runs IN-PROCESS (a plain function call under
vescan/stages/), not as a subprocess - main.py itself must therefore
be run from within the conda env that has vtk/vmtk/pyfqmr installed (see
conda_env in the config). This is faster than shelling out per stage (no
per-stage interpreter startup/VTK import) and lets later stages reuse an
earlier stage's already-loaded data directly (see e.g. the centerline/
build_graph and lobe_reachability/anatomical_segments stages' own
docstrings).
"""

import argparse
import sys

from vescan import config as configmod
from vescan.logging_config import setup_console_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a JSON experiment config (see "
                                        "configs/pipeline_config.example.json) - its 'batch' section decides "
                                        "single-patient vs batch mode and every input/output path; nothing else "
                                        "is taken from the command line.")
    return parser


def main() -> int:
    # Console logging for THIS process - a single patient's run_pipeline()
    # keeps using it directly; a batch run's own worker processes replace it
    # with their own per-patient file logging (see
    # vescan.batch._process_patient_worker()), so only this main
    # process' own summary/status lines end up on the terminal there.
    setup_console_logging()

    args = build_arg_parser().parse_args()
    cfg = configmod.load_config(args.config)

    from vescan import batch as batchmod
    return batchmod.run(cfg)


if __name__ == "__main__":
    sys.exit(main())