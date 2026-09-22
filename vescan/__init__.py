"""Python package backing main.py, the single entrypoint for the standalone
Slicer/VMTK lung centerline pipeline.

vescan.stages        - one module per pipeline stage (0-10), each
                               exposing a plain run() function plus a thin
                               argparse CLI for standalone debugging
vescan.io            - shared surface I/O (load/save, LPS/RAS,
                               largest-connected-component filtering)
vescan.lobes         - canonical pulmonary lobe order/side membership,
                               the single source of truth every stage dealing
                               with lobes imports from
vescan.config        - typed schema + loader for the experiment JSON
                               config (see configs/pipeline_config.example.json)
vescan.paths         - single source of truth for one pipeline run's
                               derived output file names
vescan.orchestrator  - run_pipeline(): runs every enabled stage
                               in-process for one surface/structure
vescan.batch         - dispatches to a single patient or a whole
                               batch (ProcessPoolExecutor) based on the
                               config's own batch.enabled - the only function
                               main.py itself calls
"""