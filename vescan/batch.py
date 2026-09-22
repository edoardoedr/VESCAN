"""Patient orchestration - the only entry point main.py calls, for both a
single patient and a full batch. Which mode runs is decided entirely by the
config's own batch.enabled (see vescan/config.py's BatchConfig) - main.py
itself takes no other input, so this module is where "one patient vs many"
actually gets resolved.

Per patient (single or, in batch mode, one of many, run via process_patient()):
  vescan.orchestrator.run_pipeline() is called once per structure
  enabled in batch.structures, into that patient's own
  results_dir/final_export_<structure>. Segmentation -> surface conversion
  is entirely that call's own stage 0 (pipeline.stages.convert/pipeline.
  convert.* - the same knob a standalone single-structure run uses), NOT a
  separate step here - vescan.stages.convert_segmentations.run() is
  idempotent (skips a file whose output already exists), so calling it 3
  times per patient (once per structure) costs almost nothing after the
  first: the first structure processed actually converts the whole
  segmentation folder, the other two see it already done.

batch.enabled=false (single patient): batch.patients_root_dir/
surfaces_root_dir/results_root_dir are used directly as this one patient's
own folders - process_patient() runs in the current process, logging to
both the terminal (this process is an attended session, unlike a batch
worker) AND its own results_root_dir/pipeline.log (see run_single()'s own
setup_console_and_file_logging() call), so a single-patient run also leaves
a persistent record, same as batch mode's.

batch.enabled=true (many patients): batch.patients_root_dir has one
subfolder per patient; up to batch.parallel_jobs patients are processed
concurrently via ProcessPoolExecutor - each patient is a genuinely separate
OS process, so unlike a single shell interleaving several background jobs'
output, this needs no manual redirection trickery for one patient's output
not to garble another's: each patient's own worker process redirects ITS OWN
stdout/stderr (both the Python-level logging AND the OS-level file
descriptors - see logging_config.setup_file_logging()) to that patient's own
pipeline.log for the whole call (see _process_patient_worker()), and only
the main process ever logs directly to the terminal (short per-patient
status lines as each one finishes). A single patient (or one structure of
one patient) failing is logged and does NOT stop the batch - it moves on and
reports a summary of failures at the end. Mind that N parallel patients means
roughly Nx CPU/RAM usage (VMTK + decimation are heavy) - lower
batch.parallel_jobs if the machine struggles.
"""

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from vescan import config as configmod
from vescan.logging_config import setup_console_and_file_logging, setup_file_logging

logger = logging.getLogger(__name__)

# (structure key in batch.structures, output subfolder suffix, vessel_type
# passed to run_pipeline - matches label_anatomical_segments.py's naming
# blocks) - same order/mapping run_pipeline_batch_data.sh used.
STRUCTURES = (
    ("artery", "artery", "artery"),
    ("veins", "veins", "vein"),
    ("airways", "airways", "airway"),
)


def process_patient(segmentation_dir: str, surfaces_dir: str, results_dir: str,
                     cfg: configmod.RootConfig) -> bool:
    """Runs every enabled structure (batch.structures) for ONE patient into
    results_dir/final_export_<structure>. Segmentation -> surface conversion
    is NOT done here directly - it happens inside run_pipeline()'s own stage
    0 (pipeline.stages.convert/pipeline.convert.*), the same single knob a
    standalone orchestrator.run_pipeline() call uses, so there is exactly
    one place that decides whether/how conversion runs. Since that's called
    once per structure here, vescan.stages.convert_segmentations.run()
    being idempotent (skips a segmentation file whose output already exists)
    is what keeps this cheap: only the FIRST structure processed for this
    patient actually converts anything; the other two see it already done
    and pass through almost instantly.

    Logs through the standard `logging` module - the caller decides
    whether/how to capture it: run_single() lets it go straight to the
    terminal, _process_patient_worker() redirects it to a log file.
    Returns True if every enabled structure succeeded."""
    try:
        from vescan.orchestrator import run_pipeline
    except ImportError as e:
        logger.error("%s (this process must run from within the '%s' conda env - every stage "
                      "imports vtk/vmtk directly).", e, cfg.conda_env)
        return False

    b = cfg.batch
    ok = True

    for structure_key, output_suffix, vessel_type in STRUCTURES:
        structure_cfg = getattr(b.structures, structure_key)
        if not structure_cfg.enabled:
            continue

        surface = os.path.join(surfaces_dir, f"{structure_cfg.segmentation_name}.{cfg.pipeline.convert.format}")
        output_dir = os.path.join(results_dir, f"final_export_{output_suffix}")
        logger.info("--- %s -> %s ---", structure_key.capitalize(), output_dir)

        # input_surface/lobe_surfaces_dir are given explicitly (the
        # well-known naming convention for this patient's surfaces_dir),
        # regardless of whether pipeline.stages.convert is enabled - if it
        # is, run_pipeline()'s stage 0 converts into surfaces_dir first
        # (segmentation_name set per structure so it targets the right
        # file); if not, surfaces_dir is assumed already populated from a
        # previous run, same as before.
        cfg.pipeline.convert.output_dir = surfaces_dir
        cfg.pipeline.convert.segmentation_name = structure_cfg.segmentation_name

        try:
            rc = run_pipeline(cfg, conda_env=cfg.conda_env, input_surface=surface, output_dir=output_dir,
                               vessel_type=vessel_type, lobe_surfaces_dir=surfaces_dir,
                               segmentation_dir=segmentation_dir)
            if rc != 0:
                logger.error("run_pipeline failed for %s (exit code %d)", structure_key, rc)
                ok = False
        except Exception:
            logger.exception("run_pipeline raised an exception for %s", structure_key)
            ok = False

    return ok


def run_single(cfg: configmod.RootConfig) -> int:
    """batch.enabled=false: batch.patients_root_dir/surfaces_root_dir/
    results_root_dir are this one patient's own folders directly (no
    per-patient subfolder). Logs to both the terminal and its own
    results_root_dir/pipeline.log (appended to, like batch mode's own -
    see logging_config.setup_console_and_file_logging())."""
    b = cfg.batch
    if not os.path.isdir(b.patients_root_dir):
        logger.error("segmentation folder not found: %s", b.patients_root_dir)
        return 1

    os.makedirs(b.results_root_dir, exist_ok=True)
    log_path = os.path.join(b.results_root_dir, "pipeline.log")
    log_file = setup_console_and_file_logging(log_path)
    try:
        logger.info("%s\n# Run started: %s\n%s", "#" * 60, time.strftime("%Y-%m-%d %H:%M:%S"), "#" * 60)
        ok = process_patient(b.patients_root_dir, b.surfaces_root_dir, b.results_root_dir, cfg)
        logger.info("=== Pipeline complete ===" if ok else "=== Pipeline finished with errors ===")
    finally:
        # setup_console_and_file_logging()'s background pump threads (see
        # logging_config._tee_native_output_to_log()) relay native VTK/VMTK
        # output into log_file from a different thread than this one -
        # closing log_file here while one of them is mid-write is a real,
        # unsynchronized race (logging's own per-handler lock protects
        # concurrent emit() calls against each other, but not against a
        # bare file.close() happening outside of it). By this point nothing
        # further should be writing (process_patient() already returned,
        # so any native output it triggered is long since finished, not
        # actually running concurrently) - this just gives the pump
        # threads a moment to drain whatever's still sitting in the pipe's
        # kernel buffer before the file goes away.
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.2)
        log_file.close()

    return 0 if ok else 1


def _process_patient_worker(patient_id: str, cfg: configmod.RootConfig) -> Tuple[str, bool, str]:
    """Runs entirely inside its own worker process (see run_batch() below).
    The very first thing it does is redirect this process' own logging AND
    OS-level stdout/stderr to this patient's own pipeline.log (see
    logging_config.setup_file_logging() - this also catches native VTK/VMTK
    output that bare Python-level redirection would miss), appended to (not
    truncated) so re-running the batch keeps prior history.
    Returns (patient_id, ok, log_path)."""
    b = cfg.batch
    segmentation_dir = os.path.join(b.patients_root_dir, patient_id)
    surfaces_dir = os.path.join(b.surfaces_root_dir, patient_id)
    results_dir = os.path.join(b.results_root_dir, patient_id)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "pipeline.log")

    log_file = setup_file_logging(log_path)
    try:
        logger.info("%s\n# Run started: %s\n%s", "#" * 60, time.strftime("%Y-%m-%d %H:%M:%S"), "#" * 60)
        logger.info("%s\n# Patient: %s\n%s", "#" * 60, patient_id, "#" * 60)
        ok = process_patient(segmentation_dir, surfaces_dir, results_dir, cfg)
    finally:
        log_file.close()

    return patient_id, ok, log_path


def _render_progress_bar(done, total, width=30):
    """'[#####-----] 5/20 (25%)' - a plain-ASCII progress bar, no extra
    dependency (tqdm et al.) needed. Deliberately bypasses `logging`
    entirely (see its only caller, run_batch()) - this is live, in-place
    terminal UI feedback (redrawn with '\\r', no trailing newline), not a
    log record; mixing it with logging's own newline-terminated,
    timestamped lines would either garble the bar or spam a timestamped
    line per redraw."""
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    pct = (done / total * 100) if total else 100
    return f"[{bar}] {done}/{total} ({pct:.0f}%)"


def run_batch(cfg: configmod.RootConfig) -> int:
    """batch.enabled=true: batch.patients_root_dir has one subfolder per
    patient; up to batch.parallel_jobs are processed concurrently."""
    b = cfg.batch
    if not os.path.isdir(b.patients_root_dir):
        logger.error("patients folder not found: %s", b.patients_root_dir)
        return 1

    patient_ids = sorted(
        entry for entry in os.listdir(b.patients_root_dir)
        if os.path.isdir(os.path.join(b.patients_root_dir, entry))
    )
    if not patient_ids:
        logger.error("no patient subfolders found in %s", b.patients_root_dir)
        return 1

    logger.info("Found %d patient(s) in %s - up to %d concurrent.",
                len(patient_ids), b.patients_root_dir, b.parallel_jobs)

    ok_patients: List[str] = []
    failed_patients: List[str] = []
    total = len(patient_ids)
    done = 0
    # Only draw the redrawing bar on a real terminal - on a redirected/piped
    # stdout (e.g. `> run.log`), a stream of bare '\r's just clutters the
    # file with no benefit, since nothing is there to overdraw the previous
    # line for.
    show_progress = sys.stdout.isatty()

    if show_progress:
        sys.stdout.write(_render_progress_bar(done, total))
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=b.parallel_jobs) as executor:
        futures = {executor.submit(_process_patient_worker, patient_id, cfg): patient_id
                   for patient_id in patient_ids}
        for future in as_completed(futures):
            patient_id = futures[future]
            if show_progress:
                # Erase the current bar line before letting logger print this
                # patient's own (real, newline-terminated) status line, then
                # redraw the bar on its own fresh line below.
                sys.stdout.write("\r" + " " * 60 + "\r")

            try:
                _patient_id, ok, log_path = future.result()
            except Exception:
                logger.exception("Patient %s FAILED (worker crashed)", patient_id)
                failed_patients.append(patient_id)
            else:
                if ok:
                    logger.info("Patient %s OK (log: %s)", patient_id, log_path)
                    ok_patients.append(patient_id)
                else:
                    logger.error("Patient %s FAILED (see %s)", patient_id, log_path)
                    failed_patients.append(patient_id)

            done += 1
            if show_progress:
                sys.stdout.write(_render_progress_bar(done, total))
                sys.stdout.flush()

    if show_progress:
        sys.stdout.write("\n")

    logger.info("=== Batch complete ===")
    logger.info("Succeeded: %d", len(ok_patients))
    logger.info("Failed:    %d", len(failed_patients))
    if failed_patients:
        for patient_id in failed_patients:
            logger.info("  %s", patient_id)

    return 1 if failed_patients else 0


def run(cfg: configmod.RootConfig) -> int:
    """The single function main.py calls - dispatches to run_single() or
    run_batch() based on cfg.batch.enabled."""
    if cfg.batch is None:
        logger.error("config has no 'batch' section - it must be present (with 'enabled': true/false) "
                      "even for a single patient, since main.py takes no other input. See "
                      "configs/pipeline_config.example.json.")
        return 1

    return run_batch(cfg) if cfg.batch.enabled else run_single(cfg)