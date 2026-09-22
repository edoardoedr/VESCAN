"""Central logging configuration for the whole pipeline.

Every module gets its own logger via `logging.getLogger(__name__)` and relies
on one of the two setup functions below having already configured the root
logger's handler/formatting - no module should configure logging itself or
fall back to bare print() for anything a human is meant to read later.

Three setups, one per main.py entry path (see vescan.batch.run()):
  - setup_console_logging() - the initial, always-on setup (before batch.
    enabled is even known - e.g. errors while loading the config itself):
    output goes straight to the terminal, like any normal command.
  - setup_console_and_file_logging() - a single patient (batch.enabled=
    false): "tees" to both the terminal (this process IS a real, attended
    terminal session) AND that patient's own pipeline.log, so a single-
    patient run also leaves a persistent record, same as batch mode's -
    including native VTK/VMTK C++ output, via a background pipe reader
    (see _tee_native_output_to_log()), not just whatever went through
    Python's own `logging` module.
  - setup_file_logging() - one batch worker process per patient
    (batch.enabled=true): output goes ONLY to that patient's own
    pipeline.log (no attended terminal to also print to - see its own
    docstring for why this one redirects the OS-level fds directly instead
    of also tee-ing to a console).
"""

import logging
import os
import sys
import threading

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_console_logging(level=logging.INFO):
    """Configures the root logger to print to the terminal (stdout)."""
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.handlers = [handler]


def _tee_native_output_to_log(level=logging.INFO):
    """Redirects the process' OS-level stdout/stderr file descriptors (1, 2)
    through a pipe each, so native C/C++ output written directly to them -
    bypassing Python's sys.stdout/sys.stderr entirely, e.g. VTK/VMTK's own
    vtkOutputWindow (see setup_file_logging()'s own docstring for why that
    matters - the exact same gap applies here) - reaches every configured
    `logging` handler too, not just whichever raw stream it happened to be
    attached to. A background daemon thread per fd reads everything written
    to it and re-emits it, line by line, through logging.getLogger("native")
    - which naturally fans out to every handler already on the root logger
    (console AND file, in setup_console_and_file_logging()'s case),
    formatted the same as every other log line, and reuses logging's own
    thread-safe handler locking instead of writing to a shared file object
    from two threads unsynchronized.

    MUST be called AFTER the root logger's own handlers are attached to a
    PRESERVED duplicate of the original stdout/stderr (not sys.stdout/
    sys.stderr themselves) - otherwise those handlers' own writes would
    loop straight back through the very pipes being set up here."""
    nativeLogger = logging.getLogger("native")

    for fd in (sys.stdout.fileno(), sys.stderr.fileno()):
        readFd, writeFd = os.pipe()
        os.dup2(writeFd, fd)
        os.close(writeFd)

        def _pump(readFd=readFd):
            with os.fdopen(readFd, "r", buffering=1, errors="replace") as reader:
                for line in reader:
                    nativeLogger.log(level, line.rstrip("\n"))

        threading.Thread(target=_pump, daemon=True).start()


def setup_console_and_file_logging(log_path, level=logging.INFO):
    """Configures the root logger to log to BOTH the terminal and log_path
    (append mode - a patient can be re-run without losing earlier history,
    same convention batch mode uses) - including native VTK/VMTK C++
    output (e.g. vtkOutputWindow's ERR/WARN lines), which bypasses Python's
    `logging` module/sys.stdout entirely and would otherwise only ever
    reach the terminal, never this file (see _tee_native_output_to_log()).

    Returns the opened log file object - the caller should keep a reference
    to it for the lifetime of the run and close it when done."""
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    # A duplicate of the terminal's REAL stdout, taken before fd 1/2 get
    # redirected below - the console handler writes here instead of to
    # sys.stdout directly, so its own output doesn't loop back through the
    # very pipes _tee_native_output_to_log() is about to set up.
    consoleStream = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(consoleStream)
    console_handler.setFormatter(formatter)
    file_handler = logging.StreamHandler(log_file)
    file_handler.setFormatter(formatter)
    root.handlers = [console_handler, file_handler]

    _tee_native_output_to_log(level=level)

    return log_file


def setup_file_logging(log_path, level=logging.INFO):
    """Configures the root logger to write to log_path (append mode - a
    patient can be re-run without losing earlier history, same convention
    the old bash orchestrator used), AND redirects this process' own OS-level
    stdout/stderr file descriptors to the same file.

    The dup2() part is the fix for a real gap the previous
    contextlib.redirect_stdout()-based approach had: that only ever patches
    the *Python* sys.stdout/sys.stderr objects, so any native VTK/VMTK C++
    output - e.g. vtkOutputWindow's default vtkErrorMacro/vtkWarningMacro
    sink, which writes straight to the process' real stderr file descriptor,
    bypassing Python entirely - would leak onto whatever terminal originally
    launched the batch instead of landing in this patient's own log,
    breaking the "one log file per patient" guarantee. Since this function
    is meant to be called once, near the top of a batch worker process (a
    genuinely separate OS process, one per patient - see
    vescan.batch._process_patient_worker()), redirecting that whole
    process' fd 1/2 is safe: it can't affect any other patient's worker or
    the main process.

    Returns the opened log file object - the caller should keep a reference
    to it for the lifetime of the redirection (closing it invalidates the
    dup2'd descriptors)."""
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.handlers = [handler]

    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

    return log_file