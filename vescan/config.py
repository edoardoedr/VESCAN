"""Typed schema + loader for the experiment JSON config (see
configs/pipeline_config.example.json) - the only input main.py takes.

Parses the JSON straight into nested dataclasses, one per config section, so
a typo in a key name or a wrong value type is caught with a precise error
instead of being silently ignored.

Every field has a sensible built-in default, so a config file can be as
partial as it likes (a key can also be explicitly `null`, meaning "use the
default") - see RootConfig/PipelineConfig below for the exact defaults.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import typing
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

IGNORED_KEYS = {"experiment_name", "notes", "description"}


class ConfigError(SystemExit):
    """Raised for a malformed config value - a SystemExit subclass so an
    unhandled one still exits main.py with a clean one-line message instead
    of a traceback."""


# ---------------------------------------------------------------------------
# pipeline.* - one section per stage, mirroring vescan/stages/
# ---------------------------------------------------------------------------

@dataclass
class StagesToggle:
    convert: bool = False  # [0] segmentation -> surface conversion, off by default since most
                            # configs already point at a pre-exported surface - turn on to start
                            # from raw segmentations instead (see pipeline.convert below)
    build_lobe_segments: bool = True  # PATIENT-level (not per-vessel-type), idempotent - see
                                       # vescan.stages.build_lobe_segments' own module docstring
    preprocess: bool = True
    endpoints: bool = True
    network: bool = True
    centerline: bool = True
    build_graph: bool = True
    cut_graph: bool = True
    clip_vessel: bool = True
    label_branches: bool = True
    lobe_reachability: bool = True
    anatomical_segments: bool = True
    statistics: bool = True  # [11] reporting only - counts/measures crossing vessels into
                              # statistics.json/.csv; nothing downstream consumes it, and
                              # a failure here never aborts the run (see orchestrator.py)


@dataclass
class PreprocessConfig:
    target_points: float = 5000.0
    # 4.0 is the value validated against Slicer's own decimated export (see
    # vescan.stages.preprocess's module docstring) - an earlier version
    # of this pipeline defaulted this to 2.0 for no documented reason; fixed.
    decimation_aggressiveness: float = 4.0
    subdivide: bool = False
    no_decimate: bool = False
    keep_all_components: bool = False


@dataclass
class EndpointsConfig:
    start_point: Optional[str] = None  # "X Y Z"; None = auto
    coordinate_system: str = "LPS"
    preprocess: bool = False
    keep_all_components: bool = False
    flip_input_lps_to_ras: Optional[bool] = None  # None = auto-detect
    # Drop any detected endpoint farther from the input surface than this
    # many times the local vessel radius there - see
    # vescan.stages.endpoints._filter_endpoints_near_surface()'s
    # docstring for why (an off-surface endpoint fails far less gracefully
    # downstream, in vtkvmtkPolyDataCenterlines), and for why the allowance
    # scales with the radius instead of being a fixed distance in mm: an
    # endpoint is a medial-axis point, so it legitimately sits about one
    # radius away from the surface, and a fixed threshold ends up discarding
    # the endpoints at the WIDEST parts of the tree.
    endpoint_distance_radius_factor: float = 1.5
    # Minimum bounding-box-diagonal coverage (network's own diagonal over the
    # input surface's) for a network extraction to be trusted as healthy
    # rather than a degenerate/aborted traversal - see
    # vescan.stages.endpoints.extract_network_robust()'s docstring.
    min_healthy_network_bbox_coverage: float = 0.9
    # vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio parameter (same
    # default VMTK itself uses).
    network_advancement_ratio: float = 1.05


@dataclass
class NetworkConfig:
    no_geometry: bool = False
    # vtkvmtkPolyDataNetworkExtraction's own AdvancementRatio parameter (same
    # default VMTK itself uses) - see EndpointsConfig.network_advancement_ratio,
    # the same parameter as used by the (separate) network extraction there.
    advancement_ratio: float = 1.05


@dataclass
class CenterlineConfig:
    curve_sampling_distance: float = 1.0
    save_voronoi: bool = True
    save_centerline_curve_model: bool = False
    simplify_voronoi: bool = False
    # None (default) = off, matching Slicer's own ExtractCenterline.py (which
    # also leaves CenterlineResampling off) - the raw centerline stays an
    # exact match of Slicer's raw CenterlineModel. A number (mm) turns on
    # vtkvmtkPolyDataCenterlines' own internal resampling (its
    # ResampleCenterlines(), linear per-cell with radius interpolation) at
    # that step length BEFORE vtkvmtkCenterlineBranchExtractor runs, cutting
    # its points-per-cell - see vescan.stages.centerline's module
    # docstring for why that step's cost scales with point density. The raw
    # centerline then stops matching Slicer's raw output 1:1 - validate on a
    # few patients (branch/graph topology unchanged) before enabling
    # dataset-wide.
    resample_before_split: Optional[float] = None
    # Delaunay tessellation tolerance, as a fraction of the surface's
    # bounding-box diagonal, for the tessellation this stage builds itself
    # (see centerline.py's build_interior_tessellation()) - mirrors
    # vtkvmtkPolyDataCenterlines' own default.
    delaunay_tolerance: float = 0.001
    # Above this fraction of centerline points lying outside the surface,
    # extraction is treated as failed - see centerline.py's
    # fraction_outside_surface()'s docstring for the real failure mode this
    # guards against (inward-facing normals). Healthy runs measure 0.2-4.3%.
    max_centerline_outside_fraction: float = 0.25
    # Above this fraction (but below max_centerline_outside_fraction), a
    # warning is logged instead of failing.
    warn_centerline_outside_fraction: float = 0.10


@dataclass
class BuildGraphConfig:
    min_branch_length: float = 1.0
    use_cached_split: bool = False
    cached_split_file: Optional[str] = None
    # Veins genuinely have more than one true co-equal root at a venous
    # confluence, unlike arteries/airways (single directional trunk) - see
    # build_graph.py's build_graph_data() docstring. Only applied when
    # vessel_type=="vein" (orchestrator.py); irrelevant to other vessels.
    add_virtual_root_for_veins: bool = True
    # Geometric matching tolerance (mm) for splicing an orphan root back onto
    # its true upstream neighbor when VMTK's own topology lookup misses it -
    # see build_graph.py's repair_orphan_roots() docstring.
    orphan_root_repair_tolerance: float = 0.5
    # A root heading a subtree shorter than this fraction of the longest
    # root's subtree is a degenerate stub, not a co-equal inflow - only
    # matters when add_virtual_root_for_veins applies (see build_graph.py's
    # find_substantial_roots() docstring).
    virtual_root_min_length_fraction: float = 0.1


@dataclass
class CutGraphConfig:
    max_generations: int = 6


@dataclass
class ClipVesselConfig:
    cap: bool = True
    add_flow_extensions: bool = False
    extension_length: float = 1.0
    extension_mode: str = "boundarynormal"


@dataclass
class LabelBranchesConfig:
    arrays: str = "GroupId,IsBifurcation,Generation,CellId"
    # Largest label "island" (fraction of the surface's total area) that
    # stages 9/10's own in-place enrichment passes will absorb into its
    # surroundings - see transfer_centerline_labels.py's
    # fill_label_islands() docstring for how this was measured.
    max_island_area_fraction: float = 0.0025


@dataclass
class AnatomicalSegmentsConfig:
    """Crossing-vessel interlobar/translobar classification (vescan.crossings,
    used by stage 10) - only used when pipeline.stages.anatomical_segments is
    true. NOTE: crossings.py's own module docstring is explicit that these
    values were measured on real data (13 crossing vessels, 3 patients) with
    wide margins, not arbitrarily chosen - "30 deg is not a tuned number, it
    is the middle of a wide gap." Change only with real data to justify it."""
    # A crossing vessel with fewer fissure piercings than this is translobar
    # by construction (a single piercing means it went through once and
    # stayed).
    min_piercings_for_interlobar: int = 2
    # Above min_piercings_for_interlobar piercings, the crossing is
    # interlobar when the median incidence angle to the fissure is below
    # this (degrees).
    interlobar_max_median_angle_deg: float = 30.0


@dataclass
class LobeReachabilityConfig:
    containment_threshold: float = 0.1
    containment_tolerance: float = 1e-4
    extract_lobe_centerlines: bool = True


@dataclass
class BuildLobeSegmentsConfig:
    """PATIENT-level lobe overview + interlobar fissures
    (vescan.stages.build_lobe_segments.run) - only used when
    pipeline.stages.build_lobe_segments is true."""
    # Extract the 4 interlobar fissure sheets from the lobe surfaces on top
    # of merging them. Off means the merged mesh carries only the three
    # LobeSegment_* arrays and no lobe_fissures.vtk is written. Costs ~2.5
    # min per patient (8 point-to-surface distance passes over the
    # full-resolution lobe meshes), paid once, not once per structure.
    extract_fissures: bool = True
    # A cell is fissure when all 3 of its vertices are within this many mm of
    # the neighbouring lobe's surface. NOT a tuning knob - adjacent lobes
    # come from one label map so their surfaces are coincident there, putting
    # the selection on a plateau (12.3% of RUL's points at 0.5 mm vs 13.4% at
    # 2.0 mm on R01-091); this is a guard against sub-voxel jitter. Exposed
    # only for a dataset whose lobes were meshed separately.
    fissure_contact_tolerance: float = 1.0
    # Connected patches of a candidate fissure below this area (mm2) are
    # dropped as artifacts - real fissures come out as one component of
    # several thousand mm2 (see build_lobe_segments.py's module docstring).
    min_fissure_component_area: float = 50.0
    # Below this total contact area (mm2), a fissure is reported as absent
    # rather than extracted - guards against a degenerate segmentation where
    # two lobes merely graze each other.
    min_fissure_area: float = 100.0
    # Which of vescan.lobes.LOBE_ORDER go into the combined overview
    # (lobe_segments.vtk) and its derived fissures - None (default) = all 5.
    # Scoped to this standalone, patient-level overview ONLY: nothing else in
    # the pipeline depends on it (see orchestrator.py), so a subset here is
    # safe - lobe_reachability/anatomical_segments (stages 9/10) always use
    # all 5, since their lobe_label numbering is hard-coded to that order.
    enabled_lobes: Optional[List[str]] = None


@dataclass
class ConvertConfig:
    """[0] segmentation -> surface conversion (vescan.stages.convert_segmentations.run) -
    converts every segmentation volume directly inside segmentation_dir into
    <segmentation_dir>_vtk/ (or output_dir, if set), one surface per label.
    Only used when pipeline.stages.convert is true."""
    segmentation_dir: Optional[str] = None  # folder of raw segmentation volumes to convert
    segmentation_name: Optional[str] = None  # stem (no extension) of the converted file that
                                              # becomes this run's input surface, e.g.
                                              # "lung_arteries" -> <output_dir>/lung_arteries.vtk -
                                              # ignored if orchestrator.run_pipeline()'s own
                                              # input_surface argument is already given
    output_dir: Optional[str] = None  # default: segmentation_dir + "_vtk"
    exclude_files: List[str] = field(default_factory=lambda: ["total_seg.nii.gz"])
    method: str = "flying-edges"  # "flying-edges" or "surface-nets"
    decimation: float = 0.0
    smoothing: float = 0.5
    compute_normals: bool = True
    surfacenets_internal_smoothing: bool = False
    coordinate_space: str = "LPS"
    format: str = "vtk"  # "vtk" or "vtp"
    # Airways only (orchestrator.run_pipeline() only honors this when
    # vessel_type=="airway" - see convert_segmentations.py's run()): merge
    # the airway lumen segmentation with a separate airway WALL segmentation
    # into one complete labelmap (union of both, single label) before
    # converting, instead of converting the lumen alone.
    include_airway_wall: bool = False
    airway_wall_suffix: str = "_wall"  # segmentation_name + this = the wall volume's own stem, e.g.
                                        # "lung_airways" + "_wall" -> "lung_airways_wall"


@dataclass
class PipelineConfig:
    stages: StagesToggle = field(default_factory=StagesToggle)
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    build_lobe_segments: BuildLobeSegmentsConfig = field(default_factory=BuildLobeSegmentsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    centerline: CenterlineConfig = field(default_factory=CenterlineConfig)
    build_graph: BuildGraphConfig = field(default_factory=BuildGraphConfig)
    cut_graph: CutGraphConfig = field(default_factory=CutGraphConfig)
    clip_vessel: ClipVesselConfig = field(default_factory=ClipVesselConfig)
    label_branches: LabelBranchesConfig = field(default_factory=LabelBranchesConfig)
    lobe_reachability: LobeReachabilityConfig = field(default_factory=LobeReachabilityConfig)
    anatomical_segments: AnatomicalSegmentsConfig = field(default_factory=AnatomicalSegmentsConfig)
    # [11] statistics has no tunables - it answers a fixed question (see
    # vescan.stages.statistics), so pipeline.stages.statistics is its
    # only switch.


# ---------------------------------------------------------------------------
# batch.* - drives main.py entirely: `enabled` is the switch between
# single-patient and multi-patient mode (see vescan/batch.py). The
# same three path fields are reused by both modes:
#   enabled=true  - patients_root_dir has one SUBFOLDER PER PATIENT (each
#                   with that patient's raw segmentation files);
#                   surfaces_root_dir/results_root_dir get one subfolder per
#                   patient too, created as needed.
#   enabled=false - patients_root_dir/surfaces_root_dir/results_root_dir are
#                   used DIRECTLY, as this one patient's own folders (no
#                   per-patient subfolder level) - main.py's whole CLI
#                   surface is just the config path, so this is also how a
#                   single-patient run now picks its input/output folders
#                   (no more --input/--output/--vessel-type flags).
# structures.{artery,veins,airways}.enabled decides which structure(s) to
# process either way - a single-patient run can process 1, 2 or all 3.
# ---------------------------------------------------------------------------

@dataclass
class StructureConfig:
    enabled: bool = True
    segmentation_name: str = ""


@dataclass
class BatchStructuresConfig:
    artery: StructureConfig = field(default_factory=lambda: StructureConfig(True, "lung_arteries"))
    veins: StructureConfig = field(default_factory=lambda: StructureConfig(True, "lung_veins"))
    airways: StructureConfig = field(default_factory=lambda: StructureConfig(True, "lung_airways"))


@dataclass
class BatchConfig:
    enabled: bool = False  # true = many patients (patients_root_dir has one subfolder per
                            # patient); false = a single patient (the three *_root_dir fields
                            # point directly at that patient's own folders)
    patients_root_dir: str = ""
    surfaces_root_dir: str = ""
    results_root_dir: str = ""
    parallel_jobs: int = 3
    # Segmentation -> surface conversion is controlled ONLY by
    # pipeline.stages.convert/pipeline.convert.* (same knob a single
    # orchestrator.run_pipeline() call uses) - there used to be a separate
    # batch.run_convert_segmentations/exclude_segmentation_files pair here,
    # which just duplicated that same decision at a different layer. Set
    # pipeline.stages.convert=true and pipeline.convert.exclude_files to
    # control conversion for a batch run too.
    structures: BatchStructuresConfig = field(default_factory=BatchStructuresConfig)


@dataclass
class RootConfig:
    conda_env: str = "vescan"
    batch: Optional[BatchConfig] = None
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# Generic dataclass <- JSON loader (works for any dataclass tree above, no
# per-field boilerplate needed when a new field is added)
# ---------------------------------------------------------------------------

def _unwrap_optional(tp):
    """Optional[X] is Union[X, None] - returns (X, True) for that shape,
    (tp, False) otherwise (including plain X and other Unions)."""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _known_paths(cls, prefix: str = "") -> set:
    paths = set()
    for f in dataclasses.fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        real_type, _optional = _unwrap_optional(typing.get_type_hints(cls)[f.name])
        if dataclasses.is_dataclass(real_type):
            paths |= _known_paths(real_type, path)
        else:
            paths.add(path)
    return paths


def _leaf_paths(data, prefix: str = "") -> set:
    paths = set()
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths |= _leaf_paths(value, path)
            else:
                paths.add(path)
    return paths


def _build(cls, data, path: str = ""):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Config error: '{path or cls.__name__}' must be an object, got {data!r}")

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data or data[f.name] is None:
            continue  # missing or explicit null -> keep the field's own default
        value = data[f.name]
        field_path = f"{path}.{f.name}" if path else f.name
        real_type, _optional = _unwrap_optional(hints[f.name])

        if dataclasses.is_dataclass(real_type):
            kwargs[f.name] = _build(real_type, value, field_path)
        elif real_type is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"Config error: '{field_path}' must be true or false, got {value!r}")
            kwargs[f.name] = value
        elif real_type in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"Config error: '{field_path}' must be a number, got {value!r}")
            kwargs[f.name] = real_type(value)
        elif real_type is str:
            if not isinstance(value, str):
                raise ConfigError(f"Config error: '{field_path}' must be a string, got {value!r}")
            kwargs[f.name] = value
        elif typing.get_origin(real_type) is list:
            (elem_type,) = typing.get_args(real_type) or (str,)
            if not isinstance(value, list) or not all(isinstance(v, elem_type) for v in value):
                raise ConfigError(f"Config error: '{field_path}' must be a list of "
                                   f"{elem_type.__name__}, got {value!r}")
            kwargs[f.name] = list(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Optional[str]) -> RootConfig:
    """Loads and validates a pipeline_config.example.json-shaped file into a
    RootConfig. path=None returns all-defaults."""
    if path is None:
        return RootConfig()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    known = _known_paths(RootConfig)
    present = _leaf_paths(data)
    for unknown in sorted(present - known - IGNORED_KEYS):
        logger.warning("unknown config key '%s' in %s (ignored - check for typos).", unknown, path)

    return _build(RootConfig, data)