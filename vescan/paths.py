"""Single source of truth for the derived output file paths of one pipeline
run (one prefix per stage, e.g. OUTPUT_DIR/01_preprocessed.vtk) - used by
both vescan.orchestrator (single patient/structure) and
vescan.batch (many), so the naming convention is defined exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from vescan.lobes import LOBE_FILENAME_BY_NAME


def default_lobe_surfaces(lobe_surfaces_dir: str) -> Dict[str, str]:
    """NAME -> path for the 5 standard lobe surfaces expected under
    lobe_surfaces_dir (a patient's converted-surfaces directory), in
    vescan.lobes.LOBE_ORDER order (LOBE_FILENAME_BY_NAME's own
    declaration order, asserted to match at import time)."""
    return {name: os.path.join(lobe_surfaces_dir, filename)
            for name, filename in LOBE_FILENAME_BY_NAME.items()}


@dataclass
class LobeOverviewPaths:
    """The PATIENT-level outputs of vescan.stages.build_lobe_segments.

    Separate from PipelinePaths because these are not per-structure: the 5
    lobe surfaces (and the fissures between them) are the same whichever of
    artery/vein/airway is being processed, so these land next to the lobe
    surfaces themselves rather than inside any one structure's own
    final_export_<structure>/ - see that stage's module docstring."""

    lobe_surfaces_dir: str

    segments: str = field(init=False)
    segments_legend: str = field(init=False)
    # The 4 interlobar fissure sheets, deduplicated to one copy each (the
    # merged `segments` mesh carries the same cells twice, once per lobe) -
    # this is the file to use as a geometric object.
    fissures: str = field(init=False)

    def __post_init__(self) -> None:
        d = self.lobe_surfaces_dir
        self.segments = os.path.join(d, "lobe_segments.vtk")
        self.segments_legend = os.path.join(d, "lobe_segments.json")
        self.fissures = os.path.join(d, "lobe_fissures.vtk")


@dataclass
class PipelinePaths:
    """All derived file paths for one OUTPUT_DIR, computed once in
    __post_init__ so callers just read attributes (paths.preprocessed, ...)
    instead of re-deriving the naming convention themselves."""

    output_dir: str

    # The 4 files important for visualization (see the "Visualizing results
    # in 3D Slicer" README section), plus the summary statistics, live
    # directly in output_dir, without a numeric prefix. Every other
    # intermediate/debug file lives under this subdirectory instead, keeping
    # output_dir itself uncluttered.
    supporting_files_dir: str = field(init=False)
    preprocessed: str = field(init=False)
    endpoints: str = field(init=False)
    network: str = field(init=False)
    network_curve: str = field(init=False)
    centerline_debug: str = field(init=False)
    voronoi: str = field(init=False)
    centerline_branches_raw: str = field(init=False)
    branch_properties: str = field(init=False)
    branch_curves: str = field(init=False)
    branch_curves_model: str = field(init=False)
    branch_models_dir: str = field(init=False)
    branch_tree: str = field(init=False)
    branch_tree_curves: str = field(init=False)
    bifurcation_points: str = field(init=False)
    branch_tree_topology: str = field(init=False)
    cut_centerline: str = field(init=False)
    cut_points: str = field(init=False)
    clipped_surface: str = field(init=False)
    branch_labeled_surface: str = field(init=False)
    lobe_reachability: str = field(init=False)
    lobe_reachability_legend: str = field(init=False)
    anatomical_segments: str = field(init=False)
    anatomical_segments_legend: str = field(init=False)
    statistics: str = field(init=False)
    statistics_csv: str = field(init=False)

    def __post_init__(self) -> None:
        d = self.output_dir
        s = self.supporting_files_dir = os.path.join(d, "supporting_files")
        self.preprocessed = os.path.join(s, "01_preprocessed.vtk")
        self.endpoints = os.path.join(s, "02_endpoints.mrk.json")
        self.network = os.path.join(s, "03_network.vtk")
        self.network_curve = os.path.join(s, "03_network_curve.mrk.json")
        # Only ever written if centerline extraction FAILS (0 points) - see
        # centerline.py's run() docstring - kept as a diagnostic artifact,
        # never a routine output, hence the name.
        self.centerline_debug = os.path.join(s, "04_centerline_FAILED_debug.vtk")
        self.voronoi = os.path.join(s, "04_voronoi.vtk")
        # Per-branch GroupIds/CenterlineIds/TractIds/Blanking/Radius, but can
        # still have overlapping/duplicated bookkeeping cells for the same
        # branch (vtkvmtkMergeCenterlines internals) - "raw" flags that;
        # branch_curves below is the clean, deduplicated view built from it.
        self.centerline_branches_raw = os.path.join(s, "04_centerline_branches_raw.vtk")
        self.branch_properties = os.path.join(s, "04_branch_properties.csv")
        self.branch_curves = os.path.join(s, "04_branch_curves.mrk.json")
        self.branch_curves_model = os.path.join(s, "04_branch_curves_model.vtk")
        self.branch_models_dir = os.path.join(s, "05_branch_models")
        # THE branch tree (see build_graph.py's module docstring): the
        # branch-split centerline, degenerate/short groups removed, with
        # both the raw GroupIds/CenterlineIds/TractIds/Blanking/Radius
        # arrays and the friendly GroupId/IsBifurcation/Generation/Length/
        # AverageRadius ones - replaces what used to be three separate,
        # largely-overlapping files (raw split, cleaned split, combined
        # model).
        self.branch_tree = os.path.join(s, "05_branch_tree.vtk")
        self.branch_tree_curves = os.path.join(s, "05_branch_tree_curves.mrk.json")
        self.bifurcation_points = os.path.join(s, "05_bifurcation_points.mrk.json")
        self.branch_tree_topology = os.path.join(s, "05_branch_tree_topology.json")
        # One of the 4 files kept directly under output_dir (unprefixed) -
        # important for visualization, see the README's "Visualizing results
        # in 3D Slicer" section.
        self.cut_centerline = os.path.join(d, "cut_centerline.vtk")
        self.cut_points = os.path.join(s, "06_cut_points.mrk.json")
        # Already includes GroupId/IsBifurcation/Generation/CellId (see
        # clip_vessel.py's module docstring) when stages.label_branches is
        # on - no separate labeled-clipped-surface file any more. Stages 9
        # and 10 further enrich this SAME file in place with reachability and
        # AnatomicalSegment arrays (see orchestrator.py's [9/11]/[10/11]
        # blocks) instead of spawning a separate labeled twin per stage. Kept
        # directly under output_dir (unprefixed) - important for
        # visualization.
        self.clipped_surface = os.path.join(d, "clipped_surface.vtk")
        # Enriched in place (not re-created) by stages 9 AND 10 - reachability
        # arrays (NumReachableLobes/LobeMask/LobeLabel/Reaches_<LOBE>), then
        # AnatomicalSegment - on top of whatever stage 8 already added - see
        # orchestrator.py's [9/11] and [10/11] blocks - so this stays the one
        # cumulative full-surface label file instead of spawning a twin per
        # stage. Named "branch_" (not just "labeled_") since GroupId/
        # IsBifurcation/Generation/CellId - branch identity - is what stage 8
        # puts here first. Kept directly under output_dir (unprefixed) -
        # important for visualization.
        self.branch_labeled_surface = os.path.join(d, "branch_labeled_surface.vtk")
        self.lobe_reachability = os.path.join(s, "09_lobe_reachability.vtk")
        self.lobe_reachability_legend = os.path.join(s, "09_lobe_reachability.json")
        # Kept directly under output_dir (unprefixed) - important for
        # visualization; its legend stays with the rest of the supporting
        # files since Slicer visualization uses the color table CSVs, not
        # this JSON.
        self.anatomical_segments = os.path.join(d, "anatomical_segments.vtk")
        self.anatomical_segments_legend = os.path.join(s, "10_anatomical_segments.json")
        # Reporting only - measured FROM the files above, never an input to
        # any other stage (see statistics.py's module docstring), so nothing
        # downstream breaks if stage 11 is off or its output is deleted. The
        # CSV carries the same numbers as the JSON in the flat shape that
        # survives being concatenated across patients into a spreadsheet.
        # Kept directly under output_dir (unprefixed), like the 4 files
        # above, since this is a primary result, not an intermediate one.
        self.statistics = os.path.join(d, "statistics.json")
        self.statistics_csv = os.path.join(d, "statistics.csv")

    def lobe_centerline(self, lobe_name: str) -> str:
        return os.path.join(self.supporting_files_dir, f"09_centerline_{lobe_name}.vtk")