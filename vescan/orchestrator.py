"""Single-patient/single-structure pipeline orchestration: run_pipeline()
calls every enabled stage under vescan/stages/ in-process, in order,
for one input surface/structure.

Lives in the vescan package (not in the root main.py script) so
vescan.batch can import it directly, including from a
ProcessPoolExecutor worker process (one per patient in batch mode) - a
worker importing the root main.py script back would be fragile under
multiprocessing's "spawn" start method (the default on macOS), which needs
every submitted callable's module to be a plain importable module, not the
__main__ entry script.
"""

import logging
import os

from vescan import config as configmod
from vescan import paths as pathsmod

logger = logging.getLogger(__name__)

# What stage 10 paints onto the surfaces. AnatomicalSegmentPoint is per-POINT,
# not per-group: transfer_centerline_labels resolves it from the nearest
# centerline point rather than the whole cell (see its module docstring), so
# the surface shows each boundary where it actually falls. Measured on real
# data that this moves 3.6% of surface points, most of it extra-lobar vessel
# the per-group array had swallowed into lobar territory (right pulmonary
# artery 694 -> 2201 points, left 1190 -> 2037).
# CrossingId/CrossingType ride along with AnatomicalSegment rather than in
# their own pass: they describe the SAME cells, so transferring them together
# keeps a surface cell's segment and its crossing kind decided by one nearest
# centerline point instead of two independent lookups that can disagree at a
# boundary.
ANATOMICAL_SEGMENT_ARRAYS = ("AnatomicalSegment", "AnatomicalSegmentPoint",
                              "CrossingId", "CrossingIdPoint",
                              "CrossingType", "CrossingTypePoint")


# Upper bound for the CrossingId range the island cleanup protects. The most
# crossing vessels seen on any structure is 5, so this is two orders of
# magnitude of headroom; it exists only because protected_values takes value
# sets, not "everything nonzero".
MAX_PROTECTED_CROSSING_ID = 500


def _crossing_values_to_protect(vessel_type):
    """Label values the island cleanup must never absorb, per array.

    A vessel crossing a lobar boundary appears on the surface as a small patch
    of its own label surrounded by another - the same shape as the artifact
    that cleanup removes, but the real finding the pipeline exists to report.
    Measured on real data that those patches land squarely in the absorbed
    size range (109 surface points on one artery tree, 160/111/109/51 on a
    venous one), so without naming them explicitly the cleanup would erase
    exactly the thing being looked for."""
    from vescan.stages.anatomical_segments import CROSSING_LOBE_LABEL, crossing_values_for
    from vescan.crossings import (CROSSING_TYPE_INTERLOBAR, CROSSING_TYPE_TRANSLOBAR,
                                          CROSSING_TYPE_UNCLASSIFIED)
    # All three: the plain crossing label and both refined ones (73-78).
    crossingSegments = crossing_values_for(vessel_type)
    # Every nonzero CrossingType is a finding, and every CrossingId names one -
    # an interlobar vessel's patches are SMALL by nature (it only grazes the
    # neighbouring lobe), so they sit even deeper in the absorbed size range
    # than the crossing segment label does. Ids are protected as a range since
    # there is no fixed value list: one per crossing vessel found.
    crossingTypes = {CROSSING_TYPE_INTERLOBAR, CROSSING_TYPE_TRANSLOBAR, CROSSING_TYPE_UNCLASSIFIED}
    crossingIds = set(range(1, MAX_PROTECTED_CROSSING_ID + 1))
    return (
        # Stage 9: lobe_reachability's own crossing marker, and its per-point
        # "inside more than one lobe" counterpart.
        {"LobeLabel": {CROSSING_LOBE_LABEL}, "LobeLabelPoint": {CROSSING_LOBE_LABEL}},
        # Stage 10: this vessel type's own crossing label (9/18/28), plus the
        # crossing grouping/kind that rides along with it.
        {"AnatomicalSegment": crossingSegments, "AnatomicalSegmentPoint": crossingSegments,
         "CrossingType": crossingTypes, "CrossingTypePoint": crossingTypes,
         "CrossingId": crossingIds, "CrossingIdPoint": crossingIds},
    )

# Stage modules are imported lazily, right before first use (see run_pipeline()
# below) rather than at module load time: they pull in vtk/vmtk/pyfqmr, only
# available inside the CONDA_ENV - importing them eagerly here would make even
# `main.py --help` require that environment, when today only actually
# running a stage does.


def run_pipeline(cfg: configmod.RootConfig, conda_env: str, input_surface, output_dir: str,
                  vessel_type: str, lobe_surfaces_dir, segmentation_dir=None) -> int:
    """input_surface/lobe_surfaces_dir may be None here (unlike every other
    argument): stage 0, if enabled, can supply defaults for both from its own
    conversion output - see below. They're only defaulted to "Artery.vtk" /
    "Data/patient_1" afterwards, once stage 0 (if any) has had a chance to
    fill them in."""
    p = cfg.pipeline
    os.makedirs(output_dir, exist_ok=True)

    try:
        # === [0] Converting segmentations ===
        if p.stages.convert:
            seg_dir = segmentation_dir or p.convert.segmentation_dir
            if not seg_dir:
                logger.error("pipeline.stages.convert is enabled but no segmentation directory was "
                              "given (segmentation_dir argument or pipeline.convert.segmentation_dir).")
                return 1
            converted_dir = p.convert.output_dir or (os.path.normpath(seg_dir) + "_vtk")

            logger.info("=== [0] Converting segmentations ===")
            from vescan.stages import convert_segmentations as convert_stage
            # Only meaningful for the airway structure's own conversion turn
            # (see convert_segmentations.run()'s docstring on why this stage
            # otherwise runs blindly once per structure) - segmentation_name
            # is "lung_arteries"/"lung_veins" during the other two turns, so
            # gating on vessel_type keeps this a no-op then.
            includeAirwayWall = p.convert.include_airway_wall and vessel_type == "airway"
            convert_stage.run(
                seg_dir,
                output_dir=converted_dir,
                exclude_files=p.convert.exclude_files,
                method=p.convert.method,
                decimation_factor=p.convert.decimation,
                smoothing_factor=p.convert.smoothing,
                compute_normals=p.convert.compute_normals,
                surface_nets_internal_smoothing=p.convert.surfacenets_internal_smoothing,
                coordinate_space=p.convert.coordinate_space,
                file_format=p.convert.format,
                include_airway_wall=includeAirwayWall,
                airway_name=p.convert.segmentation_name if includeAirwayWall else None,
                airway_wall_name=(f"{p.convert.segmentation_name}{p.convert.airway_wall_suffix}"
                                   if includeAirwayWall and p.convert.segmentation_name else None),
            )

            if input_surface is None and p.convert.segmentation_name:
                input_surface = os.path.join(converted_dir, f"{p.convert.segmentation_name}.{p.convert.format}")
            if lobe_surfaces_dir is None:
                # The lobe surfaces (lung_*_lobe_*.vtk) live in the same
                # converted folder as the structure's own surface - both come
                # from the same per-patient segmentation directory.
                lobe_surfaces_dir = converted_dir
        else:
            logger.info("=== [0] Converting segmentations: SKIPPED (stages.convert=false) ===")
    except Exception as e:
        logger.error("segmentation conversion failed: %s", e)
        return 1

    input_surface = input_surface or "Artery.vtk"
    lobe_surfaces_dir = lobe_surfaces_dir or "Data/patient_1"

    if not os.path.isfile(input_surface):
        logger.error("input surface '%s' not found.", input_surface)
        return 1

    paths = pathsmod.PipelinePaths(output_dir)
    os.makedirs(paths.supporting_files_dir, exist_ok=True)
    lobe_surfaces = pathsmod.default_lobe_surfaces(lobe_surfaces_dir)
    # Needed by stages 10/11 (the fissure sheets) whether or not the
    # patient-level block below runs in THIS invocation - an earlier run for
    # another structure of the same patient may already have written them.
    lobe_paths = pathsmod.LobeOverviewPaths(lobe_surfaces_dir)

    # PATIENT-level, not per-structure - see build_lobe_segments.py's module
    # docstring. Lands next to the lobe surfaces themselves (shared across
    # all 3 structures' own run_pipeline() calls for this patient), and is
    # idempotent (skipped if already there) since it's invoked once per
    # structure just like stage 0. A failure here is logged and does NOT
    # abort this structure's own pipeline - it's a standalone overview file,
    # nothing below depends on it.
    if p.stages.build_lobe_segments:
        extract_fissures = p.build_lobe_segments.extract_fissures
        # The fissure file counts towards "already done" ONLY when fissures
        # are switched on, so a patient processed before fissure extraction
        # existed gets it added on the next run instead of being skipped
        # forever - while a run with them off isn't forced to redo the merge
        # every time just because lobe_fissures.vtk will never appear.
        already_done = os.path.isfile(lobe_paths.segments) and (
            os.path.isfile(lobe_paths.fissures) or not extract_fissures)
        if already_done:
            logger.info("Combined lobe segments already exist at %s - skipping.", lobe_paths.segments)
        else:
            try:
                from vescan.stages import build_lobe_segments as build_lobe_segments_stage
                enabled_lobes = p.build_lobe_segments.enabled_lobes
                # Scoped to this standalone overview only (see
                # BuildLobeSegmentsConfig.enabled_lobes) - stage 9/10 below
                # keep using the unfiltered `lobe_surfaces` dict.
                overview_lobe_surfaces = ({name: path for name, path in lobe_surfaces.items()
                                            if name in enabled_lobes}
                                           if enabled_lobes is not None else lobe_surfaces)
                logger.info("=== Building combined lobe segments%s ===",
                            " and interlobar fissures" if extract_fissures else "")
                build_lobe_segments_stage.run(
                    lobe_paths.segments, overview_lobe_surfaces,
                    coordinate_space=p.convert.coordinate_space,
                    legend_output_path=lobe_paths.segments_legend,
                    fissures_output_path=lobe_paths.fissures if extract_fissures else None,
                    extract_fissures_too=extract_fissures,
                    fissure_contact_tolerance=p.build_lobe_segments.fissure_contact_tolerance,
                    min_fissure_component_area=p.build_lobe_segments.min_fissure_component_area,
                    min_fissure_area=p.build_lobe_segments.min_fissure_area,
                )
            except Exception:
                logger.exception("Building combined lobe segments failed - continuing without it.")

    # build_graph (stage 5) always (over)writes paths.branch_tree with the
    # branch-split centerline filtered/enriched - see build_graph.py's
    # module docstring and the [5/11] block below.
    clip_input = paths.branch_tree

    # Set by stage 4 below when it also produces a branch-split centerline
    # for stage 5 to reuse (see the [5/11] block) - avoids running the same
    # ~15-20 minute vtkvmtkCenterlineBranchExtractor filter twice.
    centerline_split_path = None
    # Set by stage 9 below (polyData, legend_data, coordinate_space) so
    # stage 10 can classify anatomical segments directly in memory instead
    # of re-reading the JSON/VTK it just wrote (see the [10/11] block).
    lobe_reachability_result = None
    # Which label values the island cleanup must leave alone - see
    # _crossing_values_to_protect().
    reachability_protected, anatomical_protected = _crossing_values_to_protect(vessel_type)

    try:
        # === [1/11] Preprocessing ===
        if p.stages.preprocess:
            logger.info("=== [1/11] Preprocessing ===")
            from vescan.stages import preprocess as preprocess_stage
            preprocess_stage.run(
                input_surface, paths.preprocessed,
                target_points=p.preprocess.target_points,
                decimation_aggressiveness=p.preprocess.decimation_aggressiveness,
                subdivide=p.preprocess.subdivide,
                decimate_enabled=not p.preprocess.no_decimate,
                keep_largest_component=not p.preprocess.keep_all_components,
            )
        else:
            logger.info("=== [1/11] Preprocessing: SKIPPED (stages.preprocess=false) ===")

        # === [2/11] Auto-detecting endpoints ===
        if p.stages.endpoints:
            logger.info("=== [2/11] Auto-detecting endpoints ===")
            from vescan.stages import endpoints as endpoints_stage
            start_point = [float(v) for v in p.endpoints.start_point.split()] if p.endpoints.start_point else None
            endpoints_stage.run(
                paths.preprocessed, paths.endpoints,
                start_point=start_point,
                coordinate_system=p.endpoints.coordinate_system or None,
                preprocess=p.endpoints.preprocess,
                keep_largest_component=not p.endpoints.keep_all_components,
                flip_input_lps_to_ras=p.endpoints.flip_input_lps_to_ras,
                endpoint_distance_radius_factor=p.endpoints.endpoint_distance_radius_factor,
                min_healthy_network_bbox_coverage=p.endpoints.min_healthy_network_bbox_coverage,
                network_advancement_ratio=p.endpoints.network_advancement_ratio,
            )
        else:
            logger.info("=== [2/11] Auto-detecting endpoints: SKIPPED (stages.endpoints=false) ===")

        # === [3/11] Extracting network ===
        if p.stages.network:
            logger.info("=== [3/11] Extracting network ===")
            from vescan.stages import network as network_stage
            network_stage.run(
                paths.preprocessed, paths.endpoints, paths.network,
                output_curve_path=paths.network_curve,
                computeGeometry=not p.network.no_geometry,
                advancement_ratio=p.network.advancement_ratio,
            )
        else:
            logger.info("=== [3/11] Extracting network: SKIPPED (stages.network=false) ===")

        # === [4/11] Extracting centerline ===
        if p.stages.centerline:
            logger.info("=== [4/11] Extracting centerline ===")
            from vescan.stages import centerline as centerline_stage
            # Only worth pre-computing/saving the branch-split centerline here
            # if stage 5 will actually run AND won't ignore it anyway (it
            # ignores it when told to use some other cached split instead -
            # see use_cached_split below).
            want_split_from_stage4 = p.stages.build_graph and not p.build_graph.use_cached_split
            centerline_stage.run(
                paths.preprocessed, paths.endpoints, paths.centerline_debug,
                curve_sampling_distance=p.centerline.curve_sampling_distance,
                merged_output_path=paths.centerline_branches_raw,
                properties_csv_path=paths.branch_properties,
                centerline_curve_path=paths.branch_curves,
                voronoi_output_path=paths.voronoi if p.centerline.save_voronoi else None,
                centerline_curve_model_path=(
                    paths.branch_curves_model if p.centerline.save_centerline_curve_model else None),
                simplify_voronoi=p.centerline.simplify_voronoi,
                split_output_path=paths.branch_tree if want_split_from_stage4 else None,
                resample_before_split=p.centerline.resample_before_split,
                delaunay_tolerance=p.centerline.delaunay_tolerance,
                max_centerline_outside_fraction=p.centerline.max_centerline_outside_fraction,
                warn_centerline_outside_fraction=p.centerline.warn_centerline_outside_fraction,
            )
            if want_split_from_stage4:
                centerline_split_path = paths.branch_tree
        else:
            logger.info("=== [4/11] Extracting centerline: SKIPPED (stages.centerline=false) ===")

        # === [5/11] Building centerline graph ===
        if p.stages.build_graph:
            os.makedirs(paths.branch_models_dir, exist_ok=True)
            if p.build_graph.use_cached_split:
                build_graph_input = p.build_graph.cached_split_file or paths.branch_tree
            elif centerline_split_path:
                # Stage 4 already ran vtkvmtkCenterlineBranchExtractor and
                # saved its output above - reuse it instead of recomputing
                # the same ~15-20 minute filter from scratch here.
                build_graph_input = centerline_split_path
            else:
                # Reachable when stages.centerline=False (stage 4 skipped
                # this run) and use_cached_split=False - there's no raw
                # centerline to fall back to any more (centerline.py only
                # ever saves paths.centerline_debug as a diagnostic artifact on
                # extraction FAILURE - see its own run() docstring - so on a
                # prior successful run it was never written). Reusing an old
                # run without recomputing stage 4 means pointing
                # use_cached_split/cached_split_file at that run's own
                # 05_branch_tree.vtk instead.
                logger.error("stages.build_graph is enabled but stages.centerline is disabled and "
                             "build_graph.use_cached_split is false - there is no centerline input to build the "
                             "graph from. Enable stages.centerline, or set build_graph.use_cached_split=true "
                             "with cached_split_file pointing at a previous run's 05_branch_tree.vtk.")
                return 1

            logger.info("=== [5/11] Building centerline graph ===")
            from vescan.stages import build_graph as build_graph_stage
            build_graph_stage.run(
                build_graph_input, paths.branch_models_dir,
                curve_output_path=paths.branch_tree_curves,
                bifurcation_output_path=paths.bifurcation_points,
                graph_output_path=paths.branch_tree_topology,
                min_branch_length=p.build_graph.min_branch_length,
                # Always (over)written, even when build_graph_input is this
                # same path (stage 4 already saved the raw, unfiltered split
                # there) - what's written back here is filtered/enriched,
                # not a no-op copy (see build_graph.py's module docstring).
                split_output_path=paths.branch_tree,
                # Veins genuinely have more than one true co-equal root at a
                # venous confluence, unlike arteries/airways which have one
                # directional trunk - see build_graph.py's build_graph_data()
                # docstring for why a single virtual root node is only
                # meaningful (and only added) there.
                add_virtual_root=(vessel_type == "vein" and p.build_graph.add_virtual_root_for_veins),
                orphan_root_repair_tolerance=p.build_graph.orphan_root_repair_tolerance,
                virtual_root_min_length_fraction=p.build_graph.virtual_root_min_length_fraction,
            )
        else:
            logger.info("=== [5/11] Building centerline graph: SKIPPED (stages.build_graph=false) ===")

        # === [6/11] Cutting centerline graph ===
        if p.stages.cut_graph:
            logger.info("=== [6/11] Cutting centerline graph ===")
            from vescan.stages import cut_graph as cut_graph_stage
            cut_graph_stage.run(
                paths.branch_tree_topology, paths.branch_tree, paths.cut_centerline,
                p.cut_graph.max_generations,
                cut_points_output_path=paths.cut_points,
            )
        else:
            logger.info("=== [6/11] Cutting centerline graph: SKIPPED (stages.cut_graph=false) ===")

        # Read once, shared by stage 7 (labels the clipped surface directly -
        # see clip_vessel.py's module docstring on why there's no separate
        # labeled-clipped-surface stage/file any more) and stage 8 (labels
        # the full surface).
        branch_arrays = tuple(a.strip() for a in p.label_branches.arrays.split(",") if a.strip())

        # === [7/11] Clipping vessel surface ===
        if p.stages.clip_vessel:
            logger.info("=== [7/11] Clipping vessel surface ===")
            from vescan.stages import clip_vessel as clip_vessel_stage
            clip_vessel_stage.run(
                paths.preprocessed, clip_input, paths.cut_centerline, paths.clipped_surface,
                cap=p.clip_vessel.cap,
                add_flow_extensions=p.clip_vessel.add_flow_extensions,
                extension_length=p.clip_vessel.extension_length,
                extension_mode=p.clip_vessel.extension_mode,
                label_arrays=branch_arrays if p.stages.label_branches else (),
            )
        else:
            logger.info("=== [7/11] Clipping vessel surface: SKIPPED (stages.clip_vessel=false) ===")

        # === [8/11] Labeling surface branches ===
        if p.stages.label_branches:
            logger.info("=== [8/11] Labeling surface branches ===")
            from vescan.stages import transfer_centerline_labels as transfer_labels_stage
            transfer_labels_stage.run(paths.preprocessed, paths.branch_tree, paths.branch_labeled_surface,
                                       array_names=branch_arrays)
        else:
            logger.info("=== [8/11] Labeling surface branches: SKIPPED (stages.label_branches=false) ===")

        # === [9/11] Labeling lobe reachability ===
        if p.stages.lobe_reachability:
            # LobeLabelPoint is per-POINT, not per-group: transfer_centerline_labels
            # resolves it from the nearest centerline point instead of the whole
            # cell (see its module docstring), so the surface shows the exact
            # place a vessel enters a lobe rather than the nearest group edge.
            reachability_arrays = ["NumReachableLobes", "LobeMask", "LobeLabel", "LobeLabelPoint"] + \
                [f"Reaches_{name}" for name in lobe_surfaces]
            extract_lobe_paths = ({name: paths.lobe_centerline(name) for name in lobe_surfaces}
                                   if p.lobe_reachability.extract_lobe_centerlines else {})

            logger.info("=== [9/11] Labeling lobe reachability ===")
            from vescan.stages import lobe_reachability as lobe_reachability_stage
            lobe_polydata, _reachable, lobe_legend_data, lobe_coordinate_space = lobe_reachability_stage.run(
                paths.branch_tree_topology, paths.branch_tree, paths.lobe_reachability, lobe_surfaces,
                containment_threshold=p.lobe_reachability.containment_threshold,
                tolerance=p.lobe_reachability.containment_tolerance,
                legend_output_path=paths.lobe_reachability_legend,
                extract_lobe_paths=extract_lobe_paths,
            )
            lobe_reachability_result = (lobe_polydata, lobe_legend_data, lobe_coordinate_space)

            from vescan.stages import transfer_centerline_labels as transfer_labels_stage
            # Enriches whichever full surface already exists -
            # branch_labeled_surface.vtk from stage 8, or (if that stage
            # was skipped) the plain preprocessed surface - with the
            # reachability arrays IN PLACE, instead of writing yet another
            # near-duplicate full-surface file (see lobe_reachability.py's
            # module docstring).
            full_surface_input = paths.branch_labeled_surface if p.stages.label_branches else paths.preprocessed
            transfer_labels_stage.run(full_surface_input, paths.lobe_reachability,
                                       paths.branch_labeled_surface, array_names=tuple(reachability_arrays),
                                       drop_unlabeled=True, fill_islands=True,
                                       max_island_area_fraction=p.label_branches.max_island_area_fraction,
                                       protected_values=reachability_protected)
            # Keyed on the FILE existing, not on stage 7 having run in this
            # same invocation: re-running only the labeling stages over a
            # finished output folder is a normal thing to do (that is what
            # configs/relabel_*.json are for), and gating on the stage toggle
            # left the clipped surface silently one version behind every time.
            if os.path.isfile(paths.clipped_surface):
                # paths.clipped_surface already carries the branch labels
                # (see clip_vessel.py's module docstring) when stages.
                # label_branches is on - enriched here in place too, for the
                # same reason.
                if not p.stages.clip_vessel:
                    logger.info("  (enriching the existing %s, which this run did not produce)",
                                os.path.basename(paths.clipped_surface))
                transfer_labels_stage.run(paths.clipped_surface, paths.lobe_reachability,
                                           paths.clipped_surface, array_names=tuple(reachability_arrays),
                                           drop_unlabeled=True, fill_islands=True,
                                           max_island_area_fraction=p.label_branches.max_island_area_fraction,
                                           protected_values=reachability_protected)
            else:
                logger.info("  (skipping clipped-surface reachability labels: %s does not exist)",
                            paths.clipped_surface)
        else:
            logger.info("=== [9/11] Labeling lobe reachability: SKIPPED (stages.lobe_reachability=false) ===")

        # === [10/11] Labeling anatomical segments ===
        if p.stages.anatomical_segments:
            logger.info("=== [10/11] Labeling anatomical segments ===")
            from vescan.stages import anatomical_segments as anatomical_segments_stage
            if lobe_reachability_result is not None:
                # Stage 9 ran in this same invocation - reuse its in-memory
                # result instead of re-reading the JSON/VTK it just wrote.
                lobe_polydata, lobe_legend_data, lobe_coordinate_space = lobe_reachability_result
                anatomical_segments_stage.classify_and_save(
                    lobe_legend_data, lobe_polydata, paths.anatomical_segments, vessel_type,
                    legend_output_path=paths.anatomical_segments_legend,
                    coordinate_space=lobe_coordinate_space,
                    topology_path=paths.branch_tree_topology,
                    fissures_path=lobe_paths.fissures,
                    min_piercings_for_interlobar=p.anatomical_segments.min_piercings_for_interlobar,
                    interlobar_max_median_angle_deg=p.anatomical_segments.interlobar_max_median_angle_deg,
                )
            else:
                anatomical_segments_stage.run(
                    paths.lobe_reachability_legend, paths.lobe_reachability, paths.anatomical_segments,
                    vessel_type, legend_output_path=paths.anatomical_segments_legend,
                    topology_path=paths.branch_tree_topology,
                    fissures_path=lobe_paths.fissures,
                    min_piercings_for_interlobar=p.anatomical_segments.min_piercings_for_interlobar,
                    interlobar_max_median_angle_deg=p.anatomical_segments.interlobar_max_median_angle_deg,
                )
            from vescan.stages import transfer_centerline_labels as transfer_labels_stage
            # Same "enrich in place" philosophy as stage 9: whichever full
            # surface already exists (branch_labeled_surface.vtk from
            # stage 8/9, or the plain preprocessed surface if both were
            # skipped) gets the AnatomicalSegment array added on top and
            # written back to that same file - no separate anatomical-only
            # full-surface copy.
            full_surface_input = (paths.branch_labeled_surface
                                   if p.stages.label_branches or p.stages.lobe_reachability
                                   else paths.preprocessed)
            transfer_labels_stage.run(full_surface_input, paths.anatomical_segments,
                                       paths.branch_labeled_surface, array_names=ANATOMICAL_SEGMENT_ARRAYS,
                                       drop_unlabeled=True, fill_islands=True,
                                       max_island_area_fraction=p.label_branches.max_island_area_fraction,
                                       protected_values=anatomical_protected)
            # Same reasoning as the [9/11] block above: the file's existence,
            # not stage 7's toggle, decides.
            if os.path.isfile(paths.clipped_surface):
                # paths.clipped_surface already carries the branch/
                # reachability labels (stages 7/9) - enriched here in place
                # too, for the same reason.
                if not p.stages.clip_vessel:
                    logger.info("  (enriching the existing %s, which this run did not produce)",
                                os.path.basename(paths.clipped_surface))
                transfer_labels_stage.run(paths.clipped_surface, paths.anatomical_segments,
                                           paths.clipped_surface, array_names=ANATOMICAL_SEGMENT_ARRAYS,
                                           drop_unlabeled=True, fill_islands=True,
                                           max_island_area_fraction=p.label_branches.max_island_area_fraction,
                                           protected_values=anatomical_protected)
            else:
                logger.info("  (skipping clipped-surface anatomical labels: %s does not exist)",
                            paths.clipped_surface)
        else:
            logger.info("=== [10/11] Labeling anatomical segments: SKIPPED (stages.anatomical_segments=false) ===")

        # === [11/11] Extracting statistics ===
        # Reporting only: it MEASURES the stages above and is an input to
        # nothing, so unlike every other stage a failure here must not cost
        # the caller a finished run - caught and logged, same treatment as
        # the patient-level build_lobe_segments block above. It also reads
        # its inputs off disk rather than taking them in memory, on purpose:
        # that keeps `python -m vescan.stages.statistics OUTPUT_DIR`
        # able to re-report on any past run without re-running anything.
        if p.stages.statistics:
            try:
                logger.info("=== [11/11] Extracting statistics ===")
                from vescan.stages import statistics as statistics_stage
                statistics_stage.run(
                    output_dir, vessel_type,
                    centerline_path=paths.anatomical_segments,
                    anatomical_legend_path=paths.anatomical_segments_legend,
                    topology_path=paths.branch_tree_topology,
                    fissures_path=lobe_paths.fissures,
                    output_path=paths.statistics,
                    csv_output_path=paths.statistics_csv,
                    min_piercings_for_interlobar=p.anatomical_segments.min_piercings_for_interlobar,
                    interlobar_max_median_angle_deg=p.anatomical_segments.interlobar_max_median_angle_deg,
                )
            except Exception:
                logger.exception("Extracting statistics failed - the run's own outputs are unaffected.")
        else:
            logger.info("=== [11/11] Extracting statistics: SKIPPED (stages.statistics=false) ===")

    except ImportError as e:
        logger.error("%s (main.py must be run from within the '%s' conda env - every stage now "
                      "imports vtk/vmtk directly in-process).", e, conda_env)
        return 1
    except Exception:
        logger.exception("Pipeline stage failed for '%s' (output_dir=%s) - aborting this run.",
                          vessel_type, output_dir)
        return 1

    logger.info("=== Pipeline complete ===")
    return 0