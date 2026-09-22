"""Canonical pulmonary lobe order and side membership - the single source of
truth every stage dealing with lobes imports from, instead of each keeping
its own copy.

ORDER MATTERS: this exact sequence becomes vescan.stages.
lobe_reachability's LobeMask bit order AND lobe_label numbering (see that
module's compute_lobe_labels()) - vescan.stages.anatomical_segments
decodes lobe_label back into a lobe name using THIS SAME order. If the order
lobe surfaces are actually passed to lobe_reachability.run() (built from
LOBE_FILENAME_BY_NAME below, via paths.default_lobe_surfaces()) ever
disagreed with anatomical_segments' own idea of the order, anatomical_segments
would silently decode the wrong lobe for every group - previously a real risk
since this order used to be duplicated independently in paths.py and
label_anatomical_segments.py; now defined exactly once, here.
"""

LOBE_ORDER = ["RUL", "RML", "RLL", "LLL", "LUL"]

RIGHT_LOBES = {"RUL", "RML", "RLL"}
LEFT_LOBES = {"LUL", "LLL"}

LOBE_FILENAME_BY_NAME = {
    "RUL": "lung_upper_lobe_right.vtk",
    "RML": "lung_middle_lobe_right.vtk",
    "RLL": "lung_lower_lobe_right.vtk",
    "LLL": "lung_lower_lobe_left.vtk",
    "LUL": "lung_upper_lobe_left.vtk",
}

assert list(LOBE_FILENAME_BY_NAME) == LOBE_ORDER, "LOBE_FILENAME_BY_NAME must be declared in LOBE_ORDER order"
assert RIGHT_LOBES | LEFT_LOBES == set(LOBE_ORDER), "RIGHT_LOBES/LEFT_LOBES must partition LOBE_ORDER"