"""Unit tests for the GLB orientation-fix math (pure numpy, no ComfyUI/GPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimodo_retarget_glb import _shortest_arc_matrix, _snap_to_axis  # noqa: E402


def test_snap_to_axis_picks_dominant_signed_axis():
    import numpy as np
    cases = {
        (-0.017, -0.175, 0.984): (0, 0, 1),   # tipped forward + walk lean -> +Z
        (0.02, 0.99, -0.05): (0, 1, 0),        # already up
        (-0.9, 0.1, 0.2): (-1, 0, 0),
        (0.05, -0.98, 0.1): (0, -1, 0),
    }
    for v, exp in cases.items():
        assert tuple(_snap_to_axis(np.array(v))) == exp, (v, _snap_to_axis(np.array(v)))


def test_snapped_correction_is_clean_90_multiple():
    """After snapping, the correction angle is exactly 0/90/180°, never a lean."""
    import numpy as np
    for v in ([-0.017, -0.175, 0.984], [0.3, 0.1, -0.95], [-0.88, 0.2, 0.4]):
        a = _snap_to_axis(np.array(v))
        R = _shortest_arc_matrix(a, np.array([0.0, 1.0, 0.0]))
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert min(abs(ang - k) for k in (0, 90, 180)) < 1e-6, (v, ang)


def test_shortest_arc_identity_when_aligned():
    R = _shortest_arc_matrix(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_shortest_arc_maps_a_onto_b():
    for a in ([0, 0, 1], [1, 0, 0], [0.2, -0.9, 0.3], [-0.1, -0.2, 0.98]):
        a = np.array(a, dtype=float)
        R = _shortest_arc_matrix(a, np.array([0.0, 1.0, 0.0]))
        got = R @ (a / np.linalg.norm(a))
        assert np.allclose(got, [0, 1, 0], atol=1e-6), (a, got)


def test_shortest_arc_is_pure_tilt_no_yaw():
    """A +Z up (tipped forward) must be corrected by a rotation about the X axis
    only (no spin about Y), so the character's facing is preserved."""
    R = _shortest_arc_matrix(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    # A pure X-axis rotation leaves the X basis vector unchanged.
    assert np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-6), R


def test_shortest_arc_antiparallel():
    R = _shortest_arc_matrix(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    got = R @ [0, -1, 0]
    assert np.allclose(got, [0, 1, 0], atol=1e-6), got
