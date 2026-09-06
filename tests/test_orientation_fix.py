"""Unit tests for the GLB orientation-fix math (pure numpy, no ComfyUI/GPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimodo_retarget_glb import _shortest_arc_matrix  # noqa: E402


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
