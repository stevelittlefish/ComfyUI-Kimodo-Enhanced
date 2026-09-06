"""Rest-direction-aware retarget tests (the A-pose fix).

These are pure-numpy: they build tiny SkeletonData skeletons by hand and drive
``retarget_animation`` directly, so they run without the FBX SDK, ComfyUI, or a
GPU. Run from the repo root:

    python -m pytest tests/ -q

They pin the three invariants of the rest-direction fix:
  1. legacy equivalence — a T-pose source onto a T-pose target reproduces the
     old rotation-only formula exactly (no regression to the working path);
  2. rest preserved — at the rest frame the target holds its own rest pose
     (this is what keeps an A-pose arm from snapping to horizontal);
  3. motion magnitude preserved — the geodesic angle the target swings equals
     the source's (conjugation by the direction correction is angle-preserving),
     which is precisely what the pre-fix code got wrong for A-pose rigs.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimodo_retarget_fbx import (  # noqa: E402
    BoneData,
    SkeletonData,
    SOMA_TO_MIXAMO,
    retarget_animation,
    _body_only_mapping,
    _quat_mul,
    _quat_inv,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _q_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_angle(q):
    """Geodesic rotation angle (radians) of a wxyz quaternion."""
    w = min(1.0, abs(float(q[0])))
    return 2.0 * np.arccos(w)


def _bone(name, head, parent=None, rest_rot=IDENT):
    b = BoneData(name)
    b.parent_name = parent
    b.head = np.asarray(head, dtype=float)
    b.rest_rotation = np.asarray(rest_rot, dtype=float)
    b.world_matrix = np.eye(4)
    b.world_matrix[3, :3] = b.head
    return b


def _skel(name, bones):
    s = SkeletonData(name)
    for b in bones:
        s.add_bone(b)
        s.all_nodes[b.name] = b.name
        s.node_rest_rotations[b.name] = b.rest_rotation
    s.frame_start = 0
    s.frame_end = 0
    return s


def _source_tpose():
    """SOMA-like T-pose: left arm points +X (horizontal), forearm continues +X."""
    hips = _bone("hips", [0, 0, 0])
    arm = _bone("leftarm", [0.1, 1.5, 0], parent="hips")
    fore = _bone("leftforearm", [0.4, 1.5, 0], parent="leftarm")
    return _skel("soma", [hips, arm, fore])


def _target_apose(down_deg):
    """Mixamo-like target whose left arm rests ``down_deg`` below horizontal."""
    t = np.radians(down_deg)
    d = np.array([np.cos(t), -np.sin(t), 0.0])  # +X rotated down toward -Y
    hips = _bone("mixamorig:hips", [0, 0, 0])
    arm = _bone("mixamorig:leftarm", [0.1, 1.5, 0], parent="mixamorig:hips")
    fore = _bone("mixamorig:leftforearm", arm.head + 0.3 * d, parent="mixamorig:leftarm")
    return _skel("mixamo", [hips, arm, fore])


MAPPING = {
    "hips": "mixamorig:hips",
    "leftarm": "mixamorig:leftarm",
    "leftforearm": "mixamorig:leftforearm",
}


def _apply_source_motion(src, per_bone_world_quats):
    """Attach a 2-frame world animation: frame 0 = rest, frame 1 = posed."""
    src.frame_start = 0
    src.frame_end = 1
    for name, q1 in per_bone_world_quats.items():
        b = src.get_bone(name)
        b.world_animation[0] = b.rest_rotation.copy()
        b.world_animation[1] = q1
        b.world_location_animation[0] = b.world_matrix[3, :3].copy()
        b.world_location_animation[1] = b.world_matrix[3, :3].copy()


def test_tpose_to_tpose_matches_legacy():
    """Both skeletons T-pose -> output equals the old ``s_rot * off`` formula."""
    src = _source_tpose()
    tgt = _target_apose(down_deg=0.0)  # arm horizontal == T-pose

    lift = _q_xyzw_to_wxyz(R.from_euler("z", 40, degrees=True).as_quat())
    _apply_source_motion(src, {"leftarm": lift})

    ret_rots, _ = retarget_animation(src, tgt, MAPPING)

    # Legacy: off = inv(s_rest) * t_rest (identities here), world = s_rot * off,
    # local = inv(parent_world) * world. Parent (hips) is unanimated -> identity.
    s_arm = src.get_bone("leftarm")
    t_arm = tgt.get_bone("leftarm")
    off = _quat_mul(_quat_inv(s_arm.rest_rotation), t_arm.rest_rotation)
    legacy_world = _quat_mul(lift, off)

    got = ret_rots["mixamorig:leftarm"][1]
    # hips parent is identity across frames, so local == world here
    assert np.allclose(got, legacy_world, atol=1e-6), (got, legacy_world)


def test_auto_fix_input_pose_off_matches_legacy_on_apose():
    """The toggle OFF reproduces the legacy rotation-only output even on A-pose."""
    src = _source_tpose()
    tgt = _target_apose(down_deg=65.0)

    lift = _q_xyzw_to_wxyz(R.from_euler("z", 40, degrees=True).as_quat())
    _apply_source_motion(src, {"leftarm": lift})

    ret_rots, _ = retarget_animation(src, tgt, MAPPING, auto_fix_input_pose=False)

    s_arm = src.get_bone("leftarm")
    t_arm = tgt.get_bone("leftarm")
    off = _quat_mul(_quat_inv(s_arm.rest_rotation), t_arm.rest_rotation)
    legacy_world = _quat_mul(lift, off)  # hips parent identity -> local == world
    got = ret_rots["mixamorig:leftarm"][1]
    assert np.allclose(got, legacy_world, atol=1e-6), (got, legacy_world)


def test_body_only_mapping_drops_fingers_keeps_body():
    """map_fingers=False must strip every finger bone but keep the body + hand root.

    Finger direction-matching bends SOMA's fingers into a claw on most rigs, so
    the nodes default to this body-only mapping."""
    full = SOMA_TO_MIXAMO
    body = _body_only_mapping(full)
    assert len(body) < len(full)
    for s in body:
        assert not any(tok in s.lower()
                       for tok in ("thumb", "index", "middle", "ring", "pinky")), s
    # body joints survive
    for s in ("hips", "leftarm", "rightforearm", "lefthand", "leftfoot"):
        assert s in body, s


def _bone_dir(bone, child):
    d = child.head - bone.head
    return d / np.linalg.norm(d)


def _apply_wxyz(q, v):
    return R.from_quat([q[1], q[2], q[3], q[0]]).apply(v)


def test_apose_target_bone_points_where_source_points():
    """The core invariant: the target bone must point along the SOURCE bone's
    world direction each frame — regardless of the A-pose rest offset. This is
    what the crossed-arms bug violated (the target arm swung across the body)."""
    src = _source_tpose()
    tgt = _target_apose(down_deg=65.0)

    # Source arm swung DOWN and forward (like a real SOMA walk pose), not at the
    # T-pose horizontal rest.
    s_arm = src.get_bone("leftarm")
    s_fore = src.get_bone("leftforearm")
    s_dir = _bone_dir(s_arm, s_fore)  # +X (T-pose rest)
    swing = R.from_euler("z", -80, degrees=True)  # rotate +X down toward -Y
    v_s = swing.apply(s_dir)
    s_world = _q_xyzw_to_wxyz(swing.as_quat())  # world rotation taking s_dir -> v_s
    _apply_source_motion(src, {"leftarm": s_world})

    ret_rots, _ = retarget_animation(src, tgt, MAPPING, auto_fix_input_pose=True)

    # Target arm world rotation == its local channel (hips parent is identity).
    t_arm = tgt.get_bone("mixamorig:leftarm")
    t_fore = tgt.get_bone("mixamorig:leftforearm")
    t_dir = _bone_dir(t_arm, t_fore)
    v_t = _apply_wxyz(ret_rots["mixamorig:leftarm"][1], t_dir)

    assert np.allclose(v_t, v_s, atol=1e-5), (v_t, v_s)


def test_apose_source_down_keeps_target_down():
    """Regression guard for the crossed-arms bug: a source arm hanging straight
    down must leave the target arm hanging down (~its A-pose), NOT swung out to
    the side across the body."""
    src = _source_tpose()
    tgt = _target_apose(down_deg=65.0)

    s_arm = src.get_bone("leftarm")
    s_fore = src.get_bone("leftforearm")
    swing = R.from_euler("z", -90, degrees=True)  # +X -> straight down (-Y)
    s_world = _q_xyzw_to_wxyz(swing.as_quat())
    _apply_source_motion(src, {"leftarm": s_world})

    ret_rots, _ = retarget_animation(src, tgt, MAPPING, auto_fix_input_pose=True)

    t_arm = tgt.get_bone("mixamorig:leftarm")
    t_fore = tgt.get_bone("mixamorig:leftforearm")
    t_dir = _bone_dir(t_arm, t_fore)
    v_t = _apply_wxyz(ret_rots["mixamorig:leftarm"][1], t_dir)

    # Arm must be dominantly downward (-Y), not sideways (|x| small).
    assert v_t[1] < -0.9, v_t
    assert abs(v_t[0]) < 0.2, v_t
