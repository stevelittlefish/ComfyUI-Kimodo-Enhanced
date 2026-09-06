"""Kimodo GLB Retarget Module

Retargets Kimodo SOMA skeleton motion onto a rigged glTF/GLB character and
writes the animation back into a new .glb — the GLB twin of
``kimodo_retarget_fbx.py``.

The retargeting math is skeleton-agnostic, so we reuse ``SkeletonData``,
``BoneData``, ``retarget_animation``, ``kimodo_to_source_skeleton`` and the
``SOMA_TO_MIXAMO`` mapping from the FBX module. Only the two format-specific
ends differ:

  * reading the target skeleton — from glTF nodes/skin instead of the FBX SDK
    (adapted from ComfyUI-SkinTokens-NoBlender ``glb_io._skeleton_from_gltf``);
  * writing the animation — as glTF animation samplers/channels (TRS keyframes)
    instead of FBX anim curves.

glTF joint nodes are plain TRS: the animated *local* rotation channel fully
replaces a node's rest local rotation, and the retarget already produces exactly
that local-rotation-relative-to-parent, so the mapping is direct. Only the root
gets a translation channel (from ``ret_locs``); other joints keep their static
bind translation.
"""

from __future__ import annotations

import os
import shutil
import traceback

import numpy as np
from scipy.spatial.transform import Rotation as R

# Reuse the format-agnostic pieces from the FBX module.
from kimodo_retarget_fbx import (
    BoneData,
    SkeletonData,
    SOMA_TO_MIXAMO,
    _body_only_mapping,
    kimodo_to_source_skeleton,
    retarget_animation,
)

_TAG = "[Kimodo GLB]"


def _log(msg: str):
    print(f"{_TAG} {msg}", flush=True)


# glTF component types
_FLOAT = 5126
# glTF bufferView targets are not required for animation accessors.


# ============================================================================
# Load target GLB skeleton
# ============================================================================

def _node_local_matrix(node) -> np.ndarray:
    """Row-vector-free local matrix (column-major math, translation at [:3, 3])."""
    if node.matrix:
        # glTF matrices are column-major; reshape+T to standard row-major math.
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    if node.rotation:
        x, y, z, w = node.rotation
        m[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    if node.scale:
        m[:3, :3] = m[:3, :3] @ np.diag(node.scale)
    m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    return m


def load_target_glb(filepath: str):
    """Load a rigged glb/gltf and return (gltf, SkeletonData, node_of_joint).

    ``node_of_joint`` maps joint-order index -> glTF node index (for writing the
    animation channels back onto the right nodes). Bone rest data is stored in
    the FBX-style ``SkeletonData`` layout so ``retarget_animation`` consumes it
    unchanged: ``world_matrix``/``local_matrix`` carry translation at row [3, :3]
    and ``rest_rotation`` is a world-space quaternion [w, x, y, z].
    """
    from pygltflib import GLTF2

    _log("--- Loading target GLB ---")
    _log(f"  Path: {filepath}")
    _log(f"  Exists: {os.path.exists(filepath)}  Size: {os.path.getsize(filepath)} bytes")

    ext = os.path.splitext(filepath)[1].lower()
    g = GLTF2().load(filepath) if ext == ".gltf" else GLTF2().load_binary(filepath)

    if not g.skins:
        raise ValueError("glTF has no skin — this is not a rigged character.")

    joint_nodes = list(g.skins[0].joints)

    # parent map over ALL nodes
    parent_of = {}
    for i, node in enumerate(g.nodes):
        for c in (node.children or []):
            parent_of[c] = i

    _cache: dict[int, np.ndarray] = {}

    def world_matrix(i: int) -> np.ndarray:
        if i in _cache:
            return _cache[i]
        m = _node_local_matrix(g.nodes[i])
        if i in parent_of:
            m = world_matrix(parent_of[i]) @ m
        _cache[i] = m
        return m

    joint_set = set(joint_nodes)

    skel = SkeletonData(os.path.basename(filepath))
    skel.fps = 30.0  # overwritten by the source fps at write time
    node_of_joint: list[int] = []

    def _to_fbx_layout(mat_std: np.ndarray) -> np.ndarray:
        """Standard (M@v, translation at [:3,3]) -> FBX layout (translation at [3,:3])."""
        out = np.eye(4)
        out[:3, :3] = mat_std[:3, :3]
        out[3, :3] = mat_std[:3, 3]
        return out

    for n in joint_nodes:
        node = g.nodes[n]
        name = node.name or f"bone_{n}"

        w_std = world_matrix(n)
        pn = parent_of.get(n)
        parent_name = None
        if pn is not None and pn in joint_set:
            parent_name = g.nodes[pn].name or f"bone_{pn}"
        # local matrix relative to parent (identity parent world if root)
        if pn is not None:
            l_std = np.linalg.inv(world_matrix(pn)) @ w_std
        else:
            l_std = w_std

        bone = BoneData(name)
        bone.parent_name = parent_name
        bone.has_skeleton_attr = True
        bone.world_matrix = _to_fbx_layout(w_std)
        bone.local_matrix = _to_fbx_layout(l_std)
        bone.head = w_std[:3, 3].copy()

        q_xyzw = R.from_matrix(w_std[:3, :3]).as_quat()
        rest_q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        bone.rest_rotation = rest_q

        skel.add_bone(bone)
        skel.all_nodes[name] = name
        skel.node_rest_rotations[name] = rest_q
        node_of_joint.append(n)

    skel.frame_start = 0
    skel.frame_end = 0

    _log(f"  Target skeleton built: {len(skel.bones)} bones")
    _log(f"  Target bone names: {[b.name for b in skel.bones.values()]}")

    return g, skel, node_of_joint


# ============================================================================
# Write animation channels into the glTF
# ============================================================================

def _append_accessor(g, blob: bytearray, array: np.ndarray, type_str: str,
                     with_minmax: bool = False) -> int:
    """Append float32 ``array`` to the binary blob; register bufferView+accessor."""
    from pygltflib import Accessor, BufferView

    arr = np.ascontiguousarray(array, dtype=np.float32)
    data = arr.tobytes()
    byte_offset = len(blob)
    blob.extend(data)
    while len(blob) % 4 != 0:  # 4-byte align the next view
        blob.append(0)

    g.bufferViews.append(BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data)))
    bv_index = len(g.bufferViews) - 1

    acc = Accessor(
        bufferView=bv_index,
        componentType=_FLOAT,
        count=arr.shape[0],
        type=type_str,
    )
    if with_minmax:
        acc.min = arr.min(axis=0).reshape(-1).tolist()
        acc.max = arr.max(axis=0).reshape(-1).tolist()
    g.accessors.append(acc)
    return len(g.accessors) - 1


def write_animation_to_gltf(g, node_of_joint, name_to_joint_index,
                            ret_rots: dict, ret_locs: dict,
                            frame_start: int, frame_end: int, fps: float):
    """Add a single glTF Animation with rotation (+root translation) channels."""
    from pygltflib import (
        Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler,
    )

    _log("--- Writing animation to glTF ---")
    frames = list(range(frame_start, frame_end + 1))
    times = np.array([f / float(fps) for f in frames], dtype=np.float32)

    # Existing binary blob; keep it 4-byte aligned before we append.
    blob = bytearray(g.binary_blob() or b"")
    while len(blob) % 4 != 0:
        blob.append(0)

    time_acc = _append_accessor(g, blob, times.reshape(-1, 1), "SCALAR", with_minmax=True)

    samplers: list = []
    channels: list = []

    def _add_channel(node_index: int, path: str, output_acc: int):
        samplers.append(AnimationSampler(input=time_acc, output=output_acc, interpolation="LINEAR"))
        s_idx = len(samplers) - 1
        channels.append(AnimationChannel(
            sampler=s_idx,
            target=AnimationChannelTarget(node=node_index, path=path),
        ))

    n_rot = 0
    n_loc = 0
    for name, per_frame in ret_rots.items():
        j = name_to_joint_index.get(name)
        if j is None:
            continue
        node_index = node_of_joint[j]

        # rotation: wxyz -> glTF xyzw
        quats = np.empty((len(frames), 4), dtype=np.float32)
        for k, f in enumerate(frames):
            q = per_frame.get(f)
            if q is None:
                q = np.array([1.0, 0.0, 0.0, 0.0])
            quats[k] = [q[1], q[2], q[3], q[0]]
        rot_acc = _append_accessor(g, blob, quats, "VEC4")
        _add_channel(node_index, "rotation", rot_acc)
        n_rot += 1

        if name in ret_locs:
            locs = np.empty((len(frames), 3), dtype=np.float32)
            for k, f in enumerate(frames):
                v = ret_locs[name].get(f)
                locs[k] = v if v is not None else [0.0, 0.0, 0.0]
            loc_acc = _append_accessor(g, blob, locs, "VEC3")
            _add_channel(node_index, "translation", loc_acc)
            n_loc += 1

    if not channels:
        _log("  WARNING: no channels written (no bone matches).")

    # Strip any pre-existing animations (e.g. a Mixamo bind/rest 'Layer0' that
    # rode in with the source rig) so the generated 'kimodo' clip is the ONLY
    # animation — otherwise viewers default to the defunct clip and the character
    # appears to not move until the user manually picks 'kimodo'. The old clips'
    # accessors/bufferViews are left in place (unreferenced, harmless).
    if g.animations:
        _log(f"  Stripping {len(g.animations)} pre-existing animation(s): "
             f"{[a.name for a in g.animations]}")
    g.animations = [Animation(name="kimodo", samplers=samplers, channels=channels)]

    # Update the single GLB buffer to the new length.
    while len(blob) % 4 != 0:
        blob.append(0)
    g.buffers[0].byteLength = len(blob)
    g.buffers[0].uri = None
    g.set_binary_blob(bytes(blob))

    _log(f"  Wrote {n_rot} rotation channel(s), {n_loc} translation channel(s), "
         f"{len(frames)} keyframes @ {fps} fps")


# ============================================================================
# Orientation fix
# ============================================================================

def _shortest_arc_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit-ish vector ``a`` onto ``b`` (shortest arc)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 1.0 - 1e-8:
        return np.eye(3)
    if d < -1.0 + 1e-8:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return R.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    axis = np.cross(a, b)
    return R.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(d)).as_matrix()


def _suffix_node_index(g):
    """Map lowercase bone-name suffix (e.g. 'hips', after any 'mixamorig8:') -> node index."""
    idx = {}
    for i, n in enumerate(g.nodes):
        nm = (n.name or "").split(":")[-1].lower()
        if nm and nm not in idx:
            idx[nm] = i
    return idx


def _animated_world_positions(g, anim, node_indices, frame):
    """World positions of ``node_indices`` at animation ``frame`` (index into the
    keyframe arrays), evaluating the animation's rotation/translation channels and
    the static rest TRS for everything else."""
    parent = {}
    for i, n in enumerate(g.nodes):
        for c in (n.children or []):
            parent[c] = i

    def _acc(i):
        a = g.accessors[i]
        bv = g.bufferViews[a.bufferView]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        nc = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a.type]
        return np.frombuffer(g.binary_blob(), np.float32, a.count * nc, off).reshape(a.count, nc)

    anim_rot, anim_loc = {}, {}
    for ch in anim.channels:
        s = anim.samplers[ch.sampler]
        (anim_rot if ch.target.path == "rotation" else anim_loc)[ch.target.node] = _acc(s.output)

    def local(n):
        m = _node_local_matrix(g.nodes[n])  # rest TRS
        if n in anim_rot:
            fi = min(frame, len(anim_rot[n]) - 1)
            m[:3, :3] = R.from_quat(anim_rot[n][fi]).as_matrix()
        if n in anim_loc:
            fi = min(frame, len(anim_loc[n]) - 1)
            m[:3, 3] = anim_loc[n][fi]
        return m

    out = {}
    for key, n in node_indices.items():
        M = local(n)
        m = n
        while m in parent:
            m = parent[m]
            M = local(m) @ M
        out[key] = M[:3, 3]
    return out


def _detect_up_axis_animated(g, anim):
    """World UP direction of the ANIMATED character, averaged over a few frames.

    The tip we correct lives in the animation (the retarget replaces the hips'
    bind rotation, which on a Z-up armature drops the character onto its face), so
    up is measured from the posed skeleton, not the rest pose. Two cues per frame:
    hips->neck (torso up) and ankle->knee (shin up). Returns a unit vector or None.
    """
    idx = _suffix_node_index(g)
    want = {}
    for k in ("hips", "neck", "leftleg", "leftfoot", "rightleg", "rightfoot"):
        if k in idx:
            want[k] = idx[k]
    if "hips" not in want or "neck" not in want:
        return None

    nkey = anim.samplers[0].input if anim.samplers else None
    nframes = g.accessors[nkey].count if nkey is not None else 1
    frames = sorted(set([0, nframes // 2, max(0, nframes - 1)]))

    up = np.zeros(3)
    for f in frames:
        p = _animated_world_positions(g, anim, want, f)
        cues = [p["neck"] - p["hips"]]
        for knee, ankle in (("leftleg", "leftfoot"), ("rightleg", "rightfoot")):
            if knee in p and ankle in p:
                cues.append(p[knee] - p[ankle])
        for c in cues:
            n = np.linalg.norm(c)
            if n > 1e-9:
                up += c / n
    n = np.linalg.norm(up)
    return up / n if n > 1e-9 else None


def _apply_orientation_fix(g):
    """Stand the animated character upright (Y-up) by pre-rotating scene root(s).

    Detects the animated up axis and bakes the shortest-arc rotation that maps it
    onto +Y into every scene-root node. The rotation is a pure tilt about a
    horizontal axis, so it never changes facing/yaw and is a no-op for a rig that
    already animates upright. Returns True if a non-trivial fix applied.
    """
    if not g.animations:
        return False
    up = _detect_up_axis_animated(g, g.animations[0])
    if up is None:
        _log("  Orientation fix: could not detect up axis (missing hips/neck) — skipped.")
        return False
    Rfix = _shortest_arc_matrix(up, np.array([0.0, 1.0, 0.0]))
    angle = np.degrees(np.arccos(np.clip((np.trace(Rfix) - 1) / 2, -1, 1)))
    if angle < 0.5:
        _log(f"  Orientation fix: up={np.round(up,3)} already ~+Y (Δ={angle:.2f}°) — no change.")
        return False
    _log(f"  Orientation fix: up={np.round(up,3)} -> +Y, tilt {angle:.1f}°")

    Rfix4 = np.eye(4)
    Rfix4[:3, :3] = Rfix
    roots = g.scenes[g.scene or 0].nodes
    for n in roots:
        node = g.nodes[n]
        if node.matrix:
            M = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # column-major -> row math
            M = Rfix4 @ M
            node.matrix = M.T.reshape(16).tolist()
        else:
            r = node.rotation or [0.0, 0.0, 0.0, 1.0]  # xyzw
            new_r = R.from_matrix(Rfix) * R.from_quat(r)
            node.rotation = new_r.as_quat().tolist()  # xyzw
    return True


# ============================================================================
# Public API
# ============================================================================

def export_kimodo_glb(
    motion_data,
    target_glb_path: str,
    output_path: str,
    sample_index: int = 0,
    yaw_offset: float = 0.0,
    force_scale: float = 0.0,
    map_fingers: bool = False,
    auto_fix_input_pose: bool = False,
    fix_orientation: bool = False,
) -> str:
    """Retarget Kimodo SOMA motion onto a rigged glb and save an animated glb.

    ``auto_fix_input_pose`` (opt-in) enables the rest-direction A-pose correction;
    see ``retarget_animation``. ``fix_orientation`` (opt-in) stands a rig that was
    authored tipped over (e.g. a Z-up Mixamo armature) back upright. Any
    pre-existing animations on the target are always stripped so the generated
    clip is the only one. Returns the path to the saved glb.
    """
    _log("=" * 60)
    _log("KIMODO GLB EXPORT START")
    _log("=" * 60)
    _log(f"  sample_index: {sample_index}")
    _log(f"  target_glb_path: {target_glb_path}")
    _log(f"  output_path: {output_path}")
    _log(f"  yaw_offset: {yaw_offset}  force_scale: {force_scale}")
    _log(f"  motion skeleton: {motion_data.skeleton_name}  fps: {motion_data.fps}")

    try:
        # 1. Build source skeleton from Kimodo motion (shared with the FBX path).
        src_skel = kimodo_to_source_skeleton(motion_data, sample_index)

        # 2. Load target glb skeleton.
        g, tgt_skel, node_of_joint = load_target_glb(target_glb_path)
        tgt_skel.fps = float(motion_data.fps)

        # joint-name (lowercased, as stored by SkeletonData) -> joint order index
        name_to_joint_index = {}
        for j, n in enumerate(node_of_joint):
            nm = (g.nodes[n].name or f"bone_{n}")
            name_to_joint_index[nm] = j

        _log(f"  Source bones: {len(src_skel.bones)}  Target bones: {len(tgt_skel.bones)}")

        # 3. Retarget (shared math, SOMA -> Mixamo mapping).
        mapping = SOMA_TO_MIXAMO if map_fingers else _body_only_mapping(SOMA_TO_MIXAMO)
        _log(f"  Bone mapping: {len(mapping)} entries (map_fingers={map_fingers})")
        ret_rots, ret_locs = retarget_animation(
            src_skel, tgt_skel, mapping,
            force_scale=force_scale, yaw_offset=yaw_offset,
            auto_fix_input_pose=auto_fix_input_pose,
        )
        if len(ret_rots) == 0:
            _log("WARNING: No bone pairs matched — glb will have no animation!")

        # 4. Write animation channels into the glTF (strips any stale clips).
        write_animation_to_gltf(
            g, node_of_joint, name_to_joint_index,
            ret_rots, ret_locs,
            src_skel.frame_start, src_skel.frame_end, float(motion_data.fps),
        )

        # 5. Optionally stand the character upright.
        if fix_orientation:
            _apply_orientation_fix(g)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        g.save_binary(output_path)

        if os.path.exists(output_path):
            _log(f"  SUCCESS: Saved {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            _log(f"  WARNING: File not found after save: {output_path}")

        _log("=" * 60)
        _log("KIMODO GLB EXPORT DONE")
        _log("=" * 60)
        return output_path

    except Exception as e:
        _log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        raise
