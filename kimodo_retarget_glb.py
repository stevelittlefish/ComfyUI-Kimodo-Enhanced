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

    g.animations.append(Animation(name="kimodo", samplers=samplers, channels=channels))

    # Update the single GLB buffer to the new length.
    while len(blob) % 4 != 0:
        blob.append(0)
    g.buffers[0].byteLength = len(blob)
    g.buffers[0].uri = None
    g.set_binary_blob(bytes(blob))

    _log(f"  Wrote {n_rot} rotation channel(s), {n_loc} translation channel(s), "
         f"{len(frames)} keyframes @ {fps} fps")


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
) -> str:
    """Retarget Kimodo SOMA motion onto a rigged glb and save an animated glb.

    Returns the path to the saved glb.
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
        ret_rots, ret_locs = retarget_animation(
            src_skel, tgt_skel, SOMA_TO_MIXAMO,
            force_scale=force_scale, yaw_offset=yaw_offset,
        )
        if len(ret_rots) == 0:
            _log("WARNING: No bone pairs matched — glb will have no animation!")

        # 4. Write animation channels into the glTF and save.
        write_animation_to_gltf(
            g, node_of_joint, name_to_joint_index,
            ret_rots, ret_locs,
            src_skel.frame_start, src_skel.frame_end, float(motion_data.fps),
        )

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
