import os
import sys
import uuid
from datetime import datetime

print("[Kimodo] nodes.py: starting imports...", flush=True)

import torch
import numpy as np

import folder_paths
import comfy.model_management as mm
import comfy.utils

print("[Kimodo] nodes.py: comfy imports OK", flush=True)

# ---------------------------------------------------------------------------
# Path setup: add kimodo project root to sys.path so 'kimodo' package works
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KIMODO_PROJECT_DIR = os.path.join(CURRENT_DIR, "kimodo")

if KIMODO_PROJECT_DIR not in sys.path:
    sys.path.insert(0, KIMODO_PROJECT_DIR)
    print(f"[Kimodo] Added to sys.path: {KIMODO_PROJECT_DIR}", flush=True)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Install kimodo package in-place if not importable yet
try:
    import kimodo as _kimodo_pkg
    print(f"[Kimodo] kimodo package found: {_kimodo_pkg.__file__}", flush=True)
except ImportError:
    print("[Kimodo] kimodo package not installed, running pip install -e ...", flush=True)
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-e", KIMODO_PROJECT_DIR,
        "--no-build-isolation",
    ], env={**os.environ, "SKIP_MOTION_CORRECTION_IN_SETUP": "1"})
    import kimodo as _kimodo_pkg
    print(f"[Kimodo] kimodo package installed: {_kimodo_pkg.__file__}", flush=True)

from kimodo import load_model, AVAILABLE_MODELS
from kimodo.constraints import load_constraints_lst
from kimodo.model.registry import MODEL_INFOS, KIMODO_MODELS, get_model_info
from kimodo.tools import seed_everything

print("[Kimodo] All imports OK", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KIMODO_MODELS_DIR = os.path.join(folder_paths.models_dir, "Kimodo")
os.makedirs(KIMODO_MODELS_DIR, exist_ok=True)

# Build display name list for dropdown
_MODEL_CHOICES = []
for info in MODEL_INFOS:
    if info.family == "Kimodo":
        _MODEL_CHOICES.append(info.display_name)


# ---------------------------------------------------------------------------
# Wrapper data classes
# ---------------------------------------------------------------------------
class KimodoCondData:
    """Wraps text conditioning output for passing between TextEncode and Sampler."""
    def __init__(self, text_feat, text_pad_mask, texts):
        self.text_feat = text_feat            # [B, max_len, dim] tensor on device
        self.text_pad_mask = text_pad_mask    # [B, max_len] bool tensor
        self.texts = texts                    # list of strings


class KimodoMotionData:
    """Wraps Kimodo generation output for passing between nodes."""
    def __init__(self, output_dict, model_name, skeleton_name, fps, texts, num_frames, num_samples,
                 joint_parents=None, joint_names=None, neutral_joints=None,
                 skeleton=None, constraint_lst=None):
        self.output_dict = output_dict        # dict with numpy/torch arrays
        self.model_name = model_name
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.texts = texts
        self.num_frames = num_frames
        self.num_samples = num_samples
        self.batch_size = int(output_dict["posed_joints"].shape[0])
        self.joint_parents = joint_parents    # list of ints (-1 for root)
        self.joint_names = joint_names        # list of strings
        self.neutral_joints = neutral_joints  # [J, 3] T-pose positions (numpy)
        self.skeleton = skeleton              # skeleton object (for post-process)
        self.constraint_lst = constraint_lst  # constraints (for post-process)


# ---------------------------------------------------------------------------
# Node: Load Model
# ---------------------------------------------------------------------------
class Kimodo_LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (_MODEL_CHOICES, {"default": _MODEL_CHOICES[0] if _MODEL_CHOICES else "Kimodo-SOMA-RP-v1",
                                           "tooltip": "Kimodo model variant. Models auto-download from HuggingFace on first use."}),
            },
        }

    RETURN_TYPES = ("KIMODO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Kimodo"

    def load(self, model):
        device = mm.get_torch_device()
        print(f"[Kimodo] Loading model: {model}", flush=True)

        # Resolve display name to short key
        from kimodo.model.registry import get_short_key_from_display_name
        short_key = get_short_key_from_display_name(model)
        if short_key is None:
            short_key = model

        # Set CHECKPOINT_DIR to ComfyUI models folder if models exist there
        if os.path.isdir(os.path.join(KIMODO_MODELS_DIR, model)):
            os.environ["CHECKPOINT_DIR"] = KIMODO_MODELS_DIR

        kimodo_model, resolved = load_model(
            short_key, device=str(device), return_resolved_name=True
        )

        info = get_model_info(resolved)
        display = info.display_name if info else resolved
        print(f"[Kimodo] Model loaded: {display} (skeleton={kimodo_model.skeleton.name}, fps={kimodo_model.fps})", flush=True)

        return (kimodo_model,)



# ---------------------------------------------------------------------------
# Node: Preview Motion (2D skeleton stick figure)
# ---------------------------------------------------------------------------
class Kimodo_Preview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to preview (0-indexed)"}),
                "frame_index": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "Which frame to render (0-indexed)"}),
                "image_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "Kimodo"

    def preview(self, motion, sample_index=0, frame_index=0, image_size=512):
        import cv2

        output = motion.output_dict
        joints = output["posed_joints"]  # [B, T, J, 3]
        parents = motion.joint_parents or []

        idx = min(sample_index, joints.shape[0] - 1)
        fidx = min(frame_index, joints.shape[1] - 1)
        frame_joints = joints[idx, fidx]  # [J, 3]

        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 40

        # Front view: X=horizontal, Y=vertical (inverted)
        x = frame_joints[:, 0]
        y = frame_joints[:, 1]

        # Normalize to image coordinates
        all_coords = np.stack([x, y], axis=-1)
        cmin = all_coords.min(axis=0)
        cmax = all_coords.max(axis=0)
        span = (cmax - cmin).max() + 1e-6
        margin = image_size * 0.1
        effective = image_size - 2 * margin

        px = ((x - cmin[0]) / span * effective + margin).astype(np.int32)
        py = (image_size - ((y - cmin[1]) / span * effective + margin)).astype(np.int32)

        # Draw bones using parent indices
        for j in range(len(px)):
            if j < len(parents) and parents[j] >= 0:
                p = parents[j]
                cv2.line(img, (int(px[j]), int(py[j])), (int(px[p]), int(py[p])), (100, 180, 255), 2)

        # Draw joints on top
        for j in range(len(px)):
            cv2.circle(img, (int(px[j]), int(py[j])), 3, (0, 220, 120), -1)

        cv2.putText(img, f"Frame {fidx}/{joints.shape[1]-1}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Sample {idx}/{joints.shape[0]-1}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Joints: {joints.shape[2]}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        result = img.astype(np.float32) / 255.0
        return (torch.from_numpy(result).unsqueeze(0),)


# ---------------------------------------------------------------------------
# Node: Save NPZ
# ---------------------------------------------------------------------------
class Kimodo_SaveNPZ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "filename_prefix": ("STRING", {"default": "kimodo_motion"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def save(self, motion, filename_prefix="kimodo_motion"):
        output = motion.output_dict
        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        n_samples = motion.batch_size
        saved_paths = []

        if n_samples == 1:
            path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.npz")
            single = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                      for k, v in output.items()}
            np.savez(path, **single)
            saved_paths.append(path)
            print(f"[Kimodo] Saved NPZ: {path}", flush=True)
        else:
            for i in range(n_samples):
                path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}_{i:02d}.npz")
                single = {k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                          for k, v in output.items()}
                np.savez(path, **single)
                saved_paths.append(path)
            print(f"[Kimodo] Saved {n_samples} NPZ files to {output_dir}", flush=True)

        return (saved_paths[0],)


# ---------------------------------------------------------------------------
# Node: Export BVH
# ---------------------------------------------------------------------------
class Kimodo_ExportBVH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "filename_prefix": ("STRING", {"default": "kimodo_motion"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to export (0-indexed)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def export(self, motion, filename_prefix="kimodo_motion", sample_index=0):
        output = motion.output_dict
        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton import global_rots_to_local_rots, SOMASkeleton30

        # BVH export requires SOMA skeleton
        device = mm.get_torch_device()

        # We need the skeleton - reload briefly for export
        from kimodo import load_model as _load
        # Use a lightweight approach: get skeleton from motion data
        # The skeleton info is embedded in the output

        try:
            model_for_skel = _load(motion.model_name, device=str(device), return_resolved_name=False)
            skeleton = model_for_skel.skeleton
        except Exception:
            print("[Kimodo] Warning: Could not load skeleton for BVH export. Trying SOMA default.", flush=True)
            model_for_skel = _load("kimodo-soma-rp", device=str(device), return_resolved_name=False)
            skeleton = model_for_skel.skeleton

        if "somaskel" not in skeleton.name:
            print("[Kimodo] BVH export is only supported for SOMA skeletons. Skipping.", flush=True)
            return ("",)

        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(device)

        idx = min(sample_index, motion.batch_size - 1)
        joints_pos = torch.from_numpy(output["posed_joints"][idx]).to(device)
        joints_rot = torch.from_numpy(output["global_rot_mats"][idx]).to(device)
        local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
        root_positions = joints_pos[:, skeleton.root_idx, :]

        path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.bvh")
        save_motion_bvh(path, local_rot_mats, root_positions, skeleton=skeleton, fps=motion.fps)

        print(f"[Kimodo] Saved BVH: {path}", flush=True)
        return (path,)


# ---------------------------------------------------------------------------
# Node: Preview 3D (Three.js skeleton animation)
# ---------------------------------------------------------------------------
class Kimodo_Preview3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to preview (0-indexed)"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def preview(self, motion, sample_index=0):
        import json

        output = motion.output_dict
        joints = output["posed_joints"]  # [B, T, J, 3]
        idx = min(sample_index, joints.shape[0] - 1)
        sample_joints = joints[idx]  # [T, J, 3]

        # Flatten to list for JSON
        joints_flat = sample_joints.flatten().tolist()

        # Get parent indices
        parents = motion.joint_parents or list(range(-1, sample_joints.shape[1] - 1))
        joint_names = motion.joint_names or [f"joint_{i}" for i in range(sample_joints.shape[1])]

        motion_json = json.dumps({
            "joints": joints_flat,
            "parents": parents,
            "joint_names": joint_names,
            "num_joints": int(sample_joints.shape[1]),
            "num_frames": int(sample_joints.shape[0]),
            "fps": int(motion.fps),
            "text": " ".join(motion.texts) if motion.texts else "",
            "skeleton": motion.skeleton_name,
        })

        print(f"[Kimodo] Preview3D: sample {idx}, {sample_joints.shape[0]} frames, "
              f"{sample_joints.shape[1]} joints", flush=True)

        return {"ui": {"motion_json": [motion_json]}}


# ---------------------------------------------------------------------------
# Node: Export FBX (Mixamo retarget)
# ---------------------------------------------------------------------------
class Kimodo_ExportFBX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "custom_fbx_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to Mixamo-rigged FBX character. "
                               "Supports: 'input/3d/char.fbx', absolute path, "
                               "or relative to ComfyUI input/ folder.",
                }),
                "filename_prefix": ("STRING", {"default": "kimodo_fbx"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to export (0-indexed)"}),
            },
            "optional": {
                "yaw_offset": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotate character around Y-axis (degrees).",
                }),
                "scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Force scale multiplier (0 = auto height-based scaling).",
                }),
                "direction_aware": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Correct each bone for the source-vs-target rest DIRECTION "
                               "so an A-pose rig animates without the arms flapping. "
                               "Leave ON for A-pose characters. Turn OFF for the legacy "
                               "rotation-only behaviour (only correct for a T-posed target).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    @staticmethod
    def _resolve_fbx_path(path_str: str) -> str | None:
        """Resolve FBX path: absolute, input/…, output/…, or default to input/."""
        print(f"[Kimodo] _resolve_fbx_path input: '{path_str}'", flush=True)
        if not path_str or not path_str.strip():
            print("[Kimodo] FBX path is empty", flush=True)
            return None
        path = path_str.strip().replace("\\", "/")
        print(f"[Kimodo] Normalized path: '{path}'", flush=True)

        # Absolute
        if os.path.isabs(path):
            print(f"[Kimodo] Checking absolute: '{path}' exists={os.path.exists(path)}", flush=True)
            if os.path.exists(path):
                return path

        # Relative to ComfyUI root
        comfy_root = os.path.dirname(folder_paths.get_output_directory())
        print(f"[Kimodo] ComfyUI root: '{comfy_root}'", flush=True)

        candidates = []
        if path.startswith("input/") or path.startswith("output/"):
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))
        else:
            candidates.append(os.path.normpath(os.path.join(comfy_root, "input", path)))
            candidates.append(os.path.normpath(os.path.join(comfy_root, path)))

        for c in candidates:
            exists = os.path.exists(c)
            print(f"[Kimodo] Trying: '{c}' exists={exists}", flush=True)
            if exists:
                return c

        print(f"[Kimodo] FBX path not found in any candidate location", flush=True)
        return None

    def export(self, motion, custom_fbx_path, filename_prefix="kimodo_fbx",
               sample_index=0, yaw_offset=0.0, scale=0.0, direction_aware=True):

        from kimodo_retarget_fbx import export_kimodo_fbx, HAS_FBX_SDK

        if not HAS_FBX_SDK:
            print("[Kimodo] FBX SDK not installed. Install fbxsdkpy first.", flush=True)
            return ("FBX SDK not installed",)

        resolved = self._resolve_fbx_path(custom_fbx_path)
        if resolved is None:
            print(f"[Kimodo] No valid FBX path provided. Input was: '{custom_fbx_path}'", flush=True)
            print("[Kimodo] Please provide a Mixamo FBX file path. Examples:", flush=True)
            print("[Kimodo]   Absolute: I:/ComfyUI/input/3d/character.fbx", flush=True)
            print("[Kimodo]   Relative: 3d/character.fbx (looks in ComfyUI/input/)", flush=True)
            print("[Kimodo]   Prefixed: input/3d/character.fbx", flush=True)
            return ("",)

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        idx = min(sample_index, motion.batch_size - 1)
        out_path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.fbx")

        try:
            export_kimodo_fbx(
                motion_data=motion,
                target_fbx_path=resolved,
                output_path=out_path,
                sample_index=idx,
                yaw_offset=yaw_offset,
                force_scale=scale,
                direction_aware=direction_aware,
            )
            print(f"[Kimodo] FBX exported: {out_path}", flush=True)
        except Exception as e:
            print(f"[Kimodo] FBX export error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return (f"Export failed: {e}",)

        return (out_path,)


# ---------------------------------------------------------------------------
# Node: Export GLB (retarget onto a rigged glTF/GLB character)
# ---------------------------------------------------------------------------
class Kimodo_ExportGLB:
    @classmethod
    def INPUT_TYPES(s):
        from kimodo_comfy_types import RIGGED_FILE3D_TYPES
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "rigged_glb": (RIGGED_FILE3D_TYPES, {
                    "tooltip": "Rigged glb/gltf character (e.g. from a Load3D node or "
                               "SkinTokensRig). Joints matched by Mixamo naming.",
                }),
                "filename_prefix": ("STRING", {"default": "kimodo_glb"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "Which sample to export (0-indexed)"}),
            },
            "optional": {
                "yaw_offset": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotate character around Y-axis (degrees).",
                }),
                "scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Force scale multiplier (0 = auto height-based scaling).",
                }),
                "map_fingers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Retarget the finger bones. Turn OFF to leave the hands "
                               "in their rest pose (useful if fingers distort — isolates "
                               "the body retarget from the fragile finger chain).",
                }),
                "direction_aware": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Correct each bone for the source-vs-target rest DIRECTION "
                               "so an A-pose rig animates without the arms flapping. "
                               "Leave ON for A-pose characters (e.g. SkinTokensRig). "
                               "Turn OFF for the legacy rotation-only behaviour (only "
                               "correct for a T-posed target).",
                }),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("rigged_glb",)
    FUNCTION = "export"
    CATEGORY = "Kimodo"
    OUTPUT_NODE = True

    def export(self, motion, rigged_glb, filename_prefix="kimodo_glb",
               sample_index=0, yaw_offset=0.0, scale=0.0, map_fingers=True,
               direction_aware=True):

        from kimodo_comfy_types import file3d_to_path, make_file3d
        from kimodo_retarget_glb import export_kimodo_glb

        target_path = file3d_to_path(rigged_glb)
        if not target_path or not os.path.exists(target_path):
            raise RuntimeError(
                f"Kimodo Export GLB: input character file not found ('{target_path}'). "
                "Load a rigged glb/gltf in the Load3D node."
            )

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        idx = min(sample_index, motion.batch_size - 1)
        out_path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}.glb")

        try:
            export_kimodo_glb(
                motion_data=motion,
                target_glb_path=target_path,
                output_path=out_path,
                sample_index=idx,
                yaw_offset=yaw_offset,
                force_scale=scale,
                map_fingers=map_fingers,
                direction_aware=direction_aware,
            )
            print(f"[Kimodo] GLB exported: {out_path}", flush=True)
        except Exception as e:
            print(f"[Kimodo] GLB export error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Kimodo Export GLB failed: {e}. If this says 'no skin', the loaded "
                "character is not rigged — rig it first (e.g. SkinTokensRig)."
            ) from e

        return {"ui": {"model_file": [out_path]}, "result": (make_file3d(out_path, "glb"),)}


# ---------------------------------------------------------------------------
# Node: Text Encode (split from Generate)
# ---------------------------------------------------------------------------
class Kimodo_TextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KIMODO_MODEL",),
                "prompt": ("STRING", {"default": "A person walks forward.",
                                      "multiline": True,
                                      "tooltip": "Text prompt. Use periods to separate multiple motion segments."}),
            },
        }

    RETURN_TYPES = ("KIMODO_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Kimodo/Conditioning"

    def encode(self, model, prompt):
        device = model.device

        texts = [t.strip() + "." for t in prompt.split(".") if t.strip()]
        if not texts:
            texts = [prompt]

        print(f"[Kimodo] TextEncode: {len(texts)} segment(s)", flush=True)
        for i, t in enumerate(texts):
            print(f"  [{i}] '{t}'", flush=True)

        from kimodo.model.kimodo_model import sanitize_texts
        texts = sanitize_texts(texts)

        text_feat, text_length = model.text_encoder(texts)
        text_feat = text_feat.to(device)

        empty_mask = [len(t.strip()) == 0 for t in texts]
        text_feat[empty_mask] = 0

        batch_size, maxlen = text_feat.shape[:2]
        tl = torch.tensor(text_length, device=device)
        tl[empty_mask] = 0
        text_pad_mask = torch.arange(maxlen, device=device).expand(batch_size, maxlen) < tl[:, None]

        print(f"[Kimodo] TextEncode: text_feat shape={text_feat.shape}", flush=True)

        cond = KimodoCondData(text_feat=text_feat, text_pad_mask=text_pad_mask, texts=texts)
        return (cond,)


# ---------------------------------------------------------------------------
# Node: Sampler (split from Generate — diffusion only, no post-processing)
# ---------------------------------------------------------------------------
class Kimodo_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KIMODO_MODEL",),
                "conditioning": ("KIMODO_COND",),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 30.0, "step": 0.5,
                                       "tooltip": "Duration in seconds per segment"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 16}),
                "diffusion_steps": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "constraints_json": ("STRING", {"default": "",
                                                "tooltip": "Path to constraints JSON file (optional)"}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "sample"
    CATEGORY = "Kimodo"

    def sample(self, model, conditioning, duration=5.0, seed=42, num_samples=1,
               diffusion_steps=100, constraints_json=""):

        seed_everything(seed)
        texts = conditioning.texts
        num_frames = [int(duration * model.fps)] * len(texts)
        multi_prompt = len(texts) > 1

        print(f"[Kimodo] Sampler: {len(texts)} segment(s), {num_frames[0]} frames, "
              f"{num_samples} sample(s), {diffusion_steps} steps", flush=True)

        constraint_lst = []
        if constraints_json and os.path.isfile(constraints_json):
            constraint_lst = load_constraints_lst(constraints_json, model.skeleton)
            print(f"[Kimodo] Loaded {len(constraint_lst)} constraint(s)", flush=True)

        # Call model WITHOUT post-processing
        output = model(
            texts,
            num_frames,
            num_denoising_steps=diffusion_steps,
            num_samples=num_samples,
            multi_prompt=multi_prompt,
            constraint_lst=constraint_lst,
            post_processing=False,
            return_numpy=True,
        )

        return (self._wrap_output(model, output, texts, num_frames, num_samples, constraint_lst),)

    @staticmethod
    def _wrap_output(model, output, texts, num_frames, num_samples, constraint_lst):
        from kimodo.skeleton.definitions import SOMASkeleton30
        skeleton_name = model.skeleton.name
        num_output_joints = output['posed_joints'].shape[2]
        skel_for_viz = model.skeleton
        if hasattr(model.skeleton, 'somaskel77') and num_output_joints == 77:
            skel_for_viz = model.skeleton.somaskel77
        elif hasattr(model.skeleton, 'somaskel30') and num_output_joints == 30:
            skel_for_viz = model.skeleton.somaskel30 if hasattr(model.skeleton, 'somaskel30') else model.skeleton

        joint_parents = skel_for_viz.joint_parents.cpu().tolist()
        joint_names = list(skel_for_viz.bone_order_names) if hasattr(skel_for_viz, 'bone_order_names') else []
        neutral_joints = None
        if hasattr(skel_for_viz, 'neutral_joints') and skel_for_viz.neutral_joints is not None:
            neutral_joints = skel_for_viz.neutral_joints.cpu().numpy()

        return KimodoMotionData(
            output_dict=output,
            model_name=str(getattr(model, '_resolved_name', 'unknown')),
            skeleton_name=skeleton_name,
            fps=model.fps,
            texts=texts,
            num_frames=num_frames,
            num_samples=num_samples,
            joint_parents=joint_parents,
            joint_names=joint_names,
            neutral_joints=neutral_joints,
            skeleton=model.skeleton,
            constraint_lst=constraint_lst,
        )


# ---------------------------------------------------------------------------
# Node: Post Process (foot-skate cleanup, separate from sampling)
# ---------------------------------------------------------------------------
class Kimodo_PostProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
            },
            "optional": {
                "root_margin": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Root correction margin in meters"}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "process"
    CATEGORY = "Kimodo"

    def process(self, motion, root_margin=0.04):
        skeleton = motion.skeleton
        if skeleton is None:
            print("[Kimodo] PostProcess: No skeleton available, skipping.", flush=True)
            return (motion,)

        if "g1" in skeleton.name.lower():
            print("[Kimodo] PostProcess: G1 skeleton, skipping.", flush=True)
            return (motion,)

        output = motion.output_dict

        # Need torch tensors for post-processing
        def _to_torch(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v)
            return v

        local_rot_mats = _to_torch(output["local_rot_mats"])
        root_positions = _to_torch(output["root_positions"])
        foot_contacts = _to_torch(output["foot_contacts"])

        try:
            from kimodo.postprocess import post_process_motion
            print(f"[Kimodo] PostProcess: applying foot-skate correction (root_margin={root_margin})", flush=True)

            corrected = post_process_motion(
                local_rot_mats,
                root_positions,
                foot_contacts,
                skeleton,
                motion.constraint_lst or [],
                root_margin=root_margin,
            )

            # Update output dict
            new_output = dict(output)
            for k, v in corrected.items():
                new_output[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

            # Rebuild KimodoMotionData with corrected output
            new_motion = KimodoMotionData(
                output_dict=new_output,
                model_name=motion.model_name,
                skeleton_name=motion.skeleton_name,
                fps=motion.fps,
                texts=motion.texts,
                num_frames=motion.num_frames,
                num_samples=motion.num_samples,
                joint_parents=motion.joint_parents,
                joint_names=motion.joint_names,
                neutral_joints=motion.neutral_joints,
                skeleton=motion.skeleton,
                constraint_lst=motion.constraint_lst,
            )
            print("[Kimodo] PostProcess: done.", flush=True)
            return (new_motion,)

        except Exception as e:
            print(f"[Kimodo] PostProcess error: {e}", flush=True)
            print("[Kimodo] Returning uncorrected motion.", flush=True)
            import traceback
            traceback.print_exc()
            return (motion,)


# ---------------------------------------------------------------------------
# Node mappings
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    # Loaders
    "Kimodo_LoadModel": Kimodo_LoadModel,
    # Conditioning
    "Kimodo_TextEncode": Kimodo_TextEncode,
    # Sampling
    "Kimodo_Sampler": Kimodo_Sampler,
    # Post-processing
    "Kimodo_PostProcess": Kimodo_PostProcess,
    # Preview
    "Kimodo_Preview": Kimodo_Preview,
    "Kimodo_Preview3D": Kimodo_Preview3D,
    # Export
    "Kimodo_SaveNPZ": Kimodo_SaveNPZ,
    "Kimodo_ExportBVH": Kimodo_ExportBVH,
    "Kimodo_ExportFBX": Kimodo_ExportFBX,
    "Kimodo_ExportGLB": Kimodo_ExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kimodo_LoadModel": "Kimodo Load Model",
    "Kimodo_TextEncode": "Kimodo Text Encode",
    "Kimodo_Sampler": "Kimodo Sampler",
    "Kimodo_PostProcess": "Kimodo Post Process",
    "Kimodo_Preview": "Kimodo Preview (2D)",
    "Kimodo_Preview3D": "Kimodo Preview 3D",
    "Kimodo_SaveNPZ": "Kimodo Save NPZ",
    "Kimodo_ExportBVH": "Kimodo Export BVH",
    "Kimodo_ExportFBX": "Kimodo Export FBX (Mixamo)",
    "Kimodo_ExportGLB": "Kimodo Export GLB (Mixamo)",
}
