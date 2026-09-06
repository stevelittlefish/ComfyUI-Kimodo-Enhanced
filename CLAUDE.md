# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

`ComfyUI-Kimodo-Enhanced` — a ComfyUI plugin wrapping **Kimodo** (NVIDIA's
kinematic motion-diffusion model): text prompt → 3D humanoid motion on a **SOMA**
skeleton, with export/retarget onto rigged characters (BVH, FBX-Mixamo, glb).
See `README.md` for the full node list.

The half that matters most here is the **retarget**: Kimodo generates motion on
its own SOMA skeleton (a T-pose with identity global rest rotations) and
retargets it onto a target character's skeleton.

- `nodes.py` — the ComfyUI node classes (`NODE_CLASS_MAPPINGS` at the bottom).
- `kimodo_retarget_fbx.py` — the shared retarget core: `SkeletonData`/`BoneData`,
  `kimodo_to_source_skeleton`, `SOMA_TO_MIXAMO`, and **`retarget_animation`**
  (the math), plus the FBX-SDK read/write path.
- `kimodo_retarget_glb.py` — the glb twin: reads the target skeleton from glTF
  nodes/skin and writes the animation back as glTF samplers/channels. Reuses
  `retarget_animation` unchanged.

## Retarget: rest-DIRECTION aware (the A-pose fix)

`retarget_animation` maps a source bone's motion onto a target bone. The original
code derived the source→target correction from rest **rotations** only:

```python
off = _quat_mul(_quat_inv(s_bone.rest_rotation), t_bone.rest_rotation)
t_rot = _quat_mul(s_rot, off)          # applies the source WORLD delta verbatim
```

SOMA rest rotations are identity, and glb rigs from the sister project
(`ComfyUI-SkinTokens-NoBlender`) also export identity bind rotations (their pose
lives in bone **translations**). So `off` was identity and the source's
world-space delta was applied straight onto the target — correct only if the
target is **also T-posed**. On an **A-pose** rig (arms ~65° down) this drove the
arm as if it started horizontal → the classic **arm "flapping"**.

The fix adds a per-bone **direction** correction `corr` = shortest-arc rotation
from the source bone's rest direction (toward its longest child) onto the
target's, and applies the source's rest-relative delta conjugated by it:

```python
delta_s = s_rot * inv(s_rest)
delta_t = corr * delta_s * inv(corr)
t_rot   = delta_t * t_rest
```

Key properties (see `tests/test_retarget_direction.py`):
- **Strict generalization** — when source and target share a pose (both T-pose),
  the two rest directions coincide, `corr == identity`, and the formula reduces
  **exactly** to the legacy `s_rot * off`. The working T-pose path cannot regress.
- **Rest preserved** — at the rest frame the target holds its own rest pose (an
  A-pose arm no longer snaps to horizontal).
- **Magnitude preserved** — conjugation is angle-preserving, so the target swings
  by the same angle the source does.

Scope is deliberately "A-pose and T-pose good; odd/extreme poses may be a bit
janky" — shortest-arc leaves bone-roll undefined, which is fine for that bar.
The **root/hip translation** block still uses the original `off` (unchanged),
so hip motion is byte-identical to before.

The correction is exposed as a **`direction_aware` toggle** (default ON) on both
the Export GLB and Export FBX nodes, threaded through
`retarget_animation(..., direction_aware=...)`. Turning it OFF restores the exact
legacy rotation-only path — the escape hatch if the correction ever misbehaves on
an unusual rig.

## Testing (no GPU / no FBX SDK / no ComfyUI needed)

`tests/test_retarget_direction.py` is pure-numpy: it builds tiny `SkeletonData`
skeletons and drives `retarget_animation` directly. `kimodo_retarget_fbx.py`
guards the FBX SDK import (`HAS_FBX_SDK`), so the module imports without it.

There is no venv in this repo. Reuse the sister project's:

```bash
source ~/git/ComfyUI-SkinTokens-NoBlender/.venv/bin/activate   # numpy + scipy + pytest
cd ~/git/ComfyUI-Kimodo-Enhanced
python -m pytest tests/ -q
```

Full visual confirmation (animate a real rigged glb, verify no flap) still needs
a live Kimodo run on the ComfyUI server — the CPU/local path only covers the
retarget math.

## Sister project

`~/git/ComfyUI-SkinTokens-NoBlender` produces the rigged (A-posed) glbs this pack
consumes. Its `spec/HANDOFF-animation-retarget.md` has the full root-cause story
for the flapping bug and the two-part plan (fix Kimodo — this repo — first; then
an optional T-pose bake there). This retarget fix is part A.

## Git workflow

- **Commit to `master`** (this repo's default branch), **push after every commit.**
- **Never rewrite history** — no amend, no rebase, no force-push. Fix mistakes
  with follow-up commits; revert a whole commit only if it's genuinely wrong.

## Deploy note

The ComfyUI server bakes custom nodes into its image via a `--depth=1` clone at
**build time** (`ComfyUIDocker/custom-nodes.yaml`). New commits need an **image
rebuild**, not just a restart. Check the running hash:
`docker compose exec comfyui sh -c 'git -C /srv/app/custom_nodes/<pack> rev-parse HEAD'`.
