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
lives in bone **translations**). So `off` was identity and the source's world
rotation was copied straight onto the target — correct only if the target is
**also T-posed**. The subtle part: at run time SOMA's arms **hang down**; the
horizontal T-pose is only its rest *reference*. Copying that world rotation onto
an A-pose arm (already ~65° down) over-rotates it → the classic **arm "flapping"**.

The fix folds a per-bone rest-**direction** correction `G` into `off`, so the
target bone is driven to **point where the source bone points**:

```python
G   = shortest_arc(t_dir -> s_dir)     # target rest dir -> source rest dir
off = inv(s_rest) * G * t_rest
t_rot = s_rot * off                    # unchanged application; direction now tracks
```

where `t_dir`/`s_dir` are each bone's rest direction (toward its longest child).
This makes `v_target(f) == v_source(f)` every frame: SOMA arm down → target arm
down; SOMA arm raised → target arm raised.

> **History / do not regress:** the first attempt used a *conjugation*
> `corr * delta * inv(corr)`. That is wrong — it rigidly rotates the whole motion
> by the rest offset, so a source arm hanging **down** was swung ~65° sideways and
> the **arms crossed over the chest**. The right operation is the constant
> right-multiply `G` above (point-the-bone), not a conjugation. `test_retarget_
> direction.py::test_apose_source_down_keeps_target_down` locks this in.

Key properties (see `tests/test_retarget_direction.py`):
- **Strict generalization** — when source and target share a pose (both T-pose),
  `t_dir == s_dir`, `G == identity`, and `off` reduces **exactly** to the legacy
  `inv(s_rest) * t_rest`. The working T-pose path cannot regress.
- **Direction tracking** — the target bone points along the source bone's world
  direction on every frame (the actual retarget goal).

Scope is deliberately "A-pose and T-pose good; odd/extreme poses may be a bit
janky" — shortest-arc `G` leaves bone roll/twist undefined, which is fine for
that bar. `off` is applied exactly as before, so the **root/hip translation**
block is unaffected (for the near-vertical hips `G ≈ identity`).

The correction is exposed as an **`auto_fix_input_pose` toggle** (default **OFF —
opt-in**) on both the Export GLB and Export FBX nodes, threaded through
`retarget_animation(..., auto_fix_input_pose=...)`. It is off by default so nothing
changes unless the user asks; when the target is T-posed it reduces exactly to the
legacy path, so **it never alters a T-pose mesh** even when on. Turn it on for an
A-pose rig.

### Fingers are OFF by default (`map_fingers`)

Point-the-bone is faithful, which is the problem for fingers: SOMA's 77-joint
finger **rest** orientations differ sharply from a typical rig's (e.g. a rig
thumb resting ~down vs SOMA's thumb ~up-forward, >100° apart), so matching the
source finger's absolute world direction bends the target fingers into a **claw**
(the "spastic hands" symptom). A T-pose bake would NOT help — T-pose is undefined
for fingers. So `map_fingers` defaults **OFF** on both nodes: `_body_only_mapping`
drops the finger bones, leaving the hands in their natural rest pose while the
body animates. Turn it ON only for a rig whose fingers align well with SOMA.
A proper fix (transfer the source's finger *curl* relative to its own rest,
rather than absolute direction) is possible but unbuilt.

## GLB export: strip stale clips + orientation fix

Two glb-only export behaviours (`kimodo_retarget_glb.py`):

- **Always strip pre-existing animations.** A rig imported from Mixamo often
  ships a defunct bind/rest clip (`Armature|mixamo.com|Layer0`, ~2 keyframes).
  glTF viewers play the *first* animation, so the character looks frozen until
  the user manually picks `kimodo`. `write_animation_to_gltf` now sets
  `g.animations = [kimodo]` (not append), so the generated clip is the only one.
  Unconditional — not a toggle.

- **`fix_orientation` toggle (default OFF, GLB node only).** Some rigs animate
  **face-down**. Root cause: a Mixamo-style rig has a **Z-up `Armature` root
  (+90° X)** cancelled by the **hips bind rotation (−90° X)** — but the retarget
  treats the hips as a skeleton root (the Armature isn't a skin joint, so
  `parent_name` is None) and the animation channel **replaces** the hips bind
  rotation with SOMA's ~identity, dropping the −90° X compensation → the whole
  body tips onto its face. So the tip lives in the **animation, not the rest
  pose** — `_apply_orientation_fix` measures the up axis from the **posed**
  skeleton (`_detect_up_axis_animated`, averaged over a few frames: hips→neck and
  ankle→knee), **snaps it to the nearest principal axis** (`_snap_to_axis` — the
  rig is only ever tipped by a whole multiple of 90°, so this drops the
  character's walk/lean that would otherwise leave it leaning back), and bakes the
  rotation mapping that axis onto +Y into the scene-root node(s). It's a **clean
  90° tilt about a horizontal axis, so it never changes facing/yaw**, and is a
  no-op for a rig that already animates upright.
  Validated on `~/Documents/meshes/Debug/dummy_rotated.glb` (100° tip → upright).
  A cleaner long-term fix would be to compose the hips channel with the inverse
  of the Armature's world transform in the retarget itself; the toggle is the
  pragmatic route the user asked for.

## Root motion: parent-scale fix + `animate_in_place`

In the root/hips translation block of `retarget_animation`, the world-space
displacement is now converted into the hips' local channel with the parent's
**full inverse world-linear transform** (rotation AND scale), derived from the
bone's own matrices: `parent_lin = world_lin @ inv(local_lin)`. Previously it
only inverted the parent rotation (and for a non-joint parent even that fell back
to identity), so a **Mixamo-cm rig** (`Armature` scale **0.01**) received only
**~1/100th** of the root motion — the mesh "walked on the spot" while the raw
SOMA preview travelled. For a hips that is itself the scene root, `parent_lin` is
identity, so unit-scale rigs are unchanged. This is the **default** now: the
character travels, matching the source.

**`animate_in_place` toggle (default OFF, both nodes).** When on, the root's
**horizontal** travel (SOMA is Y-up, so the X/Z ground plane) is zeroed while the
**vertical** (Y) is kept — so the character stays on the spot but still bobs,
leaves the ground, and jumps. Masked in source space before any rotation.
Tests: `test_root_translation_accounts_for_parent_scale` (unit vs 0.01 parent →
100× local delta) and `test_animate_in_place_drops_horizontal_keeps_vertical`.

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
