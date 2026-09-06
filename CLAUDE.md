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

The correction is exposed as a **`direction_aware` toggle** (default ON) on both
the Export GLB and Export FBX nodes, threaded through
`retarget_animation(..., direction_aware=...)`. Turning it OFF restores the exact
legacy rotation-only path — the escape hatch if the correction ever misbehaves on
an unusual rig.

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
