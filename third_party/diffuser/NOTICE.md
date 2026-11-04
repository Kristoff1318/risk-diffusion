# Local Diffuser inference source

The source in `src/diffuser/` derives from Michael Janner and Yilun Du's
[Diffuser](https://github.com/jannerm/diffuser), using the previously available
RB Diffuser checkout as the source copy. See LICENSE for the MIT notice and
SOURCE_MANIFEST.json for the copied files' SHA-256 hashes.

The temporal model, diffusion model/sampler, model helpers, Policy, normalizers,
D4RL environment loader, array utilities, and progress utilities are copied
byte-for-byte. Only package initializers are reduced to the inference modules
needed here. This avoids importing unrelated training, cloud, rendering,
robot-locomotion, and old risk modules. No old repository is imported at runtime.

The local project loads model architecture settings from JSON instead of
pickled Config objects. The pretrained state file is copied unchanged and its
hash is recorded in `assets/maze2d-large-v1/MANIFEST.json`. The original model,
EMA weights, normalization limits, and nominal sampling algorithm are retained.
