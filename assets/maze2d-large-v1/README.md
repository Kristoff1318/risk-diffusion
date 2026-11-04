# Local pretrained assets

- `state_1960000.pt`: unchanged pretrained Diffuser state, EMA epoch 1,960,000,
  originally used for Maze2D Large v1. Approximately 29 MB, tracked in this repo.
- `config.json`: model and diffusion constructor arguments transcribed from
  that checkpoint's saved configuration. No absolute paths or pickle configs.
- `normalization.npz`: cached original dataset limits. Inference does not need
  to rebuild or download the D4RL dataset.
- `MANIFEST.json`: provenance and hashes; checked before model loading.

These are actual local files, not symlinks, and require no other project folder.
The source license and notice are in `third_party/diffuser/` at repository root.
