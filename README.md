# Sky Sanctuary GS

Sky Sanctuary GS is a stylized floating-island sanctuary represented and rendered as
anisotropic Gaussian surfaces. The scene contains suspended islands, bridges, an open
crystal altar, a sparse grove of celestial cypress trees, translucent blue crystals,
and low cloud banks.

The final video is rendered by the custom Python Gaussian-splat renderer in this
repository. It does not use Blender or any third-party 3D rendering software.

## Environment

`setup.sh` creates the dedicated `sky_sanctuary_gs` conda environment with Python
3.10, activates it, and installs every Python dependency with `pip`.

```bash
chmod +x setup.sh run_quick.sh run.sh
./setup.sh
./run_quick.sh   # fast diagnostic render
./run.sh         # final full-quality render
```

Both run scripts use resumable stages: build the learned scene once, render bounded
frame chunks, then encode the final MP4. Individual stages can also be called
directly:

```bash
python src/pipeline.py --mode full --out outputs --stage build
python src/pipeline.py --mode full --out outputs --stage render --frame-start 0 --frame-end 12
python src/pipeline.py --mode full --out outputs --stage encode
```

## Pipeline

1. Construct an original procedural teacher scene. No external 3D assets are used.
2. Generate high-resolution stylized RGBA material fields for moss, stratified rock,
   foliage, carved stone, engraved gold, wood, crystals, and clouds.
3. Sample the teacher mesh into dense, tangent-aligned anisotropic Gaussians. Their
   scales follow local triangle area, so they cover surfaces instead of appearing as
   isolated points.
4. Fit a compact PyTorch residual material MLP that learns per-splat color, opacity,
   and emissive attributes from the teacher material targets.
5. Export the learned Gaussian scene as a 3DGS-style binary `.ply` and compressed
   `.npz`.
6. Render the fly-through with a custom NumPy surface-splat renderer:
   - Opaque splats use a depth-aware normalized Gaussian surface resolve.
   - Crystal and cloud splats use back-to-front alpha compositing.
   - Crystal emissive attributes produce a separate bloom and local cyan lighting.

Synthetic multi-view teacher images are saved for inspection in
`outputs/gaussian_scene/teacher_views/`.

## Outputs

Full mode writes to `outputs/`. Quick mode writes to `outputs_quick/`.

- `preview.png`: representative image
- `final_gaussian_splat_flythrough.mp4`: final custom-rendered video
- `gaussian_scene/learned_gaussian_splats.ply`: submission-ready 3D content
- `gaussian_scene/learned_gaussian_splats.npz`: full NumPy Gaussian attributes
- `gaussian_scene/neural_material_mlp.pt`: learned material network weights
- `teacher_scene/sky_sanctuary_teacher_scene.obj`: optional teacher-scene diagnostic
- `textures/*.png`: generated stylized RGBA material textures
- `frames/*.png`: resumable full-resolution video frames
- `scene_summary.json`: counts, alpha range, training statistics, and size

Submit only `gaussian_scene/learned_gaussian_splats.ply` as the single 3D content
piece. It is a MeshLab-loadable binary PLY and includes common 3D Gaussian Splatting
fields (`f_dc_*`, `opacity`, `scale_*`, and `rot_*`) plus RGB, alpha, emissive, and
material IDs.

## Contest Compliance

- Representative image: PNG, below 5 MB
- Video: MP4, below 10 seconds and 50 MB
- 3D content: one PLY file, below 100 MB
- Source and data are reproducible through `setup.sh` and `run.sh`
- No commercial software, external 3D assets, paid models, closed-source tools, or
  Blender rendering are used

## Technical Notes

The teacher geometry is intentionally procedural, while the submitted representation
is a learned Gaussian scene. The neural material model preserves the authored
material identity while learning lighting-aware residual color, alpha, and emissive
attributes. Crystals use actual alpha values in the texture, learned scene, PLY
export, and renderer rather than being simulated as opaque cyan objects.

The altar uses explicit staggered masonry courses over a dark grout core, plus a
large high-contrast stone texture and extra per-face Gaussian sampling. This keeps
the brick rhythm readable after splat blending. The roof-and-pillar temple was
removed to give the altar, bridges, and trees more visual breathing room.

The custom renderer is implemented in `src/pipeline.py`. Its opaque pass blends
nearby splats only within the first visible depth layer, which removes the visible
point-cell pattern without washing out surface textures or blending unrelated
geometry together.

## Dependencies and References

The project uses the open-source Python packages NumPy, Pillow, ImageIO,
imageio-ffmpeg, SciPy, PyTorch, and tqdm. No pretrained model, external texture, or
external 3D asset is used. These packages and the course contest guidelines should be
cited in the final write-up:

- Contest guidelines: https://3dml.kaist.ac.kr/3d-rendering-contest/
- NumPy: https://numpy.org/
- Pillow: https://python-pillow.org/
- ImageIO: https://imageio.readthedocs.io/
- SciPy: https://scipy.org/
- PyTorch: https://pytorch.org/
- tqdm: https://tqdm.github.io/
