# any-image-register
spatial image register

## Principles
feature-pro
all

`any-image-register` aligns spatial omics images from different sources,
modalities, or resolutions into a common coordinate space, so that spatial
barcode / spot coordinates can be mapped onto image pixels for downstream
analysis.

### Registration pipeline

1. **Preprocessing** — grayscale conversion, intensity normalization, and
   construction of an image pyramid for coarse-to-fine alignment.
2. **Correspondence estimation**
   - *Feature-based*: detect and match keypoints (e.g. SIFT-like descriptors),
     then reject outliers with RANSAC.
   - *Intensity-based*: optimize a similarity metric (normalized
     cross-correlation or mutual information) directly over image intensities.
3. **Transform estimation** — fit a transform of increasing flexibility:
   - *rigid* (rotation + translation),
   - *affine* (scaling + shear),
   - *non-rigid / deformable* (e.g. thin-plate spline) for local tissue
     distortion.
4. **Warping & evaluation** — resample the moving image into the fixed image
   space, and quantify accuracy with landmark target registration error (TRE)
   or region overlap (Dice).

### Coordinate mapping

Once the image-to-image transform is known, the same mapping is applied to the
spatial barcode coordinates (e.g. Visium spots or Stereo-seq bins), yielding a
unified image-plus-expression coordinate frame.

## Difficulty index

Registration scenarios ranked by a 1–5 ★ difficulty index. The index combines
four factors: **modality gap**, **deformation type**, **resolution gap**, and
**tissue artifacts** (folds, tears, bubbles).

| Scenario | Transform typically needed | Difficulty |
| --- | --- | --- |
| Same modality, same section re-imaged | rigid | ★ |
| Same modality, serial sections | affine | ★★ |
| Cross-scale (low-res overview ↔ high-res tiles) | affine + pyramid | ★★★ |
| Cross-modality (H&E ↔ fluorescence / IF) | affine + intensity metrics (MI) | ★★★★ |
| Cross-modality + deformable (FFPE vs fresh-frozen, torn or folded tissue) | non-rigid (TPS / deformation field) | ★★★★★ |

Rule of thumb: every additional factor beyond a plain rigid, same-modality
alignment adds roughly one ★.

## Why alignment matters

Registration puts molecular measurements back into their morphological
context. Without a shared coordinate frame, expression data and tissue images
remain parallel descriptions of the same sample; with one, every spot or cell
inherits both a molecular profile and a histological identity.

### Value

- **Morphological grounding** — annotate expression clusters with the tissue
  structures they actually occupy (tumor vs. stroma vs. immune infiltrate).
- **Multimodal integration** — combine transcriptomics, proteomics, and
  histology of the same or adjacent sections into one joint analysis.
- **3D reconstruction** — stack registered serial sections into volumetric
  tissue atlases.
- **Quality control** — verify that barcoded spots truly overlap the tissue
  and flag folds, tears, or off-tissue capture areas.

### Applications

- Mapping Visium / Stereo-seq barcode coordinates onto H&E or
  immunofluorescence images of the same section.
- Aligning consecutive sections assayed with different modalities
  (transcriptome on one, multiplex IF or proteomics on the next).
- Building cross-sample and cross-cohort spatial atlases for biomarker
  discovery.
- Computational pathology: aligning whole-slide images across stains for
  weakly supervised learning and label transfer.


## The `register` package

The repository ships a numpy-only reference implementation of the
pipeline described above:

| Module | Contents |
| --- | --- |
| `register.io` | image loading and grayscale conversion |
| `register.preprocess` | intensity normalisation, Gaussian smoothing |
| `register.pyramid` | Gaussian image pyramid for coarse-to-fine alignment |
| `register.metrics` | NCC and mutual information similarity |
| `register.metrics_fast` | vectorised shift-grid search (performance path) |
| `register.pyramid_cache` | LRU-cached pyramid construction |
| `register.transforms` | rigid (Kabsch) and affine least-squares estimation |
| `register.ransac` | RANSAC outlier rejection for correspondences |
| `register.warp` | bilinear image warping into fixed space |
| `register.coords` | spatial barcode / spot coordinate mapping |
| `register.evaluate` | landmark TRE and Dice overlap evaluation |
| `register.difficulty` | the difficulty index as executable code |
| `register.cli` | command-line entry points |

### Quickstart

```python

import numpy as np
from register.metrics import ncc
from register.ransac import ransac_affine
from register.warp import warp_image

# src/dst: matched keypoint coordinates from your favourite detector
matrix, inliers = ransac_affine(src, dst, threshold=3.0)
aligned = warp_image(moving_img, matrix, output_shape=fixed_img.shape)
print("NCC after alignment:", ncc(fixed_img, aligned))
...
```

Run the tests with `python -m pytest tests/`.
