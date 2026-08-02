# any-image-register
spatial image register

## Principles

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
