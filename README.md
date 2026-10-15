# ComfyUI-AHP_ImageCropByMask

**AHP Image Crop By Mask Batch** crops each image to the area covered by its mask. It keeps the mask's soft edges, so the crop stays aligned and clean. It is based on `ImageCropByMaskBatch` from ComfyUI-KJNodes, with fixes to how the mask is resized and used.

- Node name in ComfyUI: **AHP Image Crop By Mask Batch**
- Category: `AHP/image`
- Author: **Ali Haghighat panah**
- License: GPL-3.0 (see [LICENSE](LICENSE))

---

## How it works

1. **Find the crop box.** The bounding box of each mask is found from pixels above `mask_threshold`, then `padding` is added around it.
2. **Crop.** The image and the mask are cut with the same box.
3. **Scale (if needed).** If the crop doesn't fit `width` × `height`, or `preserve_size` is off, both are scaled with the same size and alignment. The image uses bicubic resampling and the mask uses bilinear. Neither is rounded, so the mask keeps its soft edges.
4. **Center on the canvas.** The result is placed on a `width` × `height` canvas. Empty areas get mask value 0.
5. **Blend (optional).** If `apply_mask` is on, the image is blended with `bg_color` using the mask. The soft edge becomes smooth transparency instead of a jagged cut.

The output mask keeps its soft values. Nothing is rounded to 0/1.

## Inputs

### Required

| Input | Default | What it does |
|---|---|---|
| `image` | — | Images to crop. Use one image for all masks, or one image per mask |
| `masks` | — | One mask per output crop |
| `width` | 512 | Canvas width in pixels |
| `height` | 512 | Canvas height in pixels |
| `padding` | 0 | Extra pixels around the mask bounding box |
| `preserve_size` | off | If the crop fits the canvas, keep its original size instead of scaling it up |
| `bg_color` | `0, 0, 0` | Background color. Accepts `255, 0, 0`, `1.0, 0, 0`, `#ff0000`, or `red` |

### Optional

| Input | Default | What it does |
|---|---|---|
| `mask_threshold` | 0.0 | Mask values above this count as the object when finding the crop box. Raise it to ignore faint noise |
| `apply_mask` | on | Blend the image with `bg_color` using the mask. On: soft edges become smooth transparency. Off: the image is left as it is |

## What's new compared to the original KJNodes ImageCropByMaskBatch

| Area | Original KJNodes | This version |
|---|---|---|
| Mask resize | Nearest-neighbor, which makes jagged, hard edges | Bilinear with antialiasing, which keeps soft edges |
| Image and mask alignment | Image and mask resized with different methods | Same geometry and resampling for both |
| Mask values | Rounded to 0/1 by nearest-neighbor | Kept as soft values |
| Background blend | Hard mask | Soft mask, so edges blend smoothly |
| Batch handling | Always used `image[0]` for every mask | Uses `image[i]` for mask `i` when the batches match |
| Empty mask result | Returned one tensor where two outputs were declared, which crashes | Returns two empty tensors with the correct shapes |
| Device | Some new tensors were created on CPU, which fails on GPU | All tensors stay on the input's device |
| Zero width or height | Division by zero | Clamped to at least 1 |
| Crop box | Any value above 0 counts, so faint noise expands the crop | `mask_threshold` lets you ignore noise |
| Color parsing | Depended on an external helper | Built-in parser with RGB, 0.0–1.0, hex, and common names |
| Alpha channel | Not handled | RGBA images work, and the background is blended only into RGB |

## Tips

- If the crop is pulled in by faint noise, raise `mask_threshold` to 0.05–0.2.
- Use `preserve_size` when you want the original pixel scale. Leave it off to fill the whole canvas.
- For a transparent-looking cut-out, turn `apply_mask` on and set `bg_color` to the color you want behind the object.
- For a hard cut-out, feed the mask through the AHP Separate Masks node first, then use its 0/1 output here.

## Installation

Clone into your `ComfyUI/custom_nodes` folder. No extra packages are needed:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-AHP_ImageCropByMask.git
```

Restart ComfyUI. Double-click the canvas and search for **AHP Image Crop By Mask Batch**.

## Running the tests

```bash
cd ComfyUI-AHP_ImageCropByMask
python tests/test_ahp_image_crop_by_mask_batch.py
```

The tests need `torch`. They run without ComfyUI.

## Credits

- **Author and modifications:** Ali Haghighat panah
- **Original code:** `ImageCropByMaskBatch` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (`nodes/image_nodes.py`) by kijai, licensed under GPL-3.0
- **Development assistance:** Claude (Anthropic)

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. Because it's based on GPL-3.0 code, the modified version must also be GPL-3.0. The full text is in [LICENSE](LICENSE).
