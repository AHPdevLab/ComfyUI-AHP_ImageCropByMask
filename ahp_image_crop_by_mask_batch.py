# AHP Image Crop By Mask Batch
# Copyright (C) 2026 Ali Haghighat panah
#
# Modified from ImageCropByMaskBatch in ComfyUI-KJNodes/nodes/image_nodes.py
# (https://github.com/kijai/ComfyUI-KJNodes), copyright kijai, licensed under GPL-3.0.
# All changes were made by Ali Haghighat panah. Development was assisted by Claude (Anthropic).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See LICENSE.

import re
from inspect import cleandoc

import torch
import torch.nn.functional as F

MAX_RESOLUTION = 16384

NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def string_to_color(text):
    """
    Parse a background color. Accepts:
      "255, 0, 0"   (0-255 integers)
      "1.0, 0, 0"   (0.0-1.0 floats, detected by a decimal point)
      "#ff0000"     (hex)
      "red"         (a few common names)
    Returns a list of three ints in 0-255.
    """
    t = text.strip().lower()
    if t in NAMED_COLORS:
        return list(NAMED_COLORS[t])
    if t.startswith("#") and len(t) == 7:
        return [int(t[i:i + 2], 16) for i in (1, 3, 5)]
    parts = [p for p in re.split(r"[,\s]+", t) if p]
    if len(parts) != 3:
        raise ValueError(f"bg_color must have three values, got: {text!r}")
    values = [float(p) for p in parts]
    if any("." in p for p in parts) and all(0.0 <= v <= 1.0 for v in values):
        values = [v * 255.0 for v in values]
    return [int(round(min(255.0, max(0.0, v)))) for v in values]


class AHP_ImageCropByMaskBatch:
    """
    Crops each image to the area covered by its mask, keeping the mask's soft edges.

    The mask and the image are resized with the same geometry, so they stay aligned.
    The mask is never rounded to 0/1 or resized with nearest-neighbor, so its edges stay smooth.

    Processing order
    ----------------
    1. Find the mask's bounding box, using mask_threshold, and add padding.
    2. Crop the image and the mask with the same box.
    3. If needed, scale both to fit width x height (preserve_size controls this).
    4. Center them on a width x height canvas.
    5. Optionally blend the image with bg_color using the mask (apply_mask).
       With transparent_background on, step 5 is replaced: the output is RGBA, alpha is the
       mask, and every pixel outside the mask is cleared (RGB and alpha both 0).

    Output
    ------
    images: RGB batch with the canvas size.
    masks:  float mask batch with the same canvas size. Soft edges are kept.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. Here it is "crop".
    CATEGORY (`str`):
        The category the node should appear in the UI.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Images to crop. Use one image for all masks, or one image per mask."}),
                "masks": ("MASK", {"tooltip": "One mask per output crop. Soft edges are kept."}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,
                                  "tooltip": "Canvas width in pixels."}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Canvas height in pixels."}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                                    "tooltip": "Extra pixels added around the mask bounding box."}),
                "preserve_size": ("BOOLEAN", {"default": False,
                                              "tooltip": "If the crop fits the canvas, keep its original size instead of scaling it up."}),
                "bg_color": ("STRING", {"default": "0, 0, 0",
                                        "tooltip": "Color as RGB values in range 0-255 or 0.0-1.0, or color name or hex code"}),
            },
            "optional": {
                "mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "round": 0.001, "display": "number",
                                             "tooltip": "Mask values above this count as the object when finding the crop box. Raise it to ignore faint noise."}),
                "apply_mask": ("BOOLEAN", {"default": True,
                                           "tooltip": "Blend the image with bg_color using the mask. Soft edges become smooth transparency."}),
                "transparent_background": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Output is RGBA. Alpha is the mask, and everything outside the mask is cleared, so other objects in the crop box are removed. Ignores bg_color and apply_mask."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "crop"
    CATEGORY = "AHP/image"

    def crop(self, image, masks, width, height, bg_color, padding, preserve_size,
             mask_threshold=0.0, apply_mask=True, transparent_background=False):
        B, H, W, C = image.shape
        width = max(1, int(width))
        height = max(1, int(height))
        device = image.device

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        BM = masks.shape[0]
        if B != 1 and B != BM:
            raise ValueError(f"image batch ({B}) must be 1 or match the mask batch ({BM})")

        masks = masks.detach().float().clamp(0.0, 1.0).to(device)
        if masks.shape[1] != H or masks.shape[2] != W:
            # Antialiased resize keeps the soft edge. Nearest-neighbor would make it jagged.
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode="bilinear",
                                  align_corners=False, antialias=True).squeeze(1).clamp(0.0, 1.0)

        color = torch.tensor([v / 255.0 for v in string_to_color(bg_color)],
                             dtype=torch.float32, device=device)

        out_rgb = []
        out_masks = []

        for i in range(BM):
            src = i if B == BM else 0
            m = masks[i]

            ys, xs = torch.nonzero(m > mask_threshold, as_tuple=True)
            if ys.numel() == 0:
                continue

            min_y = max(0, int(ys.min()) - padding)
            max_y = min(H, int(ys.max()) + 1 + padding)
            min_x = max(0, int(xs.min()) - padding)
            max_x = min(W, int(xs.max()) + 1 + padding)

            img = image[src, min_y:max_y, min_x:max_x, :].float()
            msk = m[min_y:max_y, min_x:max_x]

            crop_h, crop_w = img.shape[0], img.shape[1]
            new_w, new_h = crop_w, crop_h

            if not preserve_size or crop_w > width or crop_h > height:
                scale = min(width / crop_w, height / crop_h)
                new_w = max(1, int(crop_w * scale))
                new_h = max(1, int(crop_h * scale))

                # Same target size and alignment for image and mask.
                img = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
                                    mode="bicubic", align_corners=False, antialias=True)
                img = img.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
                msk = F.interpolate(msk[None, None], size=(new_h, new_w),
                                    mode="bilinear", align_corners=False, antialias=True)
                msk = msk[0, 0].clamp(0.0, 1.0)

            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2

            canvas_img = torch.zeros((height, width, C), dtype=torch.float32, device=device)
            canvas_msk = torch.zeros((height, width), dtype=torch.float32, device=device)
            canvas_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = img[:, :, :C]
            canvas_msk[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = msk

            if transparent_background:
                # Clear everything outside the mask, so other objects in the crop box are removed.
                # Soft edge pixels keep their own color; alpha carries the soft edge.
                inside = (canvas_msk > 0).float().unsqueeze(-1)
                rgb = canvas_img[..., :3] * inside
                canvas_img = torch.cat([rgb, canvas_msk.unsqueeze(-1)], dim=-1)
            elif apply_mask:
                a = canvas_msk.unsqueeze(-1)
                rgb = canvas_img[..., :3]
                canvas_img[..., :3] = rgb * a + color * (1.0 - a)

            out_rgb.append(canvas_img.to(image.dtype))
            out_masks.append(canvas_msk)

        if not out_rgb:
            out_c = 4 if transparent_background else C
            return (torch.zeros((0, height, width, out_c), dtype=image.dtype, device=device),
                    torch.zeros((0, height, width), dtype=torch.float32, device=device))

        return (torch.stack(out_rgb, dim=0), torch.stack(out_masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "AHP_ImageCropByMaskBatch": AHP_ImageCropByMaskBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AHP_ImageCropByMaskBatch": "AHP Image Crop By Mask Batch",
}
