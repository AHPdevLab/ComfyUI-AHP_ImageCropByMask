"""
Run from the package folder:   python tests/test_ahp_image_crop_by_mask_batch.py
or with pytest:                python -m pytest tests -q
"""
import importlib.util
import os
import sys

import numpy as np
import torch

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "ahp_crop_pkg", os.path.join(PKG_DIR, "ahp_image_crop_by_mask_batch.py")
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

NODE = mod.AHP_ImageCropByMaskBatch()


def crop(image, masks, width=256, height=256, bg="0, 0, 0", padding=0,
         preserve_size=False, mask_threshold=0.0, apply_mask=True, transparent_background=False):
    return NODE.crop(image, masks, width, height, bg, padding, preserve_size,
                     mask_threshold=mask_threshold, apply_mask=apply_mask,
                     transparent_background=transparent_background)


def circle_mask(H, W, cx, cy, r):
    """Soft-edged disc: values fall off over about one pixel at the boundary."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return torch.from_numpy(np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32))[None]


def fractional_count(t):
    return int(((t > 0) & (t < 1)).sum())


# ---------------------------------------------------------------- registration

def test_registration():
    assert "AHP_ImageCropByMaskBatch" in mod.NODE_CLASS_MAPPINGS
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["AHP_ImageCropByMaskBatch"] == "AHP Image Crop By Mask Batch"
    assert NODE.CATEGORY == "AHP/image"
    assert NODE.RETURN_TYPES == ("IMAGE", "MASK")
    inputs = NODE.INPUT_TYPES()
    for name in ("image", "masks", "width", "height", "padding", "preserve_size", "bg_color"):
        assert name in inputs["required"]
    assert "mask_threshold" in inputs["optional"] and "apply_mask" in inputs["optional"]
    assert inputs["optional"]["transparent_background"][1]["default"] is False


# ---------------------------------------------------------------- soft edge preservation

def test_no_resize_keeps_mask_values_exactly():
    img = torch.rand(1, 128, 128, 3)
    m = circle_mask(128, 128, 64, 64, 20)
    out_img, out_m = crop(img, m, width=256, height=256, preserve_size=True, apply_mask=False)
    ys, xs = torch.nonzero(m[0] > 0, as_tuple=True)
    ch, cw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    py, px = (256 - ch) // 2, (256 - cw) // 2
    region = out_m[0, py:py + ch, px:px + cw]
    assert torch.equal(region, m[0, ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def test_resize_keeps_soft_edge_values():
    img = torch.rand(1, 256, 256, 3)
    m = circle_mask(256, 256, 128, 128, 100)
    _, out_m = crop(img, m, width=64, height=64, preserve_size=False, apply_mask=False)
    assert fractional_count(out_m[0]) > 0, "resize turned the soft edge into hard 0/1 values"
    assert out_m.min().item() >= 0.0 and out_m.max().item() <= 1.0


def test_resize_does_not_create_hard_only_edges():
    img = torch.rand(1, 256, 256, 3)
    m = circle_mask(256, 256, 128, 128, 100)
    _, out_m = crop(img, m, width=64, height=64, apply_mask=False)
    values = out_m[0][out_m[0] > 0]
    assert ((values > 0.05) & (values < 0.95)).any()


def test_mask_threshold_ignores_faint_noise():
    img = torch.rand(1, 64, 64, 3)
    m = torch.zeros(1, 64, 64)
    m[0, 20:30, 20:30] = 1.0
    m[0, 0, 0] = 0.01  # faint speck far away
    _, out_m = crop(img, m, width=64, height=64, preserve_size=True, mask_threshold=0.1, apply_mask=False)
    ys, xs = torch.nonzero(out_m[0] > 0, as_tuple=True)
    assert ys.numel() == 100 and ys.min() > 0  # only the 10x10 square, no speck


# ---------------------------------------------------------------- image and mask alignment

def test_image_and_mask_share_geometry():
    img = torch.zeros(1, 128, 128, 3)
    img[0, 40:90, 40:90, :] = 1.0  # white square
    m = torch.zeros(1, 128, 128)
    m[0, 40:90, 40:90] = 1.0
    out_img, out_m = crop(img, m, width=50, height=50, apply_mask=False)
    inside = out_img[0][out_m[0] > 0.9]
    assert torch.allclose(inside, torch.ones_like(inside), atol=0.05)


# ---------------------------------------------------------------- batching

def test_batch_uses_matching_image_for_each_mask():
    img = torch.zeros(2, 64, 64, 3)
    img[0] = 0.2
    img[1] = 0.8
    m = torch.zeros(2, 64, 64)
    m[0, 5:20, 5:20] = 1.0
    m[1, 30:50, 30:50] = 1.0
    out_img, out_m = crop(img, m, width=64, height=64, preserve_size=True, apply_mask=False)
    assert out_img.shape == (2, 64, 64, 3)
    assert torch.allclose(out_img[0][out_m[0] > 0.5], torch.full((1,), 0.2).expand(int((out_m[0] > 0.5).sum()), 3))
    assert torch.allclose(out_img[1][out_m[1] > 0.5], torch.full((1,), 0.8).expand(int((out_m[1] > 0.5).sum()), 3))


def test_single_image_is_reused_for_all_masks():
    img = torch.rand(1, 64, 64, 3)
    m = torch.zeros(3, 64, 64)
    m[0, 0:10, 0:10] = 1
    m[1, 20:30, 20:30] = 1
    m[2, 40:50, 40:50] = 1
    out_img, out_m = crop(img, m, width=32, height=32)
    assert out_img.shape[0] == 3


def test_mismatched_image_batch_raises():
    img = torch.rand(2, 64, 64, 3)
    m = torch.ones(3, 64, 64)
    try:
        crop(img, m)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched batch sizes")


# ---------------------------------------------------------------- compositing

def test_apply_mask_uses_background_outside_mask():
    img = torch.ones(1, 64, 64, 3)
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, bg="255, 0, 0", apply_mask=True)
    assert torch.allclose(out_img[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_apply_mask_off_leaves_image_untouched():
    img = torch.full((1, 64, 64, 3), 0.5)
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, bg="255, 0, 0", apply_mask=False)
    assert out_img[0, 32, 32].tolist() == [0.5, 0.5, 0.5]  # center of the 20x20 crop


def test_transparent_background_outputs_rgba_with_mask_as_alpha():
    img = torch.rand(1, 64, 64, 3)
    m = circle_mask(64, 64, 32, 32, 15)
    out_img, out_m = crop(img, m, width=64, height=64, preserve_size=True, transparent_background=True)
    assert out_img.shape == (1, 64, 64, 4)
    assert torch.allclose(out_img[0, ..., 3], out_m[0])
    assert fractional_count(out_img[0, ..., 3]) > 0


def test_transparent_background_clears_other_objects_in_crop_box():
    img = torch.zeros(1, 64, 64, 3)
    img[0, 10:30, 10:30, :] = torch.tensor([0.2, 0.4, 0.6])  # the masked object
    img[0, 10:30, 40:55, :] = 0.9                             # a different object, not in the mask
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0   # masked object
    m[0, 50:55, 10:55] = 1.0   # a strip that stretches the crop box over the second object
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=False, transparent_background=True)
    # The second object (x 40..55, y 10..30) is inside the crop box but not in the mask. It must be cleared.
    other_region = out_img[0, 0:20, 30:45]
    assert torch.all(other_region[..., 3] == 0)
    assert torch.all(other_region[..., :3] == 0)


def test_transparent_background_keeps_object_colors_and_alpha():
    img = torch.zeros(1, 64, 64, 3)
    img[0, 10:30, 10:30, :] = torch.tensor([0.2, 0.4, 0.6])
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, transparent_background=True)
    assert torch.allclose(out_img[0, 32, 32], torch.tensor([0.2, 0.4, 0.6, 1.0]), atol=1e-5)
    assert out_img[0, 0, 0, 3].item() == 0.0


def test_transparent_background_overrides_apply_mask_and_bg_color():
    img = torch.full((1, 64, 64, 3), 0.5)
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, bg="255, 0, 0",
                      apply_mask=True, transparent_background=True)
    assert out_img[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]  # no red background
    assert out_img[0, 32, 32].tolist() == [0.5, 0.5, 0.5, 1.0]


def test_soft_edge_blends_with_background():
    img = torch.ones(1, 64, 64, 3)
    m = circle_mask(64, 64, 32, 32, 15)
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, bg="0, 0, 0", apply_mask=True)
    blended = out_img[0][..., 0]
    assert ((blended > 0.05) & (blended < 0.95)).any()


# ---------------------------------------------------------------- padding and empty input

def test_padding_adds_margin():
    img = torch.rand(1, 64, 64, 3)
    m = torch.zeros(1, 64, 64)
    m[0, 20:30, 20:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True, padding=5, apply_mask=False)
    ys, xs = torch.nonzero(out_img[0, ..., 0] > 0, as_tuple=True)  # image region incl. padding
    assert ys.numel() == 20 * 20


def test_empty_mask_returns_zero_size_outputs():
    img = torch.rand(1, 64, 64, 3)
    m = torch.zeros(1, 64, 64)
    out_img, out_m = crop(img, m, width=32, height=32)
    assert out_img.shape == (0, 32, 32, 3)
    assert out_m.shape == (0, 32, 32)


def test_mask_size_different_from_image_is_resampled():
    img = torch.rand(1, 128, 128, 3)
    m = circle_mask(64, 64, 32, 32, 15)
    _, out_m = crop(img, m, width=64, height=64, apply_mask=False)
    assert out_m.shape == (1, 64, 64)
    assert fractional_count(out_m[0]) > 0
    assert out_m.min().item() >= 0.0 and out_m.max().item() <= 1.0


def test_rgba_input_does_not_crash():
    img = torch.rand(1, 64, 64, 4)
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    out_img, _ = crop(img, m, width=64, height=64, preserve_size=True)
    assert out_img.shape == (1, 64, 64, 4)


# ---------------------------------------------------------------- color parsing

def test_string_to_color_formats():
    assert mod.string_to_color("255, 0, 0") == [255, 0, 0]
    assert mod.string_to_color("#00ff00") == [0, 255, 0]
    assert mod.string_to_color("red") == [255, 0, 0]
    assert mod.string_to_color("1.0, 0.0, 0.5") == [255, 0, 128]


def test_string_to_color_rejects_bad_input():
    try:
        mod.string_to_color("1, 2")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
