"""Defines utilities for reading and writing data."""

import itertools
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchvision.transforms.functional as V
from PIL import Image, ImageSequence
from torch import Tensor
from torchvision.transforms import InterpolationMode

VALID_CHANNEL_COUNTS = {1, 3}


def image_as_uint8(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr * 255).round().astype(np.uint8)
    raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")


def make_human_viewable_resolution(
    image: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    trg_res: tuple[int, int] = (250, 250),
) -> Tensor:
    """Resizes image to human-viewable resolution.

    Args:
        image: The image to resize, with shape (C, H, W)
        interpolation: Interpolation mode to use for image resizing
        trg_res: The target image resolution; the image will be reshaped to
            have approximately the same area as an image with this resolution

    Returns:
        The resized image
    """
    width, height = V.get_image_size(image)
    trg_height, trg_width = trg_res
    factor = math.sqrt((trg_height * trg_width) / (height * width))
    new_height, new_width = int(height * factor), int(width * factor)
    return V.resize(image, [new_height, new_width], interpolation)


def _aminmax(t: Tensor) -> tuple[Tensor, Tensor]:
    # `aminmax` isn't supported for MPS tensors, fall back to separate calls.
    minv, maxv = (t.min(), t.max()) if t.is_mps else tuple(t.aminmax())
    return minv, maxv


def standardize_image(
    image: np.ndarray | Tensor,
    *,
    log_key: str | None = None,
    normalize: bool = True,
    keep_resolution: bool = False,
) -> np.ndarray:
    """Converts an arbitrary image to shape (C, H, W).

    Args:
        image: The image tensor to log
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)
        keep_resolution: If set, preserve original image resolution, otherwise
            change image resolution to human-viewable

    Returns:
        The normalized image, with shape (H, W, C)

    Raises:
        ValueError: If the image shape is invalid
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if normalize and image.is_floating_point():
        minv, maxv = _aminmax(image)
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        image = torch.clamp((image.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] in VALID_CHANNEL_COUNTS:
            pass
        elif image.shape[2] in VALID_CHANNEL_COUNTS:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid channel count{'' if log_key is None else f' for {log_key}'}: {image.shape}")
    else:
        raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {image.shape}")

    if not keep_resolution:
        image = make_human_viewable_resolution(image)

    return image.permute(1, 2, 0).detach().cpu().numpy()


def read_gif(in_file: str | Path, *, skip_first_frame: bool = True) -> Iterator[np.ndarray]:
    """Function that reads a GIF and returns a stream of Numpy arrays.

    Args:
        in_file: The path to the input file.
        skip_first_frame: If set, skip the first frame.

    Yields:
        A stream of Numpy arrays with shape (H, W, C).
    """
    gif = Image.open(str(in_file))
    iterator = ImageSequence.Iterator(gif)
    if skip_first_frame:
        next(iterator)
    for frame in iterator:
        yield np.array(frame)


def write_gif(
    itr: Iterator[np.ndarray | Tensor],
    out_file: str | Path,
    *,
    keep_resolution: bool = False,
    fps: int | Fraction = 10,
    loop: bool = False,
    first_frame_zeros: bool = True,
) -> None:
    """Function that writes an GIF from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        keep_resolution: If set, preserve original image resolution, otherwise
            change image resolution to human-viewable.
        fps: Frames per second for the GIF.
        loop: If set, loop the GIF.
        first_frame_zeros: If set, the first frame will be all zeros.
    """

    def to_image(t: np.ndarray | Tensor) -> Image.Image:
        return Image.fromarray(standardize_image(t, keep_resolution=keep_resolution))

    first_frame = standardize_image(next(itr), keep_resolution=keep_resolution)
    first_img = Image.fromarray(np.zeros_like(first_frame) if first_frame_zeros else first_frame)
    first_img.save(
        str(out_file),
        save_all=True,
        append_images=itertools.chain((Image.fromarray(i) for i in (first_frame,)), (to_image(t) for t in itr)),
        duration=int(1000 / fps),  # Number of milliseconds per frame.
        loop=int(loop),
    )
