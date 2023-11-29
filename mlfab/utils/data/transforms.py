"""Defines a bunch of dataset transforms."""

import random
from typing import TypeVar

import torch
import torchvision.transforms.functional as V
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

Image = TypeVar("Image", Tensor, PILImage)

NormParams = tuple[float, float, float]

# Default image normalization parameters.
MEAN: NormParams = 0.48145466, 0.4578275, 0.40821073
STD: NormParams = 0.26862954, 0.26130258, 0.27577711


def square_crop(img: Image) -> Image:
    """Crops an image to a square.

    Args:
        img: The input image

    Returns:
        The cropped image, with height and width equal.
    """
    img_width, img_height = V.get_image_size(img)
    height = width = min(img_height, img_width)
    top, left = (img_height - height) // 2, (img_width - width) // 2
    return V.crop(img, top, left, height, width)


def square_resize_crop(img: Image, size: int, interpolation: InterpolationMode = InterpolationMode.NEAREST) -> Image:
    """Resizes an image to a square and then crops it.

    Args:
        img: The input image
        size: The size of the square
        interpolation: The interpolation mode to use

    Returns:
        The cropped image
    """
    img_width, img_height = V.get_image_size(img)
    min_dim = min(img_width, img_height)
    height, width = int((img_width / min_dim) * size), int((img_height / min_dim) * size)
    img = V.resize(img, [height, width], interpolation)
    top, left = (height - size) // 2, (width - size) // 2
    return V.crop(img, top, left, size, size)


def upper_left_crop(img: Image, height: int, width: int) -> Image:
    """Crops an image from the upper left corner.

    This is useful because it preserves camera intrinsics for an image.

    Args:
        img: The input image
        height: The height of the crop
        width: The width of the crop

    Returns:
        The cropped image
    """
    return V.crop(img, 0, 0, height, width)


def normalize(t: Tensor, *, mean: NormParams = MEAN, std: NormParams = STD) -> Tensor:
    """Normalizes an image tensor (by default, using ImageNet parameters).

    This can be paired with :func:`denormalize` to convert an image tensor
    to a normalized tensor for processing by a model.

    Args:
        t: The input tensor
        mean: The mean to subtract
        std: The standard deviation to divide by

    Returns:
        The normalized tensor
    """
    return V.normalize(t, mean, std)


def denormalize(t: Tensor, *, mean: NormParams = MEAN, std: NormParams = STD) -> Tensor:
    """Denormalizes a tensor.

    This can be paired with :func:`normalize` to convert a normalized tensor
    back to the original image for viewing by humans.

    Args:
        t: The input tensor
        mean: The mean to subtract
        std: The standard deviation to divide by

    Returns:
        The denormalized tensor
    """
    mean_tensor = torch.tensor(mean, device=t.device, dtype=t.dtype)
    std_tensor = torch.tensor(std, device=t.device, dtype=t.dtype)
    return (t * std_tensor[None, :, None, None]) + mean_tensor[None, :, None, None]


def random_square_crop(img: Image) -> Image:
    """Randomly crops an image to a square.

    Args:
        img: The input image

    Returns:
        The cropped image
    """
    img_width, img_height = V.get_image_size(img)
    height = width = min(img_height, img_width)
    top, left = random.randint(0, img_height - height), random.randint(0, img_width - width)
    return V.crop(img, top, left, height, width)


def random_square_crop_multi(imgs: list[Image]) -> list[Image]:
    """Randomly crops a list of images to the same size.

    Args:
        imgs: The list of images to crop

    Returns:
        The cropped images
    """
    img_dims = V.get_image_size(imgs[0])
    assert all(V.get_image_size(i) == img_dims for i in imgs[1:])
    img_width, img_height = img_dims
    height = width = min(img_width, img_height)
    top, left = random.randint(0, img_height - height), random.randint(0, img_width - width)
    return [V.crop(i, top, left, height, width) for i in imgs]


def make_size(img: Image, ref_size: tuple[int, int]) -> Image:
    """Converts an image to a specific size, zero-padding smaller dimension.

    Args:
        img: The input image
        ref_size: The reference size, as (width, height)

    Returns:
        The resized image
    """
    img_c, (img_w, img_h), (ref_w, ref_h) = V.get_image_num_channels(img), V.get_image_size(img), ref_size
    if img_h / img_w < ref_h / ref_w:  # Pad width
        new_h, new_w = (img_h * ref_w) // img_w, ref_w
    else:
        new_h, new_w = ref_h, (img_w * ref_h) // img_h
    img = V.resize(img, [new_h, new_w], InterpolationMode.BILINEAR)
    new_img = img.new_zeros(img_c, ref_h, ref_w)
    start_h, start_w = (ref_h - new_h) // 2, (ref_w - new_w) // 2
    new_img[:, start_h : start_h + new_h, start_w : start_w + new_w] = img
    return new_img


def make_same_size(img: Image, ref_img: Image) -> Image:
    """Converts an image to the same size as a reference image.

    Args:
        img: The input image
        ref_img: The reference image

    Returns:
        The input image resized to the same size as the reference image,
            zero-padding dimensions which are too small
    """
    ref_w, ref_h = V.get_image_size(ref_img)
    return make_size(img, (ref_w, ref_h))


def pil_to_tensor(pic: PILImage) -> Tensor:
    """Converts a PIL image to a tensor.

    Args:
        pic: The PIL image.

    Returns:
        The normalized image tensor.
    """
    tensor = V.pil_to_tensor(pic)
    tensor = V.convert_image_dtype(tensor)
    if tensor.shape[0] == 3:
        tensor = normalize(tensor)
    return tensor


class SquareResizeCrop(nn.Module):
    """Resizes and crops an image to a square with the target shape.

    Generally SquareCrop followed by a resize should be preferred when using
    bilinear resize, as it is faster to do the interpolation on the smaller
    image. However, nearest neighbor resize on the larger image followed by
    a crop on the smaller image can sometimes be faster.
    """

    __constants__ = ["size", "interpolation"]

    def __init__(self, size: int, interpolation: InterpolationMode = InterpolationMode.NEAREST) -> None:
        """Initializes the square resize crop.

        Args:
            size: The square height and width to resize to
            interpolation: The interpolation type to use when resizing
        """
        super().__init__()

        self.size = int(size)
        self.interpolation = InterpolationMode(interpolation)

    def forward(self, img: Image) -> Image:
        return square_resize_crop(img, self.size, self.interpolation)


class UpperLeftCrop(nn.Module):
    """Crops image from upper left corner, to preserve image intrinsics."""

    __constants__ = ["height", "width"]

    def __init__(self, height: int, width: int) -> None:
        """Initializes the upper left crop.

        Args:
            height: The max height of the cropped image
            width: The max width of the cropped image
        """
        super().__init__()

        self.height, self.width = height, width

    def forward(self, img: Image) -> Image:
        return upper_left_crop(img, self.height, self.width)
