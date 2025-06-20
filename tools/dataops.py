from typing import List, Union, Optional, Sequence
import numpy as np
import torch
import PIL.Image as Image
from torchvision import transforms

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_tuple_transform_ops(
    resize: Optional[int] = None,
    normalize: bool = True,
    unscale: bool = False
) -> 'TupleCompose':
    """
    Create a composition of image transformation operations.

    Args:
        resize: Optional size to resize images to
        normalize: Whether to normalize using ImageNet mean/std
        unscale: Whether to keep values in original scale or scale to [0,1]

    Returns:
        A TupleCompose object containing the requested transformations
    """
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(TupleNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)

class ToTensorScaled:
    """Convert a RGB PIL Image to a CHW ordered Tensor, scaling values to [0, 1].
    
    This transform handles both PIL Images and existing tensors, applying scaling
    only to new conversions from PIL Images.
    """

    def __call__(self, im: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to an image.

        Args:
            im: Input image (PIL Image or torch.Tensor)

        Returns:
            Transformed image as a torch.Tensor
        """
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        return im

    def __repr__(self) -> str:
        return "ToTensorScaled(./255)"

class TupleToTensorScaled:
    """Apply ToTensorScaled transform to a tuple of images."""

    def __init__(self) -> None:
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple: Sequence[Union[Image.Image, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Transform a tuple of images.

        Args:
            im_tuple: Sequence of input images

        Returns:
            List of transformed images as tensors
        """
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self) -> str:
        return "TupleToTensorScaled(./255)"

class ToTensorUnscaled:
    """Convert a RGB PIL Image to a CHW ordered Tensor without scaling values."""

    def __call__(self, im: Image.Image) -> torch.Tensor:
        """
        Apply the transform to an image.

        Args:
            im: Input PIL Image

        Returns:
            Transformed image as a torch.Tensor
        """
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self) -> str:
        return "ToTensorUnscaled()"

class TupleToTensorUnscaled:
    """Apply ToTensorUnscaled transform to a tuple of images."""

    def __init__(self) -> None:
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple: Sequence[Image.Image]) -> List[torch.Tensor]:
        """
        Transform a tuple of images.

        Args:
            im_tuple: Sequence of input PIL Images

        Returns:
            List of transformed images as tensors
        """
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self) -> str:
        return "TupleToTensorUnscaled()"

class TupleResize:
    """Apply resize transform to a tuple of images."""

    def __init__(self, size: Union[int, tuple], mode: int = Image.BICUBIC) -> None:
        """
        Initialize the resize transform.

        Args:
            size: Target size (int for shorter edge, tuple for exact size)
            mode: Interpolation mode (default: PIL.Image.BICUBIC)
        """
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple: Sequence[Image.Image]) -> List[Image.Image]:
        """
        Resize a tuple of images.

        Args:
            im_tuple: Sequence of input PIL Images

        Returns:
            List of resized images
        """
        return [self.resize(im) for im in im_tuple]

    def __repr__(self) -> str:
        return f"TupleResize(size={self.size})"

class TupleNormalize:
    """Apply normalization transform to a tuple of tensor images."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """
        Initialize the normalization transform.

        Args:
            mean: Sequence of mean values for each channel
            std: Sequence of standard deviation values for each channel
        """
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Normalize a tuple of tensor images.

        Args:
            im_tuple: Sequence of input tensor images

        Returns:
            List of normalized tensor images
        """
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self) -> str:
        return f"TupleNormalize(mean={self.mean}, std={self.std})"

class TupleCompose:
    """Compose multiple transforms into a single transform."""

    def __init__(self, transforms: Sequence[object]) -> None:
        """
        Initialize the composition of transforms.

        Args:
            transforms: Sequence of transform objects to apply
        """
        self.transforms = transforms

    def __call__(self, im_tuple: Sequence[Union[Image.Image, torch.Tensor]]) -> List[Union[Image.Image, torch.Tensor]]:
        """
        Apply all transforms sequentially to a tuple of images.

        Args:
            im_tuple: Sequence of input images

        Returns:
            List of transformed images
        """
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string
