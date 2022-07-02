from tkinter import image_types
import numpy as np
import kornia
import torch
import torchvision.transforms.functional as TF

from torch import Tensor
from torchvision import transforms
from collections import OrderedDict


def SceneClassificationToColorTransform(scene_classification):
    """ Converts scene classification labels to a color labels.
    See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/

    Mapping from scene classification to color:
    0:  No Data -> black
    1:  Saturated / Defective -> red
    2:  Dark Area Pixels -> gray
    3:  Cloud Shadows -> brown
    4:  Vegetation -> green
    5:  Bare Soils -> yellow
    6:  Water -> blue
    7:  Clouds low probability / Unclassified -> medium gray
    8:  Clouds medium probability -> light gray
    9:  Clouds high probability -> very light gray
    10: Cirrus -> light blue/purple
    11: Snow / Ice -> cyan

    Parameters
    ----------
    scene_classification : Tensor
        Scene classification labels.

    Returns
    -------
    Tensor
        Color scene classification labels.
    """

    """ TODO: Don't have to use OrderedDict if Python >= 3.7.
    Since 3.6 Python respects insertion order and with 3.7 it is guaranteed.
    The environment and the dependencies do require >= 3.6 anyways...
    
    See: 
    https://docs.python.org/2/library/stdtypes.html#dict.items
    https://docs.python.org/3.7/library/stdtypes.html#dict
    """

    CLASSES = OrderedDict(
        {
            "No Data": 0,
            "Saturated / Defective": 1,
            "Dark Area Pixels": 2,
            "Cloud Shadows": 3,
            "Vegetation": 4,
            "Bare Soils": 5,
            "Water": 6,
            "Clouds low probability / Unclassified": 7,
            "Clouds medium probability": 8,
            "Clouds high probability": 9,
            "Cirrus": 10,
            "Snow / Ice": 11,
        }
    )

    COLORS = OrderedDict(
        {
            "black": [0, 0, 0],
            "red": [1, 0, 0.016],
            "gray": [0.525, 0.525, 0.525],
            "brown": [0.467, 0.298, 0.043],
            "green": [0.063, 0.827, 0.176],
            "yellow": [1, 1, 0.325],
            "blue": [0.0, 0.0, 1.0],
            "medium gray": [0.506, 0.506, 0.506],
            "light gray": [0.753, 0.753, 0.753],
            "very light gray": [0.949, 0.949, 0.949],
            "light blue/purple": [0.733, 0.773, 0.925],
            "cyan": [0.325, 1, 0.980],
        }
    )
    color_scene_classification = torch.tensor(0)
    # Scene classification has multiple revisits: (revisits, 1, height, width)
    if scene_classification.ndim == 4:

        # Runs the function for each revisit.
        # Output shape: (revisits, 3 [RGB], height, width)
        color_scene_classification = torch.stack(
            [SceneClassificationToColorTransform(x[0]) for x in scene_classification]
        )

    # Scene classification has only a single revisit: (1, height, width)
    elif scene_classification.ndim == 3:
        # Output shape: (3 [RGB], height, width)
        color_scene_classification = scene_classification[0]
        height, width = scene_classification.shape
        color_scene_classification = torch.zeros((3, height, width))

        for scene_class, color in zip(CLASSES.values(), COLORS.values()):
            color_scene_classification[
                :, scene_classification == scene_class
            ] = torch.Tensor([color]).T

    return color_scene_classification


class NormalizeInverse(transforms.Normalize):
    """ Inverse of Normalize.
    Undoes the normalization and returns the reconstructed images in the input domain.
    Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    """

    def __init__(self, mean, std):
        """ Initializes the NormalizeInverse class.

        Parameters
        ----------
        mean : Tensor
            Mean of the normalization.
        std : Tensor
            Standard deviation of the normalization.
        """
        std_inverse = 1 / (std + 1e-7)
        mean_inverse = -mean * std_inverse
        super().__init__(mean=mean_inverse, std=std_inverse)

    def __call__(self, tensor):
        """ Applies the inverse normalization.

        Parameters
        ----------
        tensor : Tensor
            Tensor to be un-normalized.

        Returns
        -------
        Tensor
            Un-normalized tensor.
        """
        return super().__call__(tensor.clone())


class RandomShuffleRevisitsDict:
    """ Randomly shuffles the revisits of a dictionary. """

    def __init__(self, src="lr"):
        """ Initializes the RandomShuffleRevisitsDict class.

        Parameters
        ----------
        src : str, optional
            Source of the dictionary, by default 'lr'.
        """
        self.src = src

    # Disable gradients for effiency.
    @torch.no_grad()
    def __call__(self, item):
        """ Applies the RandomShuffleRevisitsDict class.

        Parameters
        ----------
        item : dict of Tensor
            Dictionary with the data to be shuffled.

        Returns
        -------
        dict of Tensor
            Shuffled dictionary.
        """
        number_of_revisits = item[self.src].shape[-4]
        random_shuffle_ix = torch.randperm(number_of_revisits)
        for image_type, revisit in item.items():
            if image_type != "hr":
                item[image_type] = revisit[random_shuffle_ix].clone()
        return item


class CropDict:
    """ Crops the dictionary items, while keeping the cropping proportional
    to the original/source image. """

    def __init__(self, start_x, start_y, end_x, end_y, src="lr"):
        """ Initializes the CropDict class.

        Parameters
        ----------
        start_x : int
            Starting x-coordinate of the crop.
        start_y : int
            Starting y-coordinate of the crop.
        end_x : int
            Ending x-coordinate of the crop.
        end_y : int
            Ending y-coordinate of the crop.
        src : str, optional
            Source of the original/source image, by default 'lr'.
        """

        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.src = src

    def crop(self, item):
        """ Applies the crop to the dictionary item.

        Parameters
        ----------
        item : union of dict of Tensor
            Dictionary item to be cropped.

        Returns
        -------
        union of dict of Tensor
            Cropped dictionary item.
        """
        start_x, start_y, end_x, end_y = (
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
        )
        if isinstance(item, dict):
            source_img = item[self.src]
            source_height, source_width = source_img.shape[-2:]
            for image_type, image in item.items():
                if isinstance(image, Tensor):
                    image_height, image_width = image.shape[-2:]

                    ratio_height, ratio_width = (
                        image_height / source_height,
                        image_width / source_width,
                    )

                    crop_start_x = round(start_x * ratio_width)
                    crop_start_y = round(start_y * ratio_height)
                    crop_end_x = round(end_x * ratio_width)
                    crop_end_y = round(end_y * ratio_height)

                    item[image_type] = TF.crop(
                        img=image,
                        top=crop_start_y,
                        left=crop_start_x,
                        height=crop_end_x,
                        width=crop_end_y,
                    ).clone()

        elif isinstance(item, Tensor):
            item = TF.crop(item, start_y, start_x, end_y, end_x).clone()
        return item

    # Disable gradients for effiency.
    @torch.no_grad()
    def __call__(self, item):
        """ Applies the CropDict class.

        Parameters
        ----------
        item : dict
            Dictionary with the data to be cropped.

        Returns
        -------
        dict
            Cropped dictionary.
        """
        return self.crop(item)


class RandomCropDict:
    """ Randomly crops the dictionary items, while keeping the cropping 
    proportional to the original/source image. """

    def __init__(self, src, size, batched=False):
        """ Initializes the RandomCropDict class.

        Parameters
        ----------
        src : str
            Source of the original/source image.
        size : tuple of int
            Size of the crop.
        batched : bool, optional
            Whether the data is batched, by default False.
        """
        self.src = src
        self.size = size
        self.batched = batched

    def random_crop(self, item):
        """ Applies the random crop to the dictionary item.

        Parameters
        ----------
        item : union of Tensor and dict
            Dictionary item to be cropped.

        Returns
        -------
        union of Tensor and dict
            Cropped dictionary item.
        """
        if isinstance(item, dict):
            source_image = item[self.src]
            height, width = source_image.shape[-2:]
            random_crop = transforms.RandomCrop.get_params(
                source_image, output_size=self.size
            )

            source_crop_start_y, source_crop_start_x = random_crop[:2]
            source_crop_end_y, source_crop_end_x = random_crop[2:]

            for image_type, image in item.items():
                if isinstance(image, Tensor):

                    if self.batched and image.ndim == 5:
                        batch_size, revisits = image.shape[:2]
                        image = image.flatten(start_dim=0, end_dim=1)

                    image_height, image_width = image.shape[-2:]
                    ratio_height, ratio_width = (
                        image_height / height,
                        image_width / width,
                    )

                    crop_start_y = round(source_crop_start_y * ratio_height)
                    crop_start_x = round(source_crop_start_x * ratio_width)
                    crop_end_y = round(source_crop_end_y * ratio_height)
                    crop_end_x = round(source_crop_end_x * ratio_width)

                    item[image_type] = TF.crop(
                        img=image,
                        top=crop_start_y,
                        left=crop_start_x,
                        height=crop_end_x,
                        width=crop_end_y,
                    ).clone()

                    if self.batched:
                        item[image_type] = item[image_type].unflatten(
                            dim=0, sizes=(batch_size, revisits)
                        )

        elif isinstance(item, Tensor):
            random_crop = transforms.RandomCrop.get_params(
                source_image, output_size=self.size
            )

            source_crop_start_y, source_crop_start_x = random_crop[:2]
            source_crop_end_y, source_crop_end_x = random_crop[2:]

            item[image_type] = TF.crop(
                img=item,
                top=crop_start_y,
                left=crop_start_x,
                height=crop_end_x,
                width=crop_end_y,
            ).clone()

        return item

    @torch.no_grad()
    def __call__(self, item: dict) -> dict:
        """ Applies the RandomCropDict class.

        Parameters
        ----------
        item : dict
            Dictionary with the data to be cropped.

        Returns
        -------
        dict
            Cropped dictionary.
        """
        return self.random_crop(item)


class RandomRotateFlipDict:
    """ Randomly rotates by one of the given angles 
    and flips the dictionary items horizontally or vertically. """

    def __init__(self, angles, batched: bool = False):
        """ Initializes the RandomRotateFlipDict class.

        Parameters
        ----------
        angles : list of int
            List of angles to be rotated by.
        batched : bool, optional
            Whether the data is batched, by default False.
        """
        self.angles = angles
        self.batched = batched

    # Disable gradients for efficiency.
    @torch.no_grad()
    def __call__(self, item):
        """ Applies the RandomRotateFlipDict class.

        Parameters
        ----------
        item : dict
            Dictionary with the data to be rotated and flipped.

        Returns
        -------
        dict
            Rotated and flipped dictionary.
        """
        random_angle_index = torch.randint(4, size=(1,)).item()
        angle = self.angles[random_angle_index]
        flip = torch.randint(2, size=(1,)).item()
        if isinstance(item, dict):
            for image_type, image in item.items():
                if isinstance(image, Tensor):
                    if self.batched and image.ndim == 5:
                        batch_size, revisits = image.shape[:2]
                        image = image.flatten(start_dim=0, end_dim=1)
                    item[image_type] = TF.rotate(image, angle)
                    if flip:
                        item[image_type] = TF.vflip(item[image_type])
                    if self.batched:
                        item[image_type] = item[image_type].unflatten(
                            dim=0, sizes=(batch_size, revisits)
                        )

        elif isinstance(item, Tensor):
            item = TF.rotate(item, angle)
            if flip:
                item = TF.vflip(item)
        return item


# TODO: infill bad parts from neigboring days.
class FilterData:
    """ 
    Removes revisits that exceed a given threshold of pixels of the following
    categories in a cloud mask, either of which can be selected:

    0: No Data -> black
    1: Saturated / Defective -> red
    2: Dark Area Pixels -> gray
    3: Cloud Shadows -> brown
    4: Vegetation -> green
    5: Bare Soils -> yellow
    6: Water -> blue
    7: Clouds low probability / Unclassified -> medium gray
    8: Clouds medium probability -> light gray
    9: Clouds high probability -> very light gray
    10: Cirrus -> light blue/purple
    11: Snow / Ice -> cyan

    - fill=='replace' replaces a "bad" revisit with a "nice" one;
    - fill=='zero' zeroes out a "bad" revisit; else, it is removed.
    """

    def __init__(self, thres=0.4, values=(0,), fill="replace", batched=False):
        """ Initializes the FilterData class.

        Parameters
        ----------
        thres : float, optional
            Threshold of pixels a revisit has to exceed, by default .4.
        values : tuple of int, optional
            Categories to be filtered, by default (0,).
        fill : str, optional
            Whether to replace or zero out a "bad" revisit, by default 'replace'.
        batched : bool, optional
            Whether the data is batched, by default False.
        """

        self.threshold = thres
        self.fill = fill
        self.values = Tensor(values)
        self.batched = batched
        if batched:
            # Hacky but simple
            self.values = self.values.to("cuda")

    def remove_no_data(self, item):
        low_res, low_res_classification = item["lr"], item["lrc"]

        # If data is batched:
        if self.batched and low_res_classification.ndim == 5:
            batch_size, revisits = low_res_classification.shape[:2]
            low_res = low_res.flatten(start_dim=0, end_dim=1)
            low_res_classification = low_res_classification.flatten(
                start_dim=0, end_dim=1
            )

        if low_res_classification.shape[-3] > 1:  # one-hot to index
            fraction_without_data = (
                low_res_classification[:, self.values.long()]
                .any(-3)
                .float()
                .mean(dim=(-1, -2))
            )

        else:
            fraction_without_data = (
                (low_res_classification[..., None] == self.values)
                .any(-1)
                .float()
                .mean(dim=(1, 2, 3))
            )
        good_revisits_ix = fraction_without_data < self.threshold

        # If there is at least one bad revisit.
        if sum(~good_revisits_ix) > 0:
            # Replace it with zero.
            if self.fill == "zero":
                low_res[~good_revisits_ix] = 0
                low_res_classification[~good_revisits_ix] = 0
            # Replace it with the median of revisits.
            elif self.fill == "median":
                # If there are no good revisits:
                if sum(good_revisits_ix) == 0:
                    good_revisits_ix = torch.ones_like(good_revisits_ix).bool()
                low_res[~good_revisits_ix], _ = low_res[good_revisits_ix].median(
                    dim=-4, keepdim=True
                )

        if self.batched:
            low_res = low_res.unflatten(dim=0, sizes=(batch_size, revisits))
            low_res_classification = low_res_classification.unflatten(
                dim=0, sizes=(batch_size, revisits)
            )
        item["lr"], item["lrc"] = low_res, low_res_classification
        return item

    # Disable gradients for efficiency.
    @torch.no_grad()
    def __call__(self, item):
        return self.remove_no_data(item)


def lanczos_kernel(translation_in_px, kernel_lobes=3, kernel_width=None):
    """ Generates 1D Lanczos kernels for translation and interpolation.
    Adapted from: https://github.com/ElementAI/HighRes-net/blob/master/src/lanczos.py

    Parameters
    ----------
    translation_in_px : Tensor
        Translation in (sub-)pixels, (B,1).
    kernel_lobes : int, optional
        Number of kernel lobes, by default 3.
        If kernel_lobes is None, then the width is the kernel support 
        (length of all lobes), S = 2(a + ceil(subpixel_x)) + 1.
    kernel_width : Optional[int], optional
        Kernel width, by default None.

    Returns
    -------
    Tensor
        1D Lanczos kernel, (B,) or (N,) or (S,).
    """

    device = translation_in_px.device
    dtype = translation_in_px.dtype

    absolute_rounded_translation_in_px = translation_in_px.abs().ceil().int()
    # width of kernel support
    kernel_support_width = 2 * (kernel_lobes + absolute_rounded_translation_in_px) + 1

    maximum_support_width = (
        kernel_support_width.max()
        if hasattr(kernel_support_width, "shape")
        else kernel_support_width
    )

    if (kernel_width is None) or (kernel_width < maximum_support_width):
        kernel_width = kernel_support_width

    # Width of zeros beyond kernel support
    zeros_beyond_support_width = (
        ((kernel_width - kernel_support_width) / 2).floor().int()
    )

    start = (
        -(
            kernel_lobes
            + absolute_rounded_translation_in_px
            + zeros_beyond_support_width
        )
    ).min()
    end = (
        kernel_lobes
        + absolute_rounded_translation_in_px
        + zeros_beyond_support_width
        + 1
    ).max()
    x = (
        torch.arange(start, end, dtype=dtype, device=device).view(1, -1)
        - translation_in_px
    )
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / kernel_lobes)

    # sinc(x) masked by sinc(x/a)
    k = kernel_lobes * sin_px * sin_pxa / px ** 2

    return k


def lanczos_shift(x, shift, padding=3, kernel_lobes=3):
    """ Shifts an image by convolving it with a Lanczos kernel.
    Lanczos interpolation is an approximation to ideal sinc interpolation,
    by windowing a sinc kernel with another sinc function extending up to
    a few number of its lobes (typically 3).

    Adapted from:
            https://github.com/ElementAI/HighRes-net/blob/master/src/lanczos.py

    Parameters
    ----------
    x : Tensor
        Image to be shifted, (batch_size, channels, height, width).
    shift : Tensor
        Shift in (sub-)pixels/translation parameters, (B,2).
    padding : int, optional
        Width of the padding prior to convolution, by default 3.
    kernel_lobes : int, optional
        Number of lobes of the Lanczos interpolation kernel, by default 3.

    Returns
    -------
    _type_
        _description_
    """

    (batch_size, channels, height, width) = x.shape

    # Because examples and channels are interleaved in dim 1.
    shift = shift.repeat(channels, 1)  # (B, C * 2)
    shift = shift.reshape(batch_size * channels, 2)  # (B * C, 2)
    x = x.view(1, batch_size * channels, height, width)

    # Reflection pre-padding.
    pad = torch.nn.ReflectionPad2d(padding)
    x = pad(x)

    # 1D shifting kernels.
    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    # Flip dimension of convolution and expand dims to (batch_size, channels, len(kernel_y), 1).
    kernel_y = (lanczos_kernel(y_shift, kernel_lobes=kernel_lobes).flip(1))[
        :, None, :, None
    ]
    kernel_x = (lanczos_kernel(x_shift, kernel_lobes=kernel_lobes).flip(1))[
        :, None, None, :
    ]

    # 1D-convolve image with kernels: shifts image on x- then y-axis.
    x = torch.conv1d(
        x,
        groups=kernel_y.shape[0],
        weight=kernel_y,
        padding=[kernel_y.shape[2] // 2, 0],  # "same" padding.
    )
    x = torch.conv1d(
        x,
        groups=kernel_x.shape[0],
        weight=kernel_x,
        padding=[0, kernel_x.shape[3] // 2],
    )

    # Remove padding.
    x = x[..., padding:-padding, padding:-padding]

    return x.view(batch_size, channels, height, width)


class Shift:
    """ Sub-pixel image shifter with Lanczos shifting and interpolation kernels. """

    def __init__(self, padding=5, kernel_lobes=3):
        """ Initialize Shift.

        Parameters
        ----------
        padding : int, optional
            Width of the padding prior to convolution, by default 5.
        kernel_lobes : int, optional
            Number of lobes of the Lanczos interpolation kernel, by default 3.
        """
        self.padding = padding
        self.kernel_lobes = kernel_lobes

    def __call__(self, x, shift):
        """ Shift an image by convolving it with a Lanczos kernel.

        Parameters
        ----------
        x : Tensor
            Image to be shifted, (batch_size, channels, height, width).
        shift : Tensor
            Shift in (sub-)pixels/translation parameters, (batch_size,2).

        Returns
        -------
        Tensor
            Shifted image, (batch_size, channels, height, width).
        """
        return lanczos_shift(
            x, shift.flip(-1), padding=self.padding, kernel_lobes=self.kernel_lobes
        )


class WarpPerspective:
    """ Transform module for perspective warping.
    Uses kornia.warp_perspective."""

    def __call__(self, x, warp):
        """ Warp an image by perspective transformation.

        Parameters
        ----------
        x : Tensor
            Image to be warped, (batch_size, channels, height, width).
        warp : Tensor
            Warp/homography parameters, (batch_size, 8).

        Returns
        -------
        Tensor
            Warped image, (batch_size, channels, height, width).
        """
        """
        Args:
            x: Tensor (batch_size, channels, height, width), images.
            warp: Tensor (batch_size, 8), homography parameters.
        Returns:
            Tensor (batch_size, channels, height, width), shifted images.
        """
        batch_size, channels, height, width = x.shape

        # Homography parameters with warp[2,2]=1 (B, 9)
        warp = torch.cat([warp, torch.ones(batch_size, 1)])

        # Warp matrix (B, 3, 3)
        warp = warp.view(batch_size, 3, 3)

        return kornia.warp_perspective(
            x, warp, dsize=(height, width), mode="bilinear", padding_mode="zeros"
        )

