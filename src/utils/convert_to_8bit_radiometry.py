import sys
import os
import torch
import kornia
from tqdm.auto import trange
from multiprocessing import Manager
from pathlib import Path
from torchvision.transforms import Compose, Lambda, ToPILImage
from datasets import SatelliteDataset
import numpy as np
import tifffile as tiff
from datasources import SPOT_MAX_EXPECTED_VALUE_12_BIT


def convert_all_images_to_8bit_radiometry(root):
    """ Converts all the images in the root directory to 8-bit (including a pansharpened 8-bit RGB image).

    Parameters
    ----------
    root : str
        Path to the root directory.
    """
    check_root_exists(root)
    multiprocessing_manager = Manager()
    hr_dataset, pan_dataset = generate_datasets(root, multiprocessing_manager)
    aois_progress_bar = trange(len(hr_dataset), desc="Converting to 8 bit", leave=True)
    for aoi_index in aois_progress_bar:
        aoi_name, aoi_folder = extract_aoi_name_and_folder_from_path(
            hr_dataset, aoi_index
        )
        aois_progress_bar.set_description(f"Converting to 8-bit and saving {aoi_name}")
        aois_progress_bar.refresh()
        save_8_bit_pansharpened(
            hr_dataset[aoi_index], pan_dataset[aoi_index], aoi_folder
        )


def check_root_exists(root):
    """ Checks if the root directory exists.

    Parameters
    ----------
    root : str
        Path to the root directory.
    """
    if not os.path.exists(root):
        print(f"Root directory {root} does not exist. Exiting.")
        sys.exit(1)


def generate_datasets(root, multiprocessing_manager):
    """ Generates SatelliteDataset objects for the HR and PAN images.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.

    Returns
    -------
    SatelliteDataset, SatelliteDataset
        Datasets for the HR and PAN images.
    """
    print("Loading datasets.")
    transform_hr, transform_pan = generate_transforms()
    root = str(Path(root, "SPOT", "*"))
    hr_dataset = SatelliteDataset(
        root=root,
        file_postfix="_rgbn.tiff",
        transform=transform_hr,
        multiprocessing_manager=multiprocessing_manager,
        use_tifffile_as_reader=True,
    )
    hr_pan_dataset = SatelliteDataset(
        root=root,
        file_postfix="_pan.tiff",
        transform=transform_pan,
        multiprocessing_manager=multiprocessing_manager,
        use_tifffile_as_reader=True,
    )
    return hr_dataset, hr_pan_dataset


def generate_transforms():
    """ Generate transforms that normalize the HR and PAN image ranges.

    Returns
    -------
    torchvision.transform, torchvision.transform
        Transforms to be applied to the HR and PAN images.
    """
    transform_hr = Compose(
        [
            Lambda(
                lambda hr_revisit: torch.as_tensor(hr_revisit.astype(np.int32))
                / SPOT_MAX_EXPECTED_VALUE_12_BIT
            ),
            Lambda(lambda hr_revisit: hr_revisit.clamp(min=0, max=1)),
        ]
    )

    transform_pan = Compose(
        [
            Lambda(
                lambda hr_panchromatic: torch.as_tensor(
                    hr_panchromatic.astype(np.int32)
                )
                / SPOT_MAX_EXPECTED_VALUE_12_BIT
            ),
            Lambda(lambda hr_panchromatic: hr_panchromatic.clamp(min=0, max=1)),
        ]
    )

    return transform_hr, transform_pan


def extract_aoi_name_and_folder_from_path(hr_dataset, aoi_index):
    """ Extracts the AOI name and folder from the path.

    Parameters
    ----------
    hr_dataset : SatelliteDataset
        Dataset containing the HR images.
    aoi_index : int
        Index of the AOI.

    Returns
    -------
    str, str
        AOI name and folder.
    """
    aoi_path = Path(hr_dataset.paths[aoi_index])
    aoi_name = str(aoi_path.parts[-1]) + "/"
    aoi_folder = str(Path(aoi_path, aoi_name))
    return aoi_name, aoi_folder


def save_8_bit_pansharpened(hr, pan, path):
    """ Saves the 8-bit and pansharpened images to disk.
    First, the images are fetched from the SatelliteDatasets and extracted from tensors.
    Then, the images are resized if needed, the pansharpened image is generated using HSV pansharpening,
    the images are converted to 8-bit and saved to disk to the provided path.

    Parameters
    ----------
    hr : torch.Tensor
        Tensor containing the HR image.
    pan : torch.Tensor
        Tensor containing the PAN image.
    path : str
        Path to the directory where the images should be saved.
    """
    hr_pan, hr_rgb, hr_rgbn = extract_images_from_tensors(hr, pan)
    hr_ps = pansharpen_rgb(hr_rgb, hr_pan)
    eight_bit_pan, eight_bit_rgbn, eight_bit_ps, eight_bit_rgb = (
        convert_to_8bit(hr_pan),
        convert_to_8bit(hr_rgbn),
        convert_to_8bit(hr_ps),
        convert_to_8bit(hr_rgb),
    )
    save_to_files(path, eight_bit_pan, eight_bit_rgb, eight_bit_rgbn, eight_bit_ps)


def extract_images_from_tensors(hr, pan):
    """ Extracts the images from the tensors (B,C,H,W) -> (C,H,W).

    Parameters
    ----------
    hr : torch.Tensor
        Tensor containing the HR image.
    pan : torch.Tensor
        Tensor containing the PAN image.

    Returns
    -------
    Tensor, Tensor, Tensor
        Tensors containing the pansharpened image, the RGB image and the RGBNIR image.
    """
    hr_pan = pan[0][0]  # Extract the panchromatic channel
    hr_rgb = hr[0][:3]  # Extract the RGB channels
    hr_rgbn = hr[0]  # Unpack the RGBNIR channels
    return hr_pan, hr_rgb, hr_rgbn


def pansharpen_rgb(rgb, panchromatic):
    """Pansharpen the RGB image using the panchromatic image.

    The pansharpening method used HSV pansharpening. We first convert the RGB image to HSV (hue, saturation, value). 
    Then the value channel is replaced with the panchromatic image, which has a higher spatial resolution.
    The hue and saturation channels are kept unchanged, which means that the resulting image maintains the 
    color characteristics of the original RGB image. The resulting image is converted back to RGB.

    Parameters
    ----------
    rgb : Tensor
        The lower-resolution RGB channels/image as a tensor.
    panchromatic : Tensor
        The higher-resolution panchromatic channel/image as a tensor.

    Returns
    -------
    Tensor
        The pansharpened image as a tensor.
    """
    pan = np.array(ToPILImage()(panchromatic))
    rgb = resize_rgb(rgb, (pan.shape[0], pan.shape[1]))
    hr_hsv = kornia.color.rgb_to_hsv(rgb, eps=1e-08)
    hr_hsv[2] = pan
    hr_ps = kornia.color.hsv_to_rgb(hr_hsv)
    return hr_ps


def resize_rgb(rgb, shape):
    """ Resizes the RGB image to the shape of the PAN image to prepare it for pansharpening.

    Parameters
    ----------
    rgb : torch.Tensor
        _description_
    shape : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return kornia.geometry.transform.resize(
        rgb, size=shape, interpolation="bilinear", antialias=True, align_corners=False
    )


def convert_to_8bit(image):
    """ Converts the image to 8-bit by scaling it to [0, 255] range using the ToPILImage() transform.

    Parameters
    ----------
    image : torch.Tensor
        Tensor containing the image.

    Returns
    -------
    np.ndarray
        8-bit image.
    """
    return np.array(ToPILImage()(image))


def save_to_files(path, eight_bit_pan, eight_bit_rgb, eight_bit_rgbn, eight_bit_ps):
    """ Saves the 8-bit and pansharpened images to disk.

    Parameters
    ----------
    path : str
        Path to the directory where the images should be saved.
    eight_bit_pan : np.ndarray
        8-bit image of the PAN image.
    eight_bit_rgbn : np.ndarray
        8-bit image of the RGBNIR image.
    eight_bit_ps : np.ndarray
        8-bit image of the pansharpened RGB image.
    eight_bit_rgb : np.ndarray
        8-bit image of the RGB image.
    """
    tiff.imwrite(f"{path}_pan_8bit.tiff", eight_bit_pan)
    tiff.imwrite(f"{path}_rgbn_8bit.tiff", eight_bit_rgbn)
    tiff.imwrite(f"{path}_ps_8bit.tiff", eight_bit_ps)
    tiff.imwrite(f"{path}_rgb_8bit.png", eight_bit_rgb)


if __name__ == "__main__":

    root = sys.argv[1]
    convert_all_images_to_8bit_radiometry(root)
