#!/usr/bin/env python

import os

import natsort
import numpy as np
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import tifffile
import torch
import kornia
from tqdm.auto import tqdm
from glob import glob
from typing import Callable
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from multiprocessing import Manager
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, Lambda
import pandas as pd
from pathlib import Path
from PIL import Image
from src.datasources import (
    S2_ALL_12BANDS,
    SN7_SUBDIRECTORIES,
    S2_SN7_MEAN,
    S2_SN7_STD,
    SN7_BANDS_TO_READ,
    SN7_MAX_EXPECTED_HR_VALUE,
    SPOT_RGB_BANDS,
    JIF_S2_MEAN,
    JIF_S2_STD,
    S2_ALL_BANDS,
    SPOT_MAX_EXPECTED_VALUE_8_BIT,
    SPOT_MAX_EXPECTED_VALUE_12_BIT,
    ROOT_JIF_DATA_TRAIN,
    METADATA_PATH,
)
from src.transforms import (
    CropDict,
    FilterData,
    SceneClassificationToColorTransform,
    RandomRotateFlipDict,
)


class SatelliteDataset(Dataset):
    """PyTorch Dataset class used to load satellite imagery. """

    def __init__(
        self,
        root,
        number_of_revisits=None,
        subdir="",
        file_postfix=None,
        transform=None,
        bands_to_read=None,
        file_reader=None,
        use_cache=True,
        use_tifffile_as_reader=False,
        pan_dataset=None,
        list_of_aois=None,
        multiprocessing_manager=None,
        pansharpen_hr=True,
        load_metadata=False,
    ):
        """PyTorch Dataset class used to load satellite imagery.

        Parameters
        ----------
        root : str
            Path to a directory that contains either one subdirectory per AOI or one file per AOI revisit.
            Every subdirectory would contain the revisits for that AOI.

        subdir : str, optional
            Subdirectory within the root where revisits are, by default '' (no subdirectory).

        file_postfix : str, optional
            The postfix in which the revisit files end, by default 'tif*'.
            Allows wildcards.

        number_of_revisits : int, optional
            The number of revisits to get for each AOI, by default None (all revisits).

        transform : torchvision.transform, optional
            A transform that is applied to each revisit, by default Compose([]) (no transform).

        bands_to_read : list, optional
            A list of band IDs to read, by default None.
            By GDAL convention, band IDs are 1-indexed.

        file_reader : callable, optional
            A callable function used to read the files, by default SatelliteDataset._default_file_reader().

        use_cache : bool, optional
            Cache fetched items to memory, by default True.
            Can require a lot of RAM, but greatly speeds up training.

        use_tifffile_as_reader : bool, optional
            A flag to use the tifffile library to load images, by default False.
            If false, the rasterio library is used.


        list_of_aois : pandas.DataFrame, optional
            A pandas.DataFrame containing a list of AOI names that should be loaded from the root, by default None.

        multiprocessing_manager : multiprocessing.Manager, optional
            Used to prevent CPU memory leaking when multiple DataLoader workers access the cache, by default None.
            multiprocessing.Manager data structures are used to prevent copy-on-write behaviour due to refcounts in Python.
            See: https://github.com/pytorch/pytorch/issues/13246.

        pan_dataset : SatelliteDataset, optional
            A dataset of pan-chromatic HR revisits used to pan-sharpen HR RGB revisits, by default None.

        pansharpen_hr : bool, optional
            A flag to pansharpen the HR revisits using the pan_dataset, by default True.
        """

        self.check_root_is_not_empty(root)
        self.root = root
        self.subdir = subdir
        self.file_postfix = file_postfix if file_postfix is not None else "tif*"
        self.file_reader = file_reader or self._default_file_reader()
        self.transform = transform or Compose([])
        self.bands_to_read = bands_to_read
        self.number_of_revisits = number_of_revisits
        self.multiprocessing_manager = multiprocessing_manager
        self.paths = self.load_and_sort_aoi_paths(root, list_of_aois)
        self.use_cache = use_cache
        self.cache = self.multiprocessing_manager.dict()
        self.use_tifffile_as_reader = use_tifffile_as_reader
        self.pan_dataset = pan_dataset
        self.pansharpen_hr = pansharpen_hr
        self.load_metadata = load_metadata
        if self.load_metadata:
            self.metadata = pd.read_csv(METADATA_PATH, index_col=0)

    @staticmethod
    def check_root_is_not_empty(root):
        """Checks that the root is not empty.

        Parameters
        ----------
        root : str
            Path to a directory that contains either one subdirectory per AOI or one file per AOI revisit.
        """
        assert len(glob(root)) > 0, f"No files from {root}."

    def __len__(self):
        """Returns the number of AOIs in the dataset.

        Returns
        -------
        int
            The number of AOIs in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, item: int):
        """Returns a satellite image for a given AOI index from the dataset.
        If the image was previously fetched, it is returned from the cache.
        Otherwise, it is fetched from the disk and transformed.

        Parameters
        ----------
        item : int
            The index of the AOI in the dataset.

        Returns
        -------
        Tensor
            The satellite image as a tensor.
        """
        if item not in self.cache:
            path = self.paths[item]
            if os.path.isdir(path):
                # The path is a folder
                return self.load_revisits_from_folder(path, item)
            else:
                # The path is a single file
                return self.file_reader(path)
        else:
            # The item is cached, return from cache
            return self.cache[item]

    def _default_file_reader(self):
        """Returns the default file reader used to load the satellite images.
        If self.use_tifffile_as_reader is True, the tifffile library is used.
        If the files are PNGs, PIL is used. Otherwise, the rasterio library is used.

        Returns
        -------
        Callable
            The default file reader.
        """

        def _reader(path: str):
            if self.load_metadata:
                aoi = Path(path).parent.stem
                return self.metadata.loc[aoi].head(1).to_dict("records")
            if path.endswith(".png"):
                return self.read_png(path)
            if self.use_tifffile_as_reader:
                return self.read_tiff_with_tifffile(path, self.bands_to_read)
            else:
                return self.read_tiff_with_rasterio(path, self.bands_to_read)

        return _reader

    @staticmethod
    def read_tiff_with_rasterio(path, bands_to_read=None):
        """Reads a TIFF file using rasterio.

        Parameters
        ----------
        path : str
            The path to the TIFF file.
        bands_to_read : list, optional
            A list of band IDs to read, by default None.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        return rasterio.open(path).read(indexes=bands_to_read, window=None)

    @staticmethod
    def read_tiff_with_tifffile(path, bands_to_read=None):
        """Reads a TIFF file using tifffile.

        Parameters
        ----------
        path : str
            The path to the TIFF file.
        bands_to_read : list, optional
            A list of band IDs to read, by default None.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        x = tifffile.imread(path)
        if bands_to_read is not None:
            x = x[..., bands_to_read]
        return x

    @staticmethod
    def read_png(path):
        """Reads a PNG file using PIL.
        If the image is grayscale, it is converted to "RGB" by stacking the same channel three times.

        Parameters
        ----------
        path : str
            The path to the PNG file.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        image = np.array(Image.open(path)).astype(np.float32)
        if len(image.shape) == 2:  # 2D/grayscale
            return image[None, ...]
        else:
            return image

    def load_and_sort_aoi_paths(self, root, list_of_aois=None):
        """Loads the AOI paths from the root and sorts them.

        Parameters
        ----------
        list_of_aois : pandas.DataFrame, optional
            A pandas.DataFrame containing a list of AOI names that should be loaded from the root, by default None.
        root : str
            The path to the root directory.

        Returns
        -------
        multiprocessing.Manager.list
            A list of AOI paths, naturally sorted.
        """
        if list_of_aois is not None:
            return self.multiprocessing_manager.list(
                natsort.natsorted([self.root.replace("*", aoi) for aoi in list_of_aois])
            )
        else:
            return self.multiprocessing_manager.list(natsort.natsorted(glob(root)))

    def load_revisits_from_folder(self, path, item):
        """Loads the revisits/satellite images for a given AOI from a folder.

        Parameters
        ----------
        path : str
            The path to the AOI folder.
        item : int
            The index of the AOI in the dataset.

        Returns
        -------
        Tensor
            The revisits as a tensor.
        """
        # The folder contains revisits for a single AOI, read them into an AOI-specific dataset
        aoi_dataset = self.generate_dataset_for_aoi_folder(path)
        number_of_revisits = self.determine_the_number_of_revisits_to_return(
            aoi_dataset
        )
        if number_of_revisits > 0:
            x = self.load_revisits_from_aoi_dataset(number_of_revisits, aoi_dataset)
        else:
            # No revisits are found or the number_of_revisits is set to 0
            return None
        x = self.transform(x)
        x = self.pansharpen_hr_revisits(item, x)
        self.cache_revisits(item, x)
        return x

    def generate_dataset_for_aoi_folder(self, path):
        """Generates a SatelliteDataset for a given AOI folder.
        The folder contains the revisits available for that AOI.

        Parameters
        ----------
        path : str
            The path to the AOI folder.

        Returns
        -------
        SatelliteDataset
            The SatelliteDataset for the AOI folder.
        """
        aoi_dataset = SatelliteDataset(
            root=os.path.join(path, self.subdir, f"*{self.file_postfix}"),
            transform=self.transform,  # WARNING: random transforms vary among revisits!
            file_reader=self.file_reader,
            use_cache=False,
            multiprocessing_manager=self.multiprocessing_manager,
        )
        return aoi_dataset

    def determine_the_number_of_revisits_to_return(self, aoi_dataset):
        """Determines the number of revisits to return by looking at the wanted number of revisits
        and the number of available revisits in the folder (AOI) dataset. If the number of revisits
        is set to 0, all revisits are returned. If the number of revisits is set to a positive number,
        the minimum of the two is returned.

        Parameters
        ----------
        aoi_dataset : SatelliteDataset
            The SatelliteDataset for the AOI folder.

        Returns
        -------
        int
            The number of revisits to return.
        """
        if self.number_of_revisits:
            # If there are less than number_of_revisits available, return as much as possible
            number_of_revisits = min(self.number_of_revisits, len(aoi_dataset))
        else:
            # If the number_of_revisits isn't specified, return all available revisits
            number_of_revisits = len(aoi_dataset)
        return number_of_revisits

    def load_revisits_from_aoi_dataset(self, number_of_revisits, aoi_dataset):
        """Loads the revisits from a AOI (folder) SatelliteDataset.

        Parameters
        ----------
        number_of_revisits : int
            The number of revisits to return.
        folder_dataset : SatelliteDataset
            The SatelliteDataset for the AOI folder.

        Returns
        -------
        Tensor
            The revisits as a tensor.
        """
        x = np.stack(
            [aoi_dataset[revisit] for revisit in range(number_of_revisits)], axis=0
        )
        if self.use_tifffile_as_reader:
            if x.ndim == 3:  # If image is grayscale
                # Add empty dimension for channels
                x = x[..., None]
            number_of_revisits, height, width, channels = 0, 1, 2, 3
            # Convert from channel-last to channel-first
            x = x.transpose(number_of_revisits, channels, height, width)
        return x

    def pansharpen_hr_revisits(self, item, x):
        """Pansharpen the HR revisits using their pan-chromatic channel if the pansharpening flag is set.

        Parameters
        ----------
        item : int
            The index of the AOI in the dataset.
        x : Tensor
            The revisits as a tensor.

        Returns
        -------
        Tensor
            The pansharpened revisits as a tensor.
        """
        # Check if the revisit is HR, pan-chromatic channel is available and the pansharpen flag is set
        if (
            self.file_postfix == "_rgbn.tiff"
            and self.pan_dataset is not None
            and self.pansharpen_hr
        ):
            x = self.pansharpen(x, self.pan_dataset[item])
            x = torch.as_tensor(x[None, ...])
        return x

    @staticmethod
    def pansharpen(rgb, panchromatic):
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
        # Unpack the channel from the tensor
        hr_panchromatic = panchromatic[0][0]
        # Unpack the RGB channels from the tensor
        hr_rgb = rgb[0][:3]
        # Convert to HSV (Hue, Saturation, Value)
        hr_hsv = kornia.color.rgb_to_hsv(hr_rgb, eps=1e-08)
        # Set the pan-chromatic channel as the value channel
        hr_hsv[2] = hr_panchromatic
        # Convert back to RGB
        hr_pansharpened = kornia.color.hsv_to_rgb(hr_hsv)
        return hr_pansharpened

    def cache_revisits(self, item, x):
        """Caches the AOI revisits within the multiprocessing.Manager dictionary.

        Parameters
        ----------
        item : int
            The index of the AOI in the dataset.
        x : Tensor
            The revisits as a tensor.
        """
        if self.use_cache and self.file_postfix != "_pan.tiff":
            # Don't cache pan-chromatic channels/images to save RAM
            self.cache[item] = x

    def compute_median_std(self, name=None):
        """Compute the dataset median and median-standard deviation per band/channel.
        Note: The standard deviation is computed relatively to the median, it's not the usual standard deviation.

        Parameters
        ----------
        name: str, optional
            The name of the dataset, used in the progress bar while computing, by default None.

        Returns
        -------
        torch.Tensor, torch.Tensor
            The median and median-standard deviation per band/channel.
        """

        progress_bar_description = f" for {name}" if name is not None else ""
        number_of_channels = self[0].shape[1]
        number_of_revisits, channels, height, width = 0, 1, 2, 3

        channels_over_all_revisits = torch.cat(
            [
                x.to(float)
                # Permute to channel-first
                .permute(dims=(channels, number_of_revisits, height, width))
                # Flatten to [number_of_channels, number_of_channels * height * width]
                .reshape(number_of_channels, -1)
                # Progress bar
                for x in tqdm(
                    self, desc=f"Calculating median and std{progress_bar_description}"
                )
            ],
            # Concatenate along the channel axis
            dim=1,
        )

        # Compute the median by reducing along the channel axis
        median = channels_over_all_revisits.median(dim=1, keepdims=True).values
        # Compute the median-standard deviation by subtracting the median from the channel-wise median-standard deviation
        std = ((channels_over_all_revisits - median) ** 2).mean(dim=1).sqrt()
        return median.squeeze(), std


class TransformDataset(Dataset):
    """
    PyTorch Dataset class used to apply a transform on items when fetching them.
    Source: https://gist.github.com/alkalait/c99213c164df691b5e37cd96d5ab9ab2#file-sn7dataset-py-L278
    """

    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        """
        PyTorch Dataset class used to apply a transform on items when fetching them.

        Parameters
        ----------
        dataset : torch.Dataset
            The dataset on whose items to apply the transform.
        transform : Callable
            The transform to apply on the items.
        """
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item: int):
        """Get an item from the dataset with the transform applied to it.

        Parameters
        ----------
        item : int
            The index of the item to get.

        Returns
        -------
        Tensor
            The item with the transform applied.
        """
        item = self.dataset.__getitem__(item)
        return self.transform(item)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)


class DictDataset(Dataset):
    """ PyTorch Dataset class used to wrap a dictionary of datasets.
    The item index passed to the get method refers to images from a single AOI, or patch (when cropping is applied).
    The get method returns a dictionary of the items at the item index for each dataset."""

    def __init__(self, **dictionary_of_datasets):
        """

        Parameters
        ----------
        dictionary_of_datasets : dict
            The dictionary of torch datasets to wrap.
        """
        self.datasets = {
            dataset_name: dataset
            for dataset_name, dataset in dictionary_of_datasets.items()
            if isinstance(dataset, Dataset)
        }

    def __getitem__(self, item):
        """Get an item from the datasets.

        Parameters
        ----------
        item : int
            The index of the item to get from each dataset.

        Returns
        -------
        dict
            The items from each dataset (dataset_name:item).
        """
        return {name: dataset[item] for name, dataset in self.datasets.items()}

    def __len__(self):
        """Returns the length of the smallest dataset in the dictionary of wrapped datasets.

        Returns
        -------
        int
            The length of the smallest dataset in the dictionary of wrapped datasets.
        """
        return min(len(dictionary) for dictionary in self.datasets.values())


def make_dataloaders(
    root,
    subdir,
    input_size,
    chip_size,
    bands_to_read,
    transforms,
    number_of_scenes_per_split,
    data_split_seed=42,
    number_of_revisits=None,
    file_postfix=None,
    use_tifffile_as_reader=None,
    chip_stride=None,
    scene_classification_filter_values=None,
    scene_classification_filter_threshold=None,
    subset_train=None,
    randomly_rotate_and_flip_images=True,
    pansharpen_hr=False,
    compute_median_std=False,
    list_of_aois=None,
    **kws,
):
    """Makes the dataloaders for the dataset.

    First, a SatelliteDataset is generated for each data type (LR, HR, ...).
    The HR dataset is pansharpened using its panchromatic channel if pansharpen_hr is True.

    The SatelliteDatasets are then randomly shuffled and wrapped in a TransformDataset.
    The TransformDataset crops each scene/AOI into chips of size chip_size.
    If the scene classification filter values and thresholds are set, it filters the datasets by them.
    It also randomly flips and rotates the images if randomly_rotate_and_flip_images is True.

    The datasets are then split into training, validation and test sets, and returned as dataloaders.

    Parameters
    ----------
    root : str
        The root directory of the dataset.
    subdir : str
        The subdirectories of the dataset (one for each data type, e.g. LR, HR, LRC, HR_PAN).
    input_size : int
        The size of the input images.
    chip_size : int
        The size of the chips the images will be divided into.
    bands_to_read : list
        The bands/channels to read from the images.
    transforms : list
        The transforms to apply on the images.
    number_of_scenes_per_split : int
        The number of scenes/AOIs to use per split.
    data_split_seed : int, optional
        The seed used to shuffle the scenes/AOIs, by default 42.
    number_of_revisits : int, optional
        The number of revisits to use, by default None (all revisits).
    file_postfix : str, optional
        The file postfix to use, by default None (tif*).
    use_tifffile_as_reader : _type_, optional
        Whether to use tifffile as reader, by default None (use_tifffile_as_reader=False).
    chip_stride : int, optional
        The stride to use, by default None (chip_stride=chip_size).
    scene_classification_filter_values : list, optional
        The scene classification filter values to use, by default None (no filter).
    scene_classification_filter_threshold : float, optional
        The scene classification filter threshold to use, by default None (no filter).
    subset_train : float, optional
        The fraction of the dataset to use for training, by default None (subset_train=1).
    randomly_rotate_and_flip_images : bool, optional
        Whether to randomly rotate and flip images, by default True.
    pansharpen_hr : bool, optional
        Whether to use the pansharpen HR images, by default False.
    compute_median_std : bool, optional
        Whether to compute the median and standard deviation of the images, by default False.
    list_of_aois : pandas.DataFrame
        An explicit list of AOIs to load, by default None (all AOIs).

    Returns
    -------
    dict of torch.utils.data.DataLoader
        The dataloaders.
    """
    (
        root,
        number_of_revisits,
        file_postfix,
        use_tifffile_as_reader,
        chip_stride,
    ) = set_default_argument_values(
        root,
        number_of_revisits,
        file_postfix,
        use_tifffile_as_reader,
        transforms.keys(),
        chip_size,
        chip_stride,
    )

    multiprocessing_manager = Manager()
    satellite_datasets_arguments = generate_satellite_dataset_arguments_from_kws(
        root,
        subdir,
        file_postfix,
        bands_to_read,
        transforms,
        use_tifffile_as_reader,
        number_of_revisits,
        multiprocessing_manager,
        list_of_aois,
    )

    ## TODO(ori): extract
    # satellite_datasets_arguments["metadata"]["load_metadata"] = True

    datasets = generate_satellite_datasets(
        satellite_datasets_arguments, pansharpen_hr, list_of_aois, compute_median_std, subdir
    )

    datasets = shuffle_datasets(datasets, data_split_seed)
    datasets, number_of_chips = generate_chipped_filtered_and_augmented_datasets(
        datasets,
        chip_size,
        chip_stride,
        input_size,
        randomly_rotate_and_flip_images,
        scene_classification_filter_threshold,
        scene_classification_filter_values,
    )
    if type(datasets) is dict:
        dataset_train, dataset_val, dataset_test = (
            datasets["train"][0],
            datasets["val"][0],
            datasets["test"][0],
        )
    else:
        # Train / val / test split, at the scene level
        dataset_test, dataset_train, dataset_val = split_into_train_test_val(
            datasets, number_of_chips, number_of_scenes_per_split
        )

    if subset_train is not None:
        dataset_train = reduce_training_set(dataset_train, subset_train)

    test_dataloader, train_dataloader, val_dataloader = create_dataloaders_for_datasets(
        dataset_test, dataset_train, dataset_val, kws
    )
    print(f"Train set size: {len(dataset_train)}")
    print(f"Val set size: {len(dataset_val)}")
    print(f"Test set size: {len(dataset_test)}")

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def set_default_argument_values(
    root,
    number_of_revisits,
    file_postfix,
    use_tifffile_as_reader,
    dataset_keys,
    chip_size,
    chip_stride,
):
    """ Sets the default argument values.

    Parameters
    ----------
    root : str or dict
        The root directory of the dataset.
    number_of_revisits : int or dict
        The number of revisits for each dataset.
    file_postfix : dict
        The file postfix for each dataset.
    use_tifffile_as_reader : dict
        Whether to use tifffile as reader for each dataset.
    dataset_keys : list of str
        The dataset names/keys.
    chip_size : int
        The size of the chips/patches.
    chip_stride : int
        The stride of the chips/patches.

    Returns
    -------
    dict, dict, dict, dict, int
        The default argument values.
    """
    none_dictionary = {key: None for key in dataset_keys}
    if isinstance(root, str):
        root = {"lr": root, "lrc": root, "hr": root}
    if number_of_revisits is None:
        number_of_revisits = none_dictionary
    elif isinstance(number_of_revisits, int):
        number_of_revisits = {
            "lr": number_of_revisits,
            "lrc": number_of_revisits,
            "hr": 1,
            "hr_pan": 1,
            # "metadata": 1,
        }
    file_postfix = none_dictionary if file_postfix is None else file_postfix
    use_tifffile_as_reader = (
        none_dictionary if use_tifffile_as_reader is None else use_tifffile_as_reader
    )
    chip_stride = chip_size if chip_stride is None else chip_stride
    return root, number_of_revisits, file_postfix, use_tifffile_as_reader, chip_stride


def generate_satellite_dataset_arguments_from_kws(
    root,
    subdir,
    file_postfix,
    bands_to_read,
    transforms,
    use_tifffile_as_reader,
    number_of_revisits,
    multiprocessing_manager,
    list_of_aois,
):
    """ Generates the arguments used to generate the satellite datasets.

    Parameters
    ----------
    root : dict of str
        The root directory of the datasets.
    subdir : dict of str
        The subdirectories for the datasets.
    file_postfix : dict of str
        The file postfix of the datasets.
    bands_to_read : dict of list of int
        _description_
    transforms : dict of torchvision.transforms
        The transforms to apply to the datasets.
    use_tifffile_as_reader : dict of bool
        Whether to use tifffile as reader for the datasets.
    number_of_revisits : dict of int
        The number of revisits to use for the datasets.
    multiprocessing_manager : multiprocessing.Manager
        The multiprocessing manager to use for the loading of data through the SatelliteDatasets.
    list_of_aois : pandas.DataFrame
        The list of AOIs to use for the datasets.

    Returns
    -------
    dict of dict
        The arguments to generate a SatelliteDataset for each dataset.
    """
    return {
        dataset_name: dict(
            root=root[dataset_name],
            subdir=subdir[dataset_name],
            file_postfix=file_postfix[dataset_name],
            bands_to_read=bands_to_read[dataset_name],
            transform=transforms[dataset_name],
            use_tifffile_as_reader=use_tifffile_as_reader[dataset_name],
            number_of_revisits=number_of_revisits[dataset_name],
            multiprocessing_manager=multiprocessing_manager,
            list_of_aois=list_of_aois,
        )
        for dataset_name in subdir
    }


def generate_datasets_for_split(
    split, satellite_datasets_arguments, list_of_aois, pansharpen_hr
):
    split_aois = list_of_aois[list_of_aois["split"] == split]
    datasets_arguments = satellite_datasets_arguments.copy()
    for dataset in datasets_arguments.keys():
        datasets_arguments[dataset]["list_of_aois"] = list(split_aois.index)

    return generate_satellite_datasets(
        satellite_datasets_arguments, pansharpen_hr, None
    )


def generate_satellite_datasets(
    satellite_dataset_arguments, pansharpen_hr, list_of_aois, compute_median_std, subdir
):
    """ Generates the satellite datasets using the arguments as returned by 
    generate_satellite_dataset_arguments_from_kws.

    Parameters
    ----------
    satellite_dataset_arguments : dict of dict
        The arguments to generate the satellite datasets, 
        returned by generate_satellite_dataset_arguments_from_kws.
    pansharpen_hr : bool
        Whether to use the pansharpen HR images, by default False.

    Returns
    -------
    DictDataset of SatelliteDataset
        The satellite datasets.
    """
    if isinstance(list_of_aois, pd.DataFrame) and "split" in list_of_aois.columns:

        assert {"train", "test", "val"} == set(
            list_of_aois["split"]
        ), "The list of AOIs needs to have a train/test/val split"

        train, val, test = [
            generate_datasets_for_split(
                split, satellite_dataset_arguments, list_of_aois, pansharpen_hr
            )
            for split in ["train", "val", "test"]
        ]
        return {"train": train, "val": val, "test": test}
    elif pansharpen_hr:
        datasets = generate_satellite_datasets_with_pansharpening(
            satellite_dataset_arguments, compute_median_std, subdir
        )
    else:
        datasets = generate_satellite_datasets_without_pansharpening(
            satellite_dataset_arguments, compute_median_std, subdir
        )
    return datasets


def generate_satellite_datasets_without_pansharpening(satellite_dataset_arguments, compute_median_std, subdir):
    """ Generates the satellite datasets without using pansharpening the HR images.

    Parameters
    ----------
    satellite_dataset_arguments : dict of dict
        The arguments to generate the satellite datasets, 
        returned by generate_satellite_dataset_arguments_from_kws.

    Returns
    -------
    DictDataset of SatelliteDataset
        The satellite datasets.
    """

    datasets = {
        dataset_name: SatelliteDataset(**arguments)
        for dataset_name, arguments in satellite_dataset_arguments.items()
    }    

    if compute_median_std is True:
        compute_median_std_for_datasets(datasets, subdir)

    dataset_dict = DictDataset(
        **datasets
    )
    return dataset_dict


def generate_satellite_datasets_with_pansharpening(satellite_dataset_arguments, compute_median_std, subdir):
    """ Generates the satellite datasets with pansharpened HR images.

    Parameters
    ----------
    satellite_dataset_arguments : dict of dict
        The arguments to generate the satellite datasets,
        returned by generate_satellite_dataset_arguments_from_kws.

    Returns
    -------
    DictDataset of SatelliteDataset
        The satellite datasets.
    """
    dataset_dict = {}
    if {"hr", "hr_pan"}.issubset(satellite_dataset_arguments.keys()):
        dataset_dict["hr_pan"] = SatelliteDataset(
            **satellite_dataset_arguments["hr_pan"]
        )
        satellite_dataset_arguments["hr"]["pan_dataset"] = dataset_dict["hr_pan"]
        satellite_dataset_arguments["hr"]["pansharpen_hr"] = True

    for dataset_name, arguments in satellite_dataset_arguments.items():
        if dataset_name != "hr_pan":
            satellite_dataset = SatelliteDataset(**arguments)
            dataset_dict[dataset_name] = satellite_dataset

    datasets = {
        dataset_name: satellite_dataset
        for dataset_name, satellite_dataset in dataset_dict.items()
    }
    
    if compute_median_std is True:
        compute_median_std_for_datasets(datasets, subdir)

    dataset_dict = DictDataset(
        **datasets
    )
    return dataset_dict


def compute_median_std_for_datasets(datasets, subdir):
    """ Computes the median and standard deviation of the datasets.

    Parameters
    ----------
    datasets : DictDataset of SatelliteDataset
        The datasets.
    subdir : dict of str
        The subdirectories for the datasets.
    """
    for dataset_name in subdir:
        dataset = datasets[dataset_name]
        print(f"{dataset_name}:{dataset.compute_median_std(name=dataset_name)}")


def shuffle_datasets(datasets, data_split_seed):
    """ Shuffles the datasets using the provided seed.

    Parameters
    ----------
    datasets : DictDataset of SatelliteDataset
        The datasets to shuffle.
    data_split_seed : int
        The seed to use for the shuffling.

    Returns
    -------
    DictDataset of SatelliteDataset
        The shuffled datasets.
    """
    print(f"Shuffling the dataset splits using {data_split_seed}")
    if isinstance(datasets, dict):
        return {
            key: shuffle_datasets(value, data_split_seed)
            for key, value in datasets.items()
        }
    number_of_scenes = len(datasets)
    (datasets,) = random_split(
        datasets,
        [number_of_scenes,],
        generator=torch.Generator().manual_seed(data_split_seed),
    )
    return datasets


def generate_chipped_filtered_and_augmented_datasets(
    datasets,
    chip_size,
    chip_stride,
    input_size,
    randomly_rotate_and_flip_images,
    scene_classification_filter_threshold,
    scene_classification_filter_values,
):
    """ Filters the data using the scene classification filter values and thresholds,
    crops the scenes/images into chips, applies random rotations and flips.

    Parameters
    ----------
    datasets : dict of SatelliteDataset
        The datasets to filter, crop and augment.
    chip_size : tuple
        The size of the chips the images will be divided into.
    chip_stride : int
        The stride to use.
    input_size : tuple
        The size of the input images.
    randomly_rotate_and_flip_images : bool
        Whether to randomly rotate and flip images.
    scene_classification_filter_threshold : float
        The threshold by which the scene classification filter values are filtered by.
    scene_classification_filter_values : list of int
        The scene classification filter values to use.

    Returns
    -------
    dict of SatelliteDataset
        The filtered, cropped and augmented datasets.
    """
    if isinstance(datasets, dict):
        return (
            {
                key: generate_chipped_filtered_and_augmented_datasets(
                    value,
                    chip_size,
                    chip_stride,
                    input_size,
                    randomly_rotate_and_flip_images,
                    scene_classification_filter_threshold,
                    scene_classification_filter_values,
                )
                for key, value in datasets.items()
            },
            None,
        )
    number_of_scenes = len(datasets)
    filter_data = generate_scene_classification_filters(
        scene_classification_filter_threshold, scene_classification_filter_values
    )
    randomly_rotate_and_flip_images = generate_random_rotation_and_flip_transform(
        randomly_rotate_and_flip_images
    )
    # Concatenate chipped views of the scene-level dataset, with a sliding window.
    dataset, number_of_chips = apply_filter_crop_and_rotations_to_datasets(
        datasets,
        input_size,
        chip_size,
        chip_stride,
        filter_data,
        randomly_rotate_and_flip_images,
    )
    dataset = transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes)
    return dataset, number_of_chips


def generate_scene_classification_filters(
    scene_classification_filter_threshold, scene_classification_filter_values
):
    """ Generates the torchvision.transforms.FilterData scene classification filters from the provided parameters.

    Parameters
    ----------
    scene_classification_filter_threshold : float
        The threshold by which the scene classification filter values are filtered by.
    scene_classification_filter_values : list of int
        The scene classification filter values to use.

    Returns
    -------
    torchvision.transforms.FilterData or torchvision.transforms.Compose
        The scene classification filters, if any, else a dummy filter.
    """
    # 0: no data, 3: cloud shadow, 8: cloud med prob, 9: cloud high prob, 10: cirrus, 11: snow
    if scene_classification_filter_values is not None:
        filter_data = FilterData(
            values=scene_classification_filter_values,
            thres=scene_classification_filter_threshold,
            fill="zero",
        )
    else:
        filter_data = Compose([])
    return filter_data


def generate_random_rotation_and_flip_transform(randomly_rotate_and_flip_images):
    """ Generates the random rotation and flip transform.

    Parameters
    ----------
    randomly_rotate_and_flip_images : bool
        Whether to randomly rotate and flip the images.

    Returns
    -------
    torchvision.transforms.RandomRotateFlipDict or torchvision.transforms.Compose
        The random rotation and flip transform, if any, else a dummy transform.
    """
    if randomly_rotate_and_flip_images:
        randomly_rotate_and_flip_images = RandomRotateFlipDict(angles=[0, 90, 180, 270])
    else:
        randomly_rotate_and_flip_images = Compose([])
    return randomly_rotate_and_flip_images


def apply_filter_crop_and_rotations_to_datasets(
    dataset_dict,
    input_size,
    chip_size,
    chip_stride,
    filter_data,
    randomly_rotate_and_flip_images,
):
    """
    Crops the datasets into chips/patches and applies the rotations and flips to the images.

    Parameters
    ----------
    dataset_dict : DictDataset of SatelliteDataset
        The datasets to crop and rotate.
    input_size : int
        The size of the input images.
    chip_size : int
        The size of the chips/patches.
    chip_stride : int
        The stride of the chips/patches.
    filter_data : torchvision.transforms.FilterData or torchvision.transforms.Compose
        The scene classification filters, if any, else a dummy filter.
    randomly_rotate_and_flip_images : bool
        Whether to randomly rotate and flip the images.

    Returns
    -------
    DictDataset of SatelliteDataset
        The cropped, rotated datasets.
    """
    dataset_dict_grid = []
    input_height, input_width = input_size
    chip_height, chip_width = chip_size
    stride_height, stride_width = chip_stride

    # Make sure chip isn't larger than the input size
    assert chip_height <= input_height and chip_width <= input_height

    last_stride_step_x = input_width - chip_width + 1
    last_stride_step_y = input_height - chip_height + 1
    for stride_step_x in range(0, last_stride_step_x, stride_width):
        for stride_step_y in range(0, last_stride_step_y, stride_height):
            transform_dict = Compose(
                [
                    CropDict(
                        stride_step_x, stride_step_y, chip_width, chip_height, src="lr"
                    ),
                    filter_data,
                    randomly_rotate_and_flip_images,
                ]
            )
            dataset_dict_grid.append(
                TransformDataset(dataset_dict, transform=transform_dict)
            )
    dataset = torch.utils.data.ConcatDataset(dataset_dict_grid)
    return dataset, len(dataset_dict_grid)


def transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes):
    """ Transposes the scenes and cropped chips of the datasets.

    Parameters
    ----------
    dataset : DictDataset of SatelliteDataset
        The cropped/chipped datasets.
    number_of_chips : int
        The number of chips to transpose.
    number_of_scenes : int
        The number of scenes to transpose.

    Returns
    -------
    DictDataset of SatelliteDataset
        The transposed datasets.
    """
    # Transpose scenes and chips
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., number_of_scenes * number_of_chips]
    indices = Tensor(range(number_of_scenes * number_of_chips)).int()
    # indices = [0, number_of_scenes, 2*number_of_scenes, ..., 1+number_of_scenes, 2+2*number_of_scenes, ... ]
    transposed_indices = indices.reshape(number_of_chips, number_of_scenes).T.reshape(
        indices.numel()
    )
    dataset = torch.utils.data.Subset(dataset, transposed_indices)
    assert len(dataset) == number_of_scenes * number_of_chips
    return dataset


def split_into_train_test_val(dataset, number_of_chips, number_of_scenes_per_split):
    """ Splits the dataset into train, test and validation sets.

    Parameters
    ----------
    dataset : DictDataset of SatelliteDataset
        The dataset to split.
    number_of_chips : int
        The number of chips per scene.
    number_of_scenes_per_split : int
        The number of scenes per split.

    Returns
    -------
    _type_
        _description_
    """
    number_of_train_scenes, number_of_validation_scenes, number_of_test_scenes = (
        number_of_scenes_per_split["train"],
        number_of_scenes_per_split["val"],
        number_of_scenes_per_split["test"],
    )
    dataset_train = Subset(dataset, range(number_of_train_scenes * number_of_chips))
    dataset_val = Subset(
        dataset,
        range(
            number_of_train_scenes * number_of_chips,
            (number_of_train_scenes + number_of_validation_scenes) * number_of_chips,
        ),
    )
    dataset_test = Subset(
        dataset,
        range(
            (number_of_train_scenes + number_of_validation_scenes) * number_of_chips,
            (
                number_of_train_scenes
                + number_of_validation_scenes
                + number_of_test_scenes
            )
            * number_of_chips,
        ),
    )
    return dataset_test, dataset_train, dataset_val


def reduce_training_set(dataset_train, subset_train):
    """ Reduces the training set to the provided percentage.

    Parameters
    ----------
    dataset_train : SatelliteDataset
        The training dataset.
    subset_train : float
        The percentage of the training set to use.

    Returns
    -------
    SatelliteDataset
        The reduced training dataset.
    """
    # Reduce the train set if needed
    if subset_train < 1:
        dataset_train = torch.utils.data.Subset(
            dataset_train, list(range(int(subset_train * len(dataset_train))))
        )
    return dataset_train


def create_dataloaders_for_datasets(dataset_test, dataset_train, dataset_val, kws):
    """ Creates the PyTorch dataloaders from the datasets.

    Parameters
    ----------
    dataset_test : torch.utils.data.Dataset
        The test dataset.
    dataset_train : torch.utils.data.Dataset
        The training dataset.
    dataset_val : torch.utils.data.Dataset
        The validation dataset.
    kws : dict
        The keyword arguments to use for the dataloaders.

    Returns
    -------
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader
        The dataloaders for the test, training and validation datasets.
    """
    batch_size, number_of_workers = kws.get("batch_size", 1), kws.get("num_workers", 1)
    train_dataloader = DataLoader(
        dataset_train,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # val_loader = DataLoader(dataset_val, num_workers=W, batch_size=len(dataset_val), pin_memory=True,
    val_dataloader = DataLoader(
        dataset_val,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # test_loader = DataLoader(dataset_test, num_workers=W, batch_size=len(dataset_test), pin_memory=True)
    test_dataloader = DataLoader(
        dataset_test, num_workers=number_of_workers, batch_size=1, pin_memory=True
    )
    return test_dataloader, train_dataloader, val_dataloader


def load_dataset_SN7(**kws):
    """ Parses the necessary keyword arguments, creates the transforms and loads the SN7 dataset.

     Parameters
    ----------
    **kws : dict
        The keyword arguments dictionary from which the necessary arguments are fetched, as generated by train.py.
    kws['input_size'] : tuple
        The size of the (LR) input image, by default (400, 400).
    kws['output_size'] : tuple
        The size of the (HR) output image, by default (1000, 1000).
    kws['chip_size'] : tuple
        The chip/patch size to divide the images into, by default the input size (400, 400).
    kws['normalize_lr'] : bool
        Whether to normalize the low-resolution images.
    kws['interpolation'] : torchvision.transforms.InterpolationMode
        The interpolation mode to resize the images, by default InterpolationMode.BICUBIC.
    kws['scene_classification_filter_values'] : list
        The list of LR scene classification bands to filter the low-resolution images by.
        See: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    kws['scene_classification_filter_threshold'] : float
        The threshold to filter the LR scene classification values by.
    kws['root'] : str
        The root directory of the dataset, by default './data/SN7/train/*'.
    kws['sclcolor'] : bool
        Whether to convert the scene classification values to colors, by default True.
        See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/
    kws['compute_median_std'] : bool
        Whether to compute the median and standard deviation of the low-resolution images, by default True.
        Once computed, the values can be used to normalise the low-resolution images.
    kws['train_split'] : int
        The number of scenes to use for the train split, by default 50.
    kws['val_split'] : int
        The number of scenes to use for the val split, by default 5.
    kws['test_split'] : int
        The number of scenes to use for the test split, by default 5.

    Returns
    -------
    dict of {str: torch.utils.data.DataLoader}
        The dictionary of dataloaders (hr, lr, lrc).
    """
    kws.setdefault("input_size", (400, 400))
    kws.setdefault("output_size", (1000, 1000))
    kws.setdefault("chip_size", kws["input_size"])
    kws.setdefault("normalize_lr", True)
    kws.setdefault("interpolation", InterpolationMode.BICUBIC)
    kws.setdefault("scene_classification_filter_values", (0, 1, 3, 8, 9, 10, 11))
    kws.setdefault("scene_classification_filter_threshold", 0.4)
    kws.setdefault("root", "./data/SN7/train/*/")
    kws.setdefault("scene_classification_to_color", False)
    kws.setdefault("compute_median_std", False)
    kws.setdefault("train_split", 50)
    kws.setdefault("val_split", 5)
    kws.setdefault("test_split", 5)

    kws["number_of_revisits"] = kws["revisits"]
    return make_dataloaders(
        subdir=SN7_SUBDIRECTORIES,
        bands_to_read=SN7_BANDS_TO_READ,
        transforms=make_transforms_SN7(**kws),
        number_of_scenes_per_split={
            "train": kws["train_split"],
            "val": kws["val_split"],
            "test": kws["test_split"],
        },
        **kws,
    )


def load_dataset_JIF(**kws):
    """ Parses the necessary keyword arguments, creates the transforms and loads the JIF dataset.

     Parameters
    ----------
    **kws : dict
        The keyword arguments dictionary from which the necessary arguments are fetched, as generated by train.py.
    kws['input_size'] : tuple
        The size of the (LR) input image, by default (400, 400).
    kws['output_size'] : tuple
        The size of the (HR) output image, by default (1000, 1000).
    kws['chip_size'] : tuple
        The chip/patch size to divide the images into, by default the input size (400, 400).
    kws['normalize_lr'] : bool
        Whether to normalize the low-resolution images.
    kws['interpolation'] : torchvision.transforms.InterpolationMode
        The interpolation mode to resize the images, by default InterpolationMode.BICUBIC.
    kws['scene_classification_filter_values'] : list
        The list of LR scene classification bands to filter the low-resolution images by.
        See: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    kws['scene_classification_filter_threshold'] : float
        The threshold to filter the LR scene classification values by.
    kws['root'] : str
        The root directory of the dataset, by default './data/JIF/'.
    kws['scene_classification_to_color'] : bool
        Whether to convert the scene classification values to colors, by default False.
        See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/
    kws['compute_median_std'] : bool
        Whether to compute the median and standard deviation of the low-resolution images, by default True.
        Once computed, the values can be used to normalise the low-resolution images.
    kws['train_split'] : int
        The number of scenes to use for the train split, by default 50.
    kws['val_split'] : int
        The number of scenes to use for the val split, by default 5.
    kws['test_split'] : int
        The number of scenes to use for the test split, by default 5.
    kws['lr_bands_to_use'] : list
        The list of LR bands (index) to use, by default all.
    kws['radiometry_depth'] : int
        The bit-depth of the HR (SPOT) radiometry bands, by default 12-bit (full).
    kws['data_split_seed'] : int
        The seed to use for the DataLoader random number generator, by default 42.
    kws['pansharpen_hr'] : bool
        Whether to pansharpen the HR image using its panchromatic channel, by default True.
    kws['revisits'] : int
        The number of LR revisits to use, by default 8.
    Returns
    -------
    dict of {str: torch.utils.data.DataLoader}
        The dictionary of dataloaders (hr, lr, lrc).
    """
    kws.setdefault("input_size", (160, 160))
    kws.setdefault("output_size", (1054, 1054))
    kws.setdefault("chip_size", kws["input_size"])

    kws.setdefault("normalize_lr", True)
    kws.setdefault("interpolation", InterpolationMode.BILINEAR)
    kws.setdefault("scene_classification_filter_values", None)
    kws.setdefault("scene_classification_filter_threshold", 0.4)
    kws.setdefault("scene_classification_to_color", False)
    kws.setdefault("lr_bands_to_use", "all")
    kws.setdefault("pansharpen_hr", False)
    kws.setdefault("radiometry_depth", 12)

    kws.setdefault("root", "dataset/")
    kws.setdefault("train_split", None)
    kws.setdefault("val_split", None)
    kws.setdefault("test_split", None)
    kws.setdefault("revisits", 8)

    kws.setdefault("data_split_seed", 42)
    kws.setdefault("compute_median_std", False)
    kws["number_of_revisits"] = kws["revisits"]
    print(f"Using {kws['lr_bands_to_use']} LR bands.")
    kws = set_train_test_val_split(
        kws, os.path.join(kws["root"], "hr_dataset", f"{str(kws['radiometry_depth'])}bit")
    )
    kws["root"] = set_subfolders_for_roots_JIF(kws["root"], kws["radiometry_depth"])
    transforms = make_transforms_JIF(**kws)
    hr_postfix, hr_pan_postfix = set_hr_postfix_based_on_radiometry(
        kws["radiometry_depth"]
    )
    lr_bands = select_sentinel2_bands(kws["lr_bands_to_use"])

    return make_dataloaders(
        subdir={"lr": "", "lrc": "", "hr": "", "hr_pan": ""},#, "metadata": ""},
        bands_to_read={
            "lr": lr_bands,
            "lrc": None,
            "hr": SPOT_RGB_BANDS,
            "hr_pan": None,
            # "metadata": None,
        },
        transforms=transforms,
        number_of_scenes_per_split={
            "train": kws["train_split"],
            "val": kws["val_split"],
            "test": kws["test_split"],
        },
        file_postfix={
            "lr": "-L2A_data.tiff",
            "lrc": "-CLM.tiff",
            "hr": hr_postfix,
            "hr_pan": hr_pan_postfix,
            # "metadata": hr_postfix,
        },
        **kws,
    )


def load_dataset_ProbaV(**kws):
    """ Parses the necessary keyword arguments, creates the transforms and loads the Proba-V dataset.

     Parameters
    ----------
    **kws : dict
        The keyword arguments dictionary from which the necessary arguments are fetched, as generated by train.py.
    kws['input_size'] : tuple
        The size of the (LR) input image, by default (128, 128).
    kws['output_size'] : tuple
        The size of the (HR) output image, by default (384, 384).
    kws['chip_size'] : tuple
        The chip/patch size to divide the images into, by default the input size (128, 128).
    kws['normalize_lr'] : bool
        Whether to normalize the low-resolution images.
    kws['interpolation'] : torchvision.transforms.InterpolationMode
        The interpolation mode to resize the images, by default InterpolationMode.BICUBIC.
    kws['scene_classification_filter_values'] : list
        The list of LR scene classification bands to filter the low-resolution images by.
        See: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    kws['scene_classification_filter_threshold'] : float
        The threshold to filter the LR scene classification values by.
    kws['root'] : str
        The root directory of the dataset, by default './data/probav/'.
    kws['compute_median_std'] : bool
        Whether to compute the median and standard deviation of the low-resolution images, by default True.
        Once computed, the values can be used to normalise the low-resolution images.
    kws['train_split'] : int
        The number of scenes to use for the train split, by default 50.
    kws['val_split'] : int
        The number of scenes to use for the val split, by default 5.
    kws['test_split'] : int
        The number of scenes to use for the test split, by default 5.
    kws['lr_bands_to_use'] : list
        The list of LR bands (index) to use, by default all.
    kws['data_split_seed'] : int
        The seed to use for the DataLoader random number generator, by default 42.
    kws['revisits'] : int
        The number of LR revisits to use, by default 16.
    Returns
    -------
    dict of {str: torch.utils.data.DataLoader}
        The dictionary of dataloaders ('train', 'val', 'test').
    """
    kws.setdefault("input_size", (128, 128))
    kws.setdefault("output_size", (384, 384))
    kws.setdefault("chip_size", kws["input_size"])

    kws.setdefault("normalize_lr", True)
    kws.setdefault("interpolation", InterpolationMode.BICUBIC)
    kws.setdefault("scene_classification_filter_values", None)
    kws.setdefault("scene_classification_filter_threshold", 0.4)

    kws.setdefault("root", "data/probav/")
    kws.setdefault("train_split", None)
    kws.setdefault("val_split", None)
    kws.setdefault("test_split", None)
    kws.setdefault("revisits", 16)

    kws.setdefault("compute_median_std", False)
    kws.setdefault("data_split_seed", 42)

    kws = set_train_test_val_split(kws, os.path.join(kws["root"], "train", "NIR"))
    kws["root"] = set_subfolders_for_root_probav(kws["root"])
    kws["number_of_revisits"] = {
        "lr": kws["revisits"],
        "lrc": kws["revisits"],
        "hr": 1,
        "sm": 1,
    }

    transforms = make_transforms_probav()

    return make_dataloaders(
        subdir={"lr": "", "lrc": "", "hr": "", "sm": ""},
        bands_to_read={
            "lr": S2_ALL_12BANDS["true_color"],
            "lrc": None,
            "hr": None,
            "sm": None,
        },
        transforms=transforms,
        number_of_scenes_per_split={
            "train": kws["train_split"],
            "val": kws["val_split"],
            "test": kws["test_split"],
        },
        file_postfix={
            "lr": "_LR.png",
            "lrc": "_QM.png",
            "hr": "_HR.png",
            "sm": "_SM.png",
        },
        use_tifffile_as_reader={"lr": False, "lrc": False, "hr": False, "sm": False},
        **kws,
    )


def select_sentinel2_bands(lr_bands_to_use):
    """ Selects the Sentinel 2 bands to use based on the keyword argument.
    The keyword argument can be one of the dictionary keys of S2_ALL_12BANDS in datasources.py:
    - 'all'
    - 'true_color'

    Parameters
    ----------
    lr_bands_to_use : str
        The keyword argument to select which Sentinel 2 bands to use.

    Returns
    -------
    list
        The list of Sentinel 2 bands to use, selected from S2_ALL_12BANDS.
    """
    lr_bands = (
        S2_ALL_12BANDS["true_color"]
        if lr_bands_to_use == "true_color"
        else S2_ALL_BANDS
    )
    return lr_bands


def set_hr_postfix_based_on_radiometry(radiometry_depth):
    """ Sets the postfix for the HR images based on the radiometry depth.

    Parameters
    ----------
    radiometry_depth : int
        The bit-depth of the HR (SPOT) radiometry bands.


    Returns
    -------
    _type_
        _description_
    """
    hr_postfix = "_ps_8bit.tiff" if radiometry_depth == 8 else "_ps.tiff"
    hr_pan_postfix = "_pan_8bit.tiff" if radiometry_depth == 8 else "_pan.tiff"
    return hr_postfix, hr_pan_postfix


def make_transforms_SN7(
    input_size=(400, 400),
    output_size=(1000, 1000),
    interpolation=InterpolationMode.BICUBIC,
    normalize_lr=True,
    scene_classification_to_color=False,
    **kws,
):
    """ Make the transforms for the SN7 dataset.
    The transforms normalize and resize the images to the appropriate sizes.

    Parameters
    ----------
    input_size : tuple
        The size of the (LR) input image, by default (400, 400).
    output_size : tuple
        The size of the (HR) output image, by default (1000, 1000).
    interpolation : torchvision.transforms.InterpolationMode, optional
        InterpolationMode to use when resizing the images, by default InterpolationMode.BILINEAR.
    normalize_lr : bool, optional
        A flag to normalize the LR images, by default True.
    scene_classification_to_color : bool, optional
        Converts the scene classification layer values to colors, by default False.
        See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/
    **kws : dict
        The keyword arguments dictionary from which the input_size and output_size are fetched.

    Returns
    -------
    dict of {str : Callable}
        The LR, HR and scene classification transforms.
    """
    lr_bands_to_use = np.array(S2_ALL_12BANDS["true_color"]) - 1  # 1-indexed bands

    if normalize_lr:
        normalize_lr = Normalize(
            mean=S2_SN7_MEAN[lr_bands_to_use], std=S2_SN7_STD[lr_bands_to_use]
        )
    else:
        normalize_lr = Compose([])

    transform_lr = Compose(
        [
            Lambda(lambda lr_revisit: torch.as_tensor(lr_revisit)),
            normalize_lr,
            Resize(size=input_size, interpolation=interpolation, antialias=True),
        ]
    )

    transform_lrc = Compose(
        [
            Lambda(
                lambda lr_scene_classification: torch.as_tensor(lr_scene_classification)
            ),
            # Categorical
            Resize(size=input_size, interpolation=InterpolationMode.NEAREST),
            # NOTE: Categorical to RGB, interferes with FilterData.
            SceneClassificationToColorTransform
            if scene_classification_to_color
            else Compose([]),
        ]
    )

    transform_hr = Compose(
        [
            Lambda(
                lambda hr_revisit: torch.as_tensor(hr_revisit)
                / SN7_MAX_EXPECTED_HR_VALUE
            ),  # sensor-dependent
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(lambda high_res_revisit: high_res_revisit.clamp(min=0, max=1)),
        ]
    )

    return {"lr": transform_lr, "lrc": transform_lrc, "hr": transform_hr}


def make_transforms_JIF(
    input_size=(160, 160),
    output_size=(1054, 1054),
    interpolation=InterpolationMode.BICUBIC,
    normalize_lr=True,
    scene_classification_to_color=False,
    radiometry_depth=12,
    lr_bands_to_use="all",
    **kws,
):
    """ Make the transforms for the JIF dataset.
    The transforms normalize and resize the images to the appropriate sizes.

    Parameters
    ----------
    input_size : tuple
        The size of the (LR) input image, by default (160, 160).
    output_size : tuple
        The size of the (HR) output image, by default (1054, 1054).
    interpolation : torchvision.transforms.InterpolationMode, optional
        InterpolationMode to use when resizing the images, by default InterpolationMode.BILINEAR.
    normalize_lr : bool, optional
        A flag to normalize the LR images, by default True.
    scene_classification_to_color : bool, optional
        Converts the scene classification layer values to colors, by default False.
        See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/
    **kws : dict
        The keyword arguments dictionary from which the input_size and output_size are fetched.

    Returns
    -------
    dict of {str : Callable}
        The LR, HR and scene classification transforms.
    """

    if radiometry_depth == 8:
        maximum_expected_hr_value = SPOT_MAX_EXPECTED_VALUE_8_BIT
    else:
        maximum_expected_hr_value = SPOT_MAX_EXPECTED_VALUE_12_BIT

    transforms = {}
    if lr_bands_to_use == "true_color":
        lr_bands_to_use = np.array(S2_ALL_12BANDS["true_color"]) - 1
    else:
        lr_bands_to_use = np.array(S2_ALL_BANDS) - 1

    if normalize_lr:
        normalize = Normalize(
            mean=JIF_S2_MEAN[lr_bands_to_use], std=JIF_S2_STD[lr_bands_to_use]
        )
    else:
        normalize = Compose([])

    transforms["lr"] = Compose(
        [
            Lambda(lambda lr_revisit: torch.as_tensor(lr_revisit)),
            normalize,
            Resize(size=input_size, interpolation=interpolation, antialias=True),
        ]
    )

    transforms["lrc"] = Compose(
        [
            Lambda(
                lambda lr_scene_classification: torch.as_tensor(lr_scene_classification)
            ),
            # Categorical
            Resize(size=input_size, interpolation=InterpolationMode.NEAREST),
            # Categorical to RGB; NOTE: interferes with FilterData
            SceneClassificationToColorTransform
            if scene_classification_to_color
            else Compose([]),
        ]
    )

    transforms["hr"] = Compose(
        [
            Lambda(
                lambda hr_revisit: torch.as_tensor(hr_revisit.astype(np.int32))
                / maximum_expected_hr_value
            ),
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(lambda high_res_revisit: high_res_revisit.clamp(min=0, max=1)),
        ]
    )

    transforms["hr_pan"] = Compose(
        [
            Lambda(
                lambda hr_panchromatic: torch.as_tensor(
                    hr_panchromatic.astype(np.int32)
                )
                / maximum_expected_hr_value
            ),  # sensor-dependent
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(
                lambda high_res_panchromatic: high_res_panchromatic.clamp(min=0, max=1)
            ),
        ]
    )
    # transforms["metadata"] = Compose([])
    return transforms


def set_subfolders_for_roots_JIF(root, radiometry_depth):
    """ Set the subfolders for LR/HR data roots for the JIF dataset.

    Parameters
    ----------
    root : str
        The root directory of the dataset.

    Returns
    -------
    dict
        The keyword arguments dictionary with the updated roots.
    """
    if radiometry_depth == 8:
        return {
            "lr": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "lrc": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "hr": os.path.join(root, "hr_dataset", "8bit", "*", ""),
            "hr_pan": os.path.join(root, "hr_dataset", "8bit", "*", ""),
            # "metadata": os.path.join(root, "hr_dataset", "8bit", "*", ""),
        }
    else:
        return {
            "lr": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "lrc": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "hr": os.path.join(root, "hr_dataset","12bit", "*", ""),
            "hr_pan": os.path.join(root, "hr_dataset", "12bit", "*", ""),
            # "metadata": os.path.join(root, "hr_dataset",  "*", ""),
        }


def set_train_test_val_split(kws, root):
    """ Set the train/test/val split for the JIF dataset.
    If the split is not set, the dataset is split into 80/10/10.

    Parameters
    ----------
    kws : dict
        The keyword arguments dictionary from which the split is fetched, as generated by train.py.

    Returns
    -------
    dict
        The keyword arguments dictionary with the updated split.
    """
    number_of_aois = len(os.listdir(root))
    if kws["train_split"] is None:
        kws["train_split"] = int(number_of_aois * 0.8)
    if kws["val_split"] is None:
        kws["val_split"] = int(number_of_aois * 0.1)
    if kws["test_split"] is None:
        kws["test_split"] = int(number_of_aois * 0.1)
    return kws


def set_subfolders_for_root_probav(root):
    """ Set the subfolders for LR/HR data roots for the Proba-V dataset.

    Parameters
    ----------
    root : str
        The root directory of the dataset.
    """
    return {
        "lr": os.path.join(root, "train", "*", "*"),
        "lrc": os.path.join(root, "train", "*", "*"),
        "hr": os.path.join(root, "train", "*", "*"),
        "sm": os.path.join(root, "train", "*", "*"),
    }


def make_transforms_probav():
    """ Make the transforms for the Proba-V dataset.

    Returns
    -------
    dict
        The dictionary with the transforms for LR/LRC/HR/SM.
    """
    transform_to_tensor = Compose([Lambda(lambda x: torch.as_tensor(x))])
    transforms = {
        "lr": transform_to_tensor,
        "lrc": transform_to_tensor,
        "sm": transform_to_tensor,
        "hr": transform_to_tensor,
    }
    return transforms


if __name__ == "__main__":
    """ Median and std of dataset bands. """
    dataset_lr = SatelliteDataset(
        root=ROOT_JIF_DATA_TRAIN, subdir="S2", number_of_revisits=10
    )
    dataset_hr = SatelliteDataset(
        root=ROOT_JIF_DATA_TRAIN, subdir="images", number_of_revisits=1
    )
    print(f"S2 median / std: {dataset_lr.compute_median_std()}")
    print(f"PlanetScope median / std: {dataset_hr.compute_median_std()}")
