from gc import collect
from eolearn.core import FeatureType, EOWorkflow, linearly_connect_tasks, OutputTask
from eolearn.io import SentinelHubInputTask, ExportToTiffTask
from sentinelhub import BBox, CRS, DataCollection
from tqdm.auto import tqdm
import os
from dataset_generation.SentinelHub import SentinelHub
from pathlib import Path
from datetime import timedelta
import json
import concurrent
from concurrent.futures import as_completed
from glob import glob
from pathlib import Path
from dataset_generation.Visualiser import Visualiser

import tifffile as tiff
from PIL import Image
import numpy as np


class SentinelDownloader:
    """ Downloads Sentinel2 imagery from SentinelHub.
    """

    def __init__(
        self,
        aois,
        selected_revisits,
        root_path="data/downloads/Sentinel2",
        bands="ALL",
        metadata="ALL",
        download_L2A=True,
        download_L1C=True,
        download_masks=True,
        download_metadata=True,
        overwrite=False,
        crs=CRS.WGS84,
    ):
        """ Initialises the SentinelDownloader class.
            
        Parameters
        ----------
        aois : pandas.DataFrame
            AOIs to download (must have a bounding box).
        selected_revisits : pandas.DataFrame
            Selected revisits to download for each AOI.
        root_path : str, optional
            Root path to save the downloaded data, by default "data/downloads/Sentinel2".
        bands : str, optional
            Bands to download, by default "ALL".
            Can be a dict specifying the bands for L1C and L2A, or a list if it's the same bands for both.
            A list of available bands can be found here: 
            https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial"
        metadata : str, optional
            Metadata to download, by default "ALL".
        download_L2A : bool, optional
            Whether to download L2A data, by default True.
        download_L1C : bool, optional
            Whether to download L1C data, by default True.
        download_masks : bool, optional
            Whether to download data and cloud masks, by default True.
        download_metadata : bool, optional
            Whether to download metadata (e.g. view/sun azimuth/zenit angles), by default True.
        overwrite : bool, optional
            Whether to overwrite if some data is already downloaded, by default False.
        crs : _type_, optional
            CRS of the data, by default CRS.WGS84.
        """
        self.aois = aois
        self.selected_revisits = selected_revisits
        # self.merge_aois_with_selected_revisits()
        self.root_path = root_path
        self.set_bands(bands)
        self.set_metadata(metadata)
        self.download_L2A = download_L2A
        self.download_L1C = download_L1C
        self.download_masks = download_masks
        self.download_metadata = download_metadata
        self.overwrite = overwrite
        self.crs = crs
        #self.download_revisits_multithreaded()
        #self.revisits_paths = self.get_downloaded_revisits_paths()
        #self.visualiser = Visualiser(self.selected_revisits)

    def merge_aois_with_selected_revisits(self):
        """ Merges the AOIs with their selected revisits. """
        self.revisits_to_download = self.aois.merge(
            self.selected_revisits, left_index=True, right_index=True
        )
        if "area_y" in self.revisits_to_download.columns:
            self.revisits_to_download.drop(columns=["area_y"], inplace=True)
            self.revisits_to_download.rename(columns={"area_x": "area"}, inplace=True)
        if "name_x" in self.revisits_to_download.columns:
            self.revisits_to_download.drop(columns=["name_x"], inplace=True)
            self.revisits_to_download.rename(columns={"name_y": "name"}, inplace=True)
        if "bounds_x" in self.revisits_to_download.columns:
            self.revisits_to_download.drop(columns=["bounds_x"], inplace=True)
            self.revisits_to_download.rename(
                columns={"bounds_y": "bounds"}, inplace=True
            )
        if "target_date_x" in self.revisits_to_download.columns:
            self.revisits_to_download.drop(columns=["target_date_x"], inplace=True)
            self.revisits_to_download.rename(
                columns={"target_date_y": "target_date"}, inplace=True
            )
        self.revisits_to_download.set_index("name", inplace=True)

    def visualise_downloaded_revisits(self, save_gif=False):
        """ Visualises the downloaded revisits.

        Parameters
        ----------
        save_gif : bool, optional
            If true, the visualisation will be saved to a gif (data/visualisations/s2_revisits.gif) and previewed, by default False.
            If false, returns an interactive widget with the list of revisit images visualised.

        Returns
        -------
        IPython.display.Image or ipywidgets.interact
            The visualisation of the downloaded revisits.
        """
        return self.visualiser.visualise_downloaded_revisits(
            self.revisits_paths, save_gif
        )

    def get_downloaded_revisits_paths(self):
        """ Returns the paths to the downloaded revisits.

        Returns
        -------
        dict
            The paths to the downloaded revisits.
        """
        self.revisits_paths = glob(
            str(Path(self.root_path, "*", "L2A", "*-L2A_data.tiff"))
        )
        return self.revisits_paths

    def get_revisit_rgb(self, revisit):
        """ Returns the RGB image of a revisit.

        Parameters
        ----------
        revisit : int
            The index of the revisit.

        Returns
        -------
        PIL.Image
            The RGB image of the revisit.
        """
        image = tiff.imread(self.revisits_paths[revisit])
        rgb = image[:, :, 1:4][:, :, ::-1]
        rgb = rgb / np.max(rgb)
        rgb = np.uint8(255 * rgb)
        return Image.fromarray(rgb)

    def download_revisits(self):
        """ Downloads the revisits and verifies all files have successfully downloaded. """
        self.print_download_settings()
        print(f"\tThe download is single-threaded.")
        for _, revisit in tqdm(
            self.revisits_to_download.iterrows(),
            desc="Downloading Sentinel2 revisits",
            total=len(self.revisits_to_download),
        ):
            self.download_revisit(revisit)
        self.verify_download()

    def download_revisits_multithreaded(self):
        """ Multi-threaded download of the revisits. """
        self.print_download_settings()
        print(f"\tThe download is multi-threaded.")
        for aoi_name in self.revisits_to_download.index:
            aoi_path = Path(self.root_path, aoi_name)
            if not aoi_path.exists():
                os.makedirs(str(aoi_path))
        with concurrent.futures.ThreadPoolExecutor() as executor_thread:
            downloads = [
                executor_thread.submit(self.download_revisit, revisit)
                for index, revisit in self.revisits_to_download.iterrows()
            ]
            for task in tqdm(
                as_completed(downloads),
                total=len(downloads),
                desc="Downloading Sentinel2 revisits",
            ):
                pass
        self.verify_download()

    def print_download_settings(self):
        """ Prints out the different download settings/parameters being used. """
        print(
            f"Downloading {len(self.revisits_to_download)} revisits with the following parameters:"
        )
        print(f"\tRoot path: {self.root_path}")
        print(f"\tOverwrite: {self.overwrite}")
        print(f"\tDownload L1C: {self.download_L1C}")
        print(f"\tDownload L2A: {self.download_L2A}")
        print(f"\tDownload masks: {self.download_masks}")
        print(f"\tDownload metadata: {self.download_metadata}")
        print(f"\tBands: {self.bands}")
        print(f"\tMetadata: {self.metadata}")

    def verify_download(self):
        """ Verifies all files have successfully downloaded. """
        missing = False
        for aoi_name in set(self.revisits_to_download.index):
            number_of_revisits = len(self.revisits_to_download.loc[aoi_name])
            expected_number_of_files = (
                len(self.metadata) + 1
            ) * number_of_revisits + number_of_revisits
            aoi_path = Path(self.root_path, aoi_name)
            if not aoi_path.exists():
                print(f"AOI {aoi_name} not downloaded")

            number_of_downloaded_L2A = len(os.listdir(f"{str(aoi_path)}/L2A"))
            number_of_downloaded_L1C = len(os.listdir(f"{str(aoi_path)}/L1C"))

            if self.download_L1C:
                if number_of_downloaded_L1C != expected_number_of_files:
                    print(
                        f"AOI {aoi_name} missing {expected_number_of_files-number_of_downloaded_L1C} L1C files. Please rerun download with overwrite=False to download the missing files."
                    )
                    missing = True
            if self.download_L2A:
                if number_of_downloaded_L2A != expected_number_of_files:
                    print(
                        f"AOI {aoi_name} missing {expected_number_of_files-number_of_downloaded_L2A} L2A files. Please rerun download with overwrite=False to download the missing files."
                    )
                    missing = True
        if not missing:
            print("All files downloaded")

    def download_revisit(self, revisit):
        """Download Sentinel-2 L1C and L2A data for a revisit, defined by a bounding box and a date.
        The revisit data is saved in a subfolder of the given folder, named after the revisit index.

        Parameters
        ----------
        revisit : pandas.Series
            A revisit defined by a bounding box and a date.
        folder : str
            The folder where the data will be saved.
        """
        path = self.root_path
        if not path.endswith("/"):
            path += "/"
        if not os.path.exists(path):
            os.makedirs(path)

        filename = f"{revisit.name}-{revisit.n}"
        filename = Path(filename)
        filename = str(filename.with_suffix(""))

        path = f"{path}{revisit.name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + "L1C/"):
            os.makedirs(path + "L1C/")
        if not os.path.exists(path + "L2A/"):
            os.makedirs(path + "L2A/")
        
        l2a_filename = f"{filename}-L2A_data.tiff"
        l1c_filename = f"{filename}-L1C_data.tiff"
        
        if self.download_L1C and (
            not os.path.isfile(f"{path}L1C/{l1c_filename}") or self.overwrite
        ):
            self.download_sentinel2(revisit, path + "L1C/", filename, collection="L1C")
        if self.download_L2A and (
            not os.path.isfile(f"{path}L2A/{l2a_filename}") or self.overwrite
        ):
            self.download_sentinel2(revisit, path + "L2A/", filename, collection="L2A")
        return True

    def download_sentinel2(self, revisit, folder, filename, collection="L2A"):
        """Download Sentinel-2 L2A data for a revisit, defined by a bounding box and a date.

        Parameters
        ----------
        revisit : pandas.Series
            A revisit defined by a bounding box and a date.
        folder : str
            The folder where the data will be saved.
        """
        if collection == "L2A":
            data_collection = DataCollection.SENTINEL2_L2A
            bands = self.bands["L2A"]
            bands_feature = (FeatureType.DATA, "L2A_data")
        elif collection == "L1C":
            data_collection = DataCollection.SENTINEL2_L1C
            bands = self.bands["L1C"]
            bands_feature = (FeatureType.DATA, "L1C_data")
        else:
            raise ValueError(f"Collection {collection} is not supported.")
        bbox = BBox(revisit["bounds"], crs=CRS.WGS84)
        date = revisit["datetime"]
        get_sentinel_image = SentinelHubInputTask(
            data_collection=data_collection,
            bands=bands,
            bands_feature=bands_feature,
            additional_data=self.metadata,
            resolution=10,
            maxcc=1,
            time_difference=timedelta(hours=2),
            config=SentinelHub.get_config(),
            max_threads=3,
        )

        save_task = OutputTask("eopatch")
        workflow_nodes = linearly_connect_tasks(get_sentinel_image, save_task)
        workflow = EOWorkflow(workflow_nodes)
        result = workflow.execute(
            {workflow_nodes[0]: {"bbox": bbox, "time_interval": (date, date)},}
        )

        eopatch = result.outputs["eopatch"]
        self.save_patch_to_tiff(eopatch, filename, folder)
        self.save_metadata_alt(revisit, filename, folder)

    def save_metadata_alt(self, revisit, filename, path):
        """Save metadata for a revisit to a json file in the given path under the given order name and index.

        Parameters
        ----------
        revisit : pandas.Series
            A revisit Series containing the metadata (cloud_cover, highres_date, delta, area, date).
        date : datetime.datetime
            The date of the revisit.
        path : str
            The path where the metadata will be saved.
        order_name : str
            The name of the order under which the metadata will be saved.
        i : int
            The index of the revisit in the order under which the metadata will be saved.
        """
        try:
            metadata = {}
            metadata["cloud_cover"] = revisit["cloud_cover"]
            metadata["target_date"] = str(revisit["target_date"])
            metadata["delta"] = str(revisit["delta"])
            metadata["area"] = revisit["area"]
            metadata["datetime"] = revisit["datetime"]
            with open(f"{path}/{filename}.metadata", "w") as text_file:
                text_file.write(json.dumps(metadata, default=str))
        except Exception as e:
            print(e)

    def save_patch_to_tiff(self, eopatch, scene_name, folder):
        """Save the downloaded revisit patch (eopatch) to a tiff file under the given folder and scene name.

        Parameters
        ----------
        eopatch : eolearn.core.EOPatch
            The eopatch containing the revisit data.
        scene_name : str
            The name of the scene.
        folder : str
            The folder where the data will be saved.
        """
        for name, data in eopatch.data.items():
            feature = (FeatureType.DATA, name)
            band_indices = (0, data.shape[-1])
            image_dtype = data.dtype
            for i, x in enumerate(eopatch.timestamp):
                task = ExportToTiffTask(
                    feature=feature,
                    folder=folder,
                    band_indices=band_indices,
                    date_indices=[i],
                    image_dtype=image_dtype,
                )
                filename = f"{scene_name}-{name}.tiff"
                task.execute(eopatch, filename=filename)
        for name, mask in eopatch.mask.items():
            feature = (FeatureType.MASK, name)
            band_indices = (0, data.shape[-1])
            image_dtype = data.dtype
            for i, x in enumerate(eopatch.timestamp):
                task = ExportToTiffTask(
                    feature=feature,
                    folder=folder,
                    band_indices=band_indices,
                    date_indices=[i],
                    image_dtype=image_dtype,
                )
                filename = f"{scene_name}-{name}.tiff"
                task.execute(eopatch, filename=filename)

    def set_bands(self, bands):
        """ Set the bands to be downloaded.

        Parameters
        ----------
        bands : str, dict or list
            The bands to be downloaded. Possible values are:
            - "ALL": download all bands for L1C and L2A
            - a dict of separate bands for L1C and L2A
            - a list of bands used both for L1C and L2A

        Raises
        ------
        Exception
            If the bands aren't a list, dict or 'ALL'.
        """
        if bands == "ALL":
            self.bands = {
                "L2A": [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                ],
                "L1C": [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B10",
                    "B11",
                    "B12",
                ],
            }
        elif isinstance(bands, list) or isinstance(bands, dict):
            self.bands = bands
        else:
            raise Exception(
                """Bands must be a list of specifying the bands to download for L1C or L2A, a dict specifying the bands for each or 'ALL'.\n 
                A list of bands can be found here: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial"""
            )

    def set_metadata(self, metadata):
        """ Set the metadata to be downloaded.

        Parameters
        ----------
        metadata : str or list of str
            The metadata to be downloaded. Possible values are:
            - "ALL": download all metadata for L1C and L2A
            - a list of metadata used both for L1C and L2A

        Raises
        ------
        Exception
            If the metadata isn't 'ALL' or a list of available metadata keys: 
            dataMask, CLM, CLP, viewZenithMean, viewAzimuthMean, sunAzimuthAngles, sunZenithAngles
        """
        # TODO (ori): Add SCL and other bands for L2A?
        available_metadata = {
            "dataMask": FeatureType.MASK,
            "CLM": FeatureType.MASK,
            "CLP": FeatureType.MASK,
            "viewZenithMean": FeatureType.DATA,
            "viewAzimuthMean": FeatureType.DATA,
            "sunAzimuthAngles": FeatureType.DATA,
            "sunZenithAngles": FeatureType.DATA,
        }
        if metadata == "ALL":
            self.metadata = [
                feature_name[::-1] for feature_name in available_metadata.items()
            ]

        elif isinstance(metadata, list) and all(isinstance(x, str) for x in metadata):
            self.metadata = [(available_metadata[x], x) for x in metadata]
        else:
            raise Exception(
                'Metadata must be a list of strings, or "ALL".\n Available metadata are: {}'.format(
                    list(available_metadata.keys())
                )
            )

