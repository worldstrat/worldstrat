from urllib.request import urlretrieve
from urllib.error import HTTPError
from pathlib import Path
import subprocess
import zipfile
import os
from tqdm.notebook import tqdm
import rasterio
from numpy import nan

LC_FILE_PATH = "dataset_generation/stratification_datasets/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.tif"
SMOD_FILE_PATH = "dataset_generation/stratification_datasets/smod/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0.tif"


class StratificationDatasets:
    """ Loads the datasets used to stratify sampled points (ESA CCI LCCS/IPCC, GLHS-SMOD).
    If the files are not found, provides automated download and extraction or instructions to download them manually. """

    def __init__(self):
        """ Initializes the class by checking if the datasets are downloaded and loading them. """
        StratificationDatasets.check_landcover_files_exist()
        StratificationDatasets.check_smod_files_exist()
        self.smod_dataset = StratificationDatasets.load_smod_dataset()
        self.landcover_dataset = StratificationDatasets.load_landcover_dataset()

    @staticmethod
    def load_smod_dataset(SMOD_FILE=SMOD_FILE_PATH):
        """Loads the SMOD dataset.

        Returns
        -------
        rasterio.DatasetReader
            DatasetReader object containing the SMOD dataset.
        """
        StratificationDatasets.check_smod_files_exist(SMOD_FILE)
        return rasterio.open(SMOD_FILE)

    @staticmethod
    def load_landcover_dataset(LC_FILE=LC_FILE_PATH):
        """Loads the landcover dataset.

        Returns
        -------
        rasterio.DatasetReader
            DatasetReader object containing the landcover dataset.
        """
        StratificationDatasets.check_landcover_files_exist(LC_FILE)
        return rasterio.open(LC_FILE)

    @staticmethod
    def download_and_prepare_landcover_dataset(
        download_link="https://download-0009-clone.copernicus-climate.eu/cache-compute-0009/cache/data6/dataset-satellite-land-cover-276f3dbb-e889-40fc-85ef-b0bb5ea8c25f.zip?download",
    ):
        """Downloads the landcover dataset and prepares it for use.
        """
        StratificationDatasets.download_dataset(
            download_path="dataset_generation/stratification_datasets/landcover/landcover.zip",
            download_link=download_link,
            description="Downloading land cover dataset",
        )
        StratificationDatasets.unzip_dataset(
            zip_path="dataset_generation/stratification_datasets/landcover/landcover.zip",
            unzip_path="dataset_generation/stratification_datasets/landcover",
        )
        StratificationDatasets.convert_landcover_dataset()

    @staticmethod
    def download_and_prepare_smod_dataset():
        """Downloads the SMOD dataset and prepares it for use.
        """
        StratificationDatasets.download_dataset(
            download_path="dataset_generation/stratification_datasets/smod/smod.zip",
            download_link="https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_POP_GLOBE_R2019A/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K/V2-0/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0.zip",
            description="Downloading GSHL-SMOD dataset.",
        )
        StratificationDatasets.unzip_dataset(
            zip_path="dataset_generation/stratification_datasets/smod/smod.zip",
            unzip_path="dataset_generation/stratification_datasets/smod",
        )

    @staticmethod
    def download_dataset(
        download_path, download_link, description="Downloading dataset"
    ):
        """
        Download a dataset from the provided URL and save it to the provided path.
        Implements a custom download progress bar hook that urlretrieve uses to report progress.

        Parameters
        ----------
        download_path : str
            Path to save the downloaded dataset to.
        download_link : str
            URL to download the dataset from.
        description : str, optional
            Description string to display in the download progress bar, by default "Downloading dataset"
        """
        parent_folder = Path(download_path).parent
        Path(parent_folder).mkdir(parents=True, exist_ok=True)
        progress_bar = tqdm(total=100, desc=description)

        def download_progress_hook(count, blockSize, totalSize):
            percent = int(count * blockSize * 100 / totalSize)
            progress_bar.update(percent - progress_bar.n)

        try:
            print(
                f"Downloading the dataset to {download_path}. This might take a few minutes."
            )
            urlretrieve(download_link, download_path, reporthook=download_progress_hook)
        except HTTPError as error:
            if error.code == 404 and "landcover" in download_path:
                raise Warning(
                    "The download link has expired. Please generate a new one by visiting https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=form and pass it to the download_and_prepare_landcover_dataset() function. The code was written using the 2020 v2.1.1 version of the dataset."
                )

    @staticmethod
    def unzip_dataset(zip_path, unzip_path):
        """Unzips a zip file to a given path.

        Parameters
        ----------
        zip_path : str
            Path to the zip file.
        unzip_path : str
            Path to the folder where the zip file will be unzipped.
        """
        print(f"Unzipping {zip_path} to {unzip_path}. This might take a minute.")
        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall(unzip_path)

    @staticmethod
    def convert_landcover_dataset(
        input_file="dataset_generation/stratification_datasets/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc",
        output_file=LC_FILE_PATH,
    ):
        """Converts the land cover dataset from netCDF to GeoTIFF.

        Parameters
        ----------
        input_file : str, optional
            The path to the downloaded original ESA CCI LC file, by default "dataset_generation/stratification_datasets/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"
        output_file : str, optional
            The path to the output GeoTIFF file, by default LC_FILE_PATH
        """
        print(
            f"Converting {input_file} to {output_file} using gdalwarp. This might take a few minutes."
        )
        if os.path.isfile(output_file):
            print(f"{output_file} already exists. Skipping conversion.")
            return
        command = f"gdalwarp -of Gtiff -co COMPRESS=LZW -co TILED=YES -ot Byte -te -180.0000000 -90.0000000 180.0000000 90.0000000 -tr 0.002777777777778 0.002777777777778 -t_srs EPSG:4326 NETCDF:{input_file}:lccs_class {output_file}"
        output = subprocess.run(
            command, check=True, capture_output=True, shell=True, cwd=os.getcwd()
        ).stdout
        print(output)

    @staticmethod
    def check_landcover_files_exist(LC_FILE=LC_FILE_PATH):
        """Checks if the ESA CCI LC file exists at the expected location. If not, explains how to change the location and offers a function that downloads it and prepares it for usage.
        Also offers instructions on how to manually download and prepare the files.


        Parameters
        ----------
        LC_FILE : str, optional
            The expected location of the ESA CCI LC dataset file, by default LC_FILE_PATH.

        Returns
        -------
        bool
            True if the file exists at the expected location, raises a FileNotFoundError otherwise.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the expected location.
        """
        if os.path.isfile(LC_FILE):
            return True
        else:
            raise FileNotFoundError(
                f"""Landcover file not found at the expected location {LC_FILE}. Please run the StratificationDatasets.download_and_prepare_landcover_dataset() function to download and prepare the dataset for usage. \n
                If the file exists, but isn't at the expected location, please move it to the expected location or edit the path in the StratificationDatasets class. \n
                Instructions to prepare the file manually:
                Please download it from https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=form. The code was written while using the 2020 v2.1.1 version.
                Unzip it, and convert it using the following command:
                gdalwarp - of Gtiff - co COMPRESS=LZW - co TILED=YES - ot Byte - te - 180.0000000 - 90.0000000 180.0000000 90.0000000 - tr 0.002777777777778 0.002777777777778 - t_srs EPSG: 4326 NETCDF: C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc: lccs_class C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.tif"""
            )

    @staticmethod
    def check_smod_files_exist(SMOD_FILE=SMOD_FILE_PATH):
        """Checks if the SMOD file exists at the expected location. If not, explains how to change the location and offers a function that downloads it and prepares it for usage.
        Also offers instructions on how to manually download and prepare the files.

        Parameters
        ----------
        SMOD_FILE : str, optional
            The expected location of the SMOD dataset file, by default SMOD_FILE_PATH.

        Returns
        -------
        bool
            True if the file exists at the expected location, raises a FileNotFoundError otherwise.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist at the expected location.
        """
        if os.path.isfile(SMOD_FILE):
            return True
        else:
            raise FileNotFoundError(
                f"""GHSL-SMOD file {SMOD_FILE} not found. Please run the StratificationDatasets.download_and_prepare_smod_dataset() function to download and prepare the dataset for usage.\n 
                If the file exists, but isn't at the expected location, please move it to the expected location or edit the path in the StratificationDatasets class. \n
                Instructions to prepare the file manually:
                Please download it from https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_POP_GLOBE_R2019A/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K/V2-0/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0.zip.
                Make an smod folder and unzip it there (smod/{SMOD_FILE})."""
            )


# List of ESA CCI LC class IDs: http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
CLASSES = [
    10,
    11,
    12,
    20,
    30,
    40,
    50,
    60,
    61,
    62,
    70,
    71,
    72,
    80,
    81,
    82,
    90,
    100,
    160,
    170,
    110,
    130,
    180,
    190,
    120,
    121,
    122,
    140,
    150,
    151,
    152,
    153,
    200,
    201,
    202,
    210,
    220,
]

LCCS_classes = {
    0: "No Data",
    10: "Cropland, rainfed",
    11: "Herbaceous cover",
    12: "Tree or shrub cover",
    20: "Cropland, irrigated or post‐flooding",
    30: "Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)",
    40: "Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)",
    50: "Tree cover, broadleaved, evergreen, closed to open (>15%)",
    60: "Tree cover, broadleaved, deciduous, closed to open (>15%)",
    61: "Tree cover, broadleaved, deciduous, closed (>40%)",
    62: "Tree cover, broadleaved, deciduous, open (15‐40%)",
    70: "Tree cover, needleleaved, evergreen, closed to open (>15%)",
    71: "Tree cover, needleleaved, evergreen, closed (>40%)",
    72: "Tree cover, needleleaved, evergreen, open (15‐40%)",
    80: "Tree cover, needleleaved, deciduous, closed to open (>15%)",
    81: "Tree cover, needleleaved, deciduous, closed (>40%)",
    82: "Tree cover, needleleaved, deciduous, open (15‐40%)",
    90: "Tree cover, mixed leaf type (broadleaved and needleleaved)",
    100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
    110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
    120: "Shrubland",
    121: "Evergreen shrubland",
    122: "Deciduous shrubland",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation (tree, shrub, herbaceous cover) (<15%)",
    152: "Sparse shrub (<15%)",
    153: "Sparse herbaceous cover (<15%)",
    160: "Tree cover, flooded, fresh or brakish water",
    170: "Tree cover, flooded, saline water",
    180: "Shrub or herbaceous cover, flooded, fresh/saline/brakish water",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice",
}

LCCS_TO_IPCC = {
    0: "None",
    10: "Agriculture",
    11: "Agriculture",
    12: "Agriculture",
    20: "Agriculture",
    30: "Agriculture",
    40: "Agriculture",
    50: "Forest",
    60: "Forest",
    61: "Forest",
    62: "Forest",
    70: "Forest",
    71: "Forest",
    72: "Forest",
    80: "Forest",
    81: "Forest",
    82: "Forest",
    90: "Forest",
    100: "Forest",
    110: "Grassland",
    120: "Other",
    121: "Other",
    122: "Other",
    130: "Grassland",
    140: "Other",
    150: "Other",
    151: "Other",
    152: "Other",
    153: "Other",
    160: "Forest",
    170: "Forest",
    180: "Wetland",
    190: "Settlement",
    200: "Other",
    201: "Other",
    202: "Other",
    210: "Water",
    220: "None",
}

SMOD_classes = {
    10: "Water",
    11: "Rural: Very Low Dens",
    12: "Rural: Low Dens",
    13: "Rural: cluster",
    21: "Urban: Suburban",
    22: "Urban: Semi-dense",
    23: "Urban: Dense",
    30: "Urban: Centre",
    0: "None",
    nan: "None",
    -200: "Unknown",
}
