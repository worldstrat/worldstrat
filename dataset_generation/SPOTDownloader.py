from dataset_generation.SentinelHub import *
from dataset_generation.SPOTOrder import *

import warnings
import rasterio
import os
import numpy as np
from PIL import Image
from numpy import uint16
import math
from pathlib import Path
from tifffile import imread
from glob import glob
from tqdm.notebook import tqdm

class SPOTDownloader:
    """ Downloads Sentinel2 imagery from SentinelHub.
    """

    @staticmethod
    def all_bands_evalstring(
        input_bands=["B0", "B1", "B2", "B3", "PAN"], output_bands=5
    ):
        """Returns the evalstring that downloads all bands.

        Parameters
        ----------
        input_bands : list, optional
            The input bands to be downloaded, by default ["B0", "B1", "B2", "B3", "PAN"].
        output_bands : int, optional
            The number of output bands, by default 5.

        Returns
        -------
        str
            The evalstring that downloads all bands.
        """
        return (
            """
            //VERSION=3
            function setup() {
                return {
                    input:[{bands:"""
            + str(input_bands)
            + """, units:"DN"}],
                    output: { bands: """
            + str(output_bands)
            + """,
                    sampleType: "UINT16"}
                }
            }
            function evaluatePixel(samples) {
                return [samples.B2, samples.B1, samples.B0, samples.B3, samples.PAN]
            }
        """
        )

    @staticmethod
    def rgbn_evalstring(input_bands=["B0", "B1", "B2", "B3"], output_bands=4):
        """Returns the evalstring that downloads the RGBN bands.

        Parameters
        ----------
        input_bands : list, optional
            The input bands to be downloaded, by default ["B0", "B1", "B2", "B3"].
        output_bands : int, optional
            The number of output bands, by default 4.

        Returns
        -------
        str
            The evalstring that downloads the RGBN bands.
        """
        return (
            """
            //VERSION=3
            function setup() {
                return {
                    input:[{bands:"""
            + str(input_bands)
            + """, units:"DN"}],
                    output: { bands: """
            + str(output_bands)
            + """,
                    sampleType: "UINT16"}
                }
            }
            function evaluatePixel(samples) {
                return [samples.B2, samples.B1, samples.B0, samples.B3]
            }
        """
        )

    @staticmethod
    def pan_evalstring(input_bands=["PAN"], output_bands=1):
        """Returns the evalstring that downloads the PAN band.

        Parameters
        ----------
        input_bands : list, optional
            The input bands to be downloaded, by default ["PAN"].
        output_bands : int, optional
            The number of output bands, by default 1.

        Returns
        -------
        str
            The evalstring that downloads the PAN band.
        """
        return (
            """
            //VERSION=3
            function setup() {
                return {
                    input:[{bands:"""
            + str(input_bands)
            + """, units:"DN"}],
                    output: { bands: """
            + str(output_bands)
            + """,
                    sampleType: "UINT16"}
                }
            }
            function evaluatePixel(samples) {
                return [samples.PAN]
            }
        """
        )

    @staticmethod
    def download_order(order, size, evalscript=None):
        """Processes and returns a truecolor image using SentinelHub for a given order.

        Parameters
        ----------
        order : pandas.Series
            A completed order containing the imagery that will be downloaded.
        image_width : int, optional
            The wanted image width in px, by default 2500 (max).
        image_height : int, optional
            The wanted image height in px, by default 2500 (max).

        Returns
        -------
        dict
            The SentinelHub API response containing the truecolor PNG (RGBA).
        """
        evalscript = (
            SPOTDownloader.all_bands_evalstring() if evalscript is None else evalscript
        )
        if pd.isna(order["input.bounds.properties.crs"]):
            order[
                "input.bounds.properties.crs"
            ] = "http://www.opengis.net/def/crs/EPSG/0/4326"
        oauth_session = SentinelHub.generate_oauth_session()
        response = oauth_session.post(
            "https://services.sentinel-hub.com/api/v1/process",
            json={
                "input": {
                    "bounds": {
                        "properties": {"crs": order["input.bounds.properties.crs"]},
                        "bbox": order["input.bounds.bbox"],
                    },
                    "data": [{"type": f"byoc-{order['collectionId']}"}],
                },
                "output": {
                    "width": size,
                    "height": size,
                    "responses": [{"format": {"type": "image/tiff"}}],
                },
                "evalscript": evalscript,
            },
        )
        from rasterio.io import MemoryFile

        try:
            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset:
                    data_array = dataset.read()
            return data_array
        except:
            raise Exception(response.content)

    @staticmethod
    def download_highres_image_to_file(order, path, filename):
        """Downloads a high resolution revisit to an image file.

        Parameters
        ----------
        order : pandas.Series
            A completed order containing the imagery that will be downloaded.
        size : int
            The wanted image width and height in px.
        path : str
            The path to the image file.
        filename : str
            The filename under which the image will be saved.
        """
        # TODO: support non-square orders
        # TODO: get the band resolutions programatically?

        if not path.endswith("/"):
            path += "/"
        if not os.path.exists(path):
            os.makedirs(path)

        filename = Path(filename)
        filename = str(filename.with_suffix(""))  # Remove extension

        if not os.path.exists(path + filename):
            os.makedirs(path + filename)

        pan_filename = f"{filename}/{filename}_pan.tiff"
        rgbn_filename = f"{filename}/{filename}_rgbn.tiff"
        if os.path.isfile(f"{path}{pan_filename}"):
            return True
        if os.path.isfile(f"{path}{rgbn_filename}"):
            return True
        area = float(order["sqkm"])
        side_m = math.sqrt(area) * 1000
        rgbn_resolution = 6  # metres/px
        pan_resolution = 1.5  # metres/px
        rgbn_size = int(side_m / rgbn_resolution)
        pan_size = int(side_m / pan_resolution)

        rgbn_image = SPOTDownloader.download_order(
            order, rgbn_size, evalscript=SPOTDownloader.rgbn_evalstring()
        )
        pan_image = SPOTDownloader.download_order(
            order, pan_size, evalscript=SPOTDownloader.pan_evalstring()
        )

        crs = order["input.bounds.properties.crs"]
        SPOTDownloader.save_image_data_to_tiff(rgbn_image, path, rgbn_filename, crs)
        SPOTDownloader.save_image_data_to_tiff(pan_image, path, pan_filename, crs)
        return True

    @staticmethod
    def save_image_data_to_tiff(image, path, filename, crs, dtype=uint16):
        warnings.filterwarnings("ignore")
        count, width, height = image.shape
        with rasterio.open(
            f"{path}{filename}",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            nodata=None,
            transform=None,
        ) as dst:
            dst.write(image)

    @staticmethod
    def download_spot_for_entire_collection(collection_id, destination_path):
        """Downloads the entire collection to a given path.

        Parameters
        ----------
        collection_id : int
            The ID of the collection to download.
        destination_path : str
            The path to the folder where the images will be saved.
        """
        orders = SPOTOrder.get_all_orders_for_collection(collection_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            res = [
                executor.submit(
                    SPOTDownloader.download_highres_image_to_file,
                    order,
                    destination_path,
                    order["name"],
                )
                for _, order in orders.iterrows()
            ]
            for task in tqdm(as_completed(res), total=len(res)):
                pass
            concurrent.futures.wait(res)  # Wait for all threads to finish
        return res

    @staticmethod
    def download_spot_for_orders(orders, destination_path="data/downloads/SPOT"):
        """Downloads the orders to a given path.

        Parameters
        ----------
        orders : pandas.DataFrame
            The orders to download.
        destination_path : str
            The path to the folder where the images will be saved.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            res = [
                executor.submit(
                    SPOTDownloader.download_highres_image_to_file,
                    order,
                    destination_path,
                    order["name"],
                )
                for _, order in orders.iterrows()
            ]
            for task in tqdm(as_completed(res), total=len(res)):
                pass
            concurrent.futures.wait(res)  # Wait for all threads to finish
        return res

    @staticmethod
    def get_revisit_rgb(revisit_path):
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
        image = imread(revisit_path)
        rgb = image[:, :, 0:3][:, :, ::-1]
        rgb = rgb / np.max(rgb)
        rgb = np.uint8(255 * rgb)
        return Image.fromarray(rgb)

    @staticmethod
    def get_downloaded_revisits_paths(root_path):
        """ Returns the paths to the downloaded revisits.

        Returns
        -------
        dict
            The paths to the downloaded revisits.
        """
        revisits_paths = glob(str(Path(root_path, "*", "*_rgbn.tiff")))
        return revisits_paths

    @staticmethod
    def visualise_downloaded_revisits(root_path="data/downloads/SPOT/", save_gif=False):
        """ Visualises the downloaded revisits.

        Parameters
        ----------
        save_gif : bool, optional
            If true, the visualisation will be saved to a gif (revisits.gif) and previewed, by default False.
            If false, returns an interactive widget with the list of revisit images visualised.

        Returns
        -------
        IPython.display.Image or ipywidgets.interact
            The visualisation of the downloaded revisits.
        """

        revisits_paths = SPOTDownloader.get_downloaded_revisits_paths(root_path)
        visualiser = Visualiser(revisits_paths)
        return visualiser.visualise_downloaded_SPOT_revisits(
            revisits_paths, save_gif=save_gif
        )

    @staticmethod
    def split_order_into_tiles(order, n_tiles, target_area=None):
        """ Splits an order into n_tiles tiles. This is useful when the order is too big to download in one go.
        The maximum SentinelHub image size is 2500x2500 pixels, and if the order is larger, e.g. 25 km²,
        the panchromatic channel at it's native resolution of 1.5m is too big to fit in the 2500x2500 pixels.

        This method allows you to split the order into smaller tiles, and download them separately.

        Parameters
        ----------
        order : pandas.Series
            The order to split.
        n_tiles : int
            The number of tiles to split the order into.

        Returns
        -------
        pandas.DataFrame
            The tiles.

        Raises
        ------
        ValueError
            If the number of tiles isn't a perfect square.
        """
        n_tiles = np.sqrt(n_tiles)

        if n_tiles % 1 != 0:
            raise ValueError("n_tiles must be a perfect square")
        n_tiles = int(n_tiles)

        order["sqkm"] = (
            order["sqkm"] / n_tiles ** 2 if target_area is None else target_area
        )
        bbox = order["input.bounds.bbox"]
        x, y = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x_step, y_step = x / n_tiles, y / n_tiles
        tiles = pd.DataFrame()
        for x_tile in range(n_tiles):
            for y_tile in range(n_tiles):
                tile_bbox = bbox.copy()
                tile_bbox[0] += x_tile * x_step
                tile_bbox[1] += y_tile * y_step
                tile_bbox[2] = tile_bbox[0] + x_step
                tile_bbox[3] = tile_bbox[1] + y_step
                order_copy = order.copy()
                order_copy["input.bounds.bbox"] = tile_bbox
                order_copy["name"] = f'{order_copy["name"]}-{x_tile+1}-{y_tile+1}'
                tiles = pd.concat([tiles, order_copy])

        return tiles

    def download_micah_asmspotter_order(order, root):
        """ Downloads an order from the Micah/ASMSpotter collection, which is 22.5 km².
        The order is split into 9 tiles to make the area equal to the rest of the dataset (2.5 km²).

        Parameters
        ----------
        order : pandas.Series
            The order to download.
        root : str
            The root path where to save the downloaded images.

        Returns
        -------
        pandas.DataFrame
            The tiles metadata.
        """
        if float(order["sqkm"]) == 2.5:
            SPOTDownloader.download_highres_image_to_file(order, root, order["name"])
            return order
        n_tiles = 9
        tiles = SPOTDownloader.split_order_into_tiles(order, n_tiles, target_area=2.5)
        for _, tile in tiles.iterrows():
            SPOTDownloader.download_highres_image_to_file(tile, root, tile["name"])

        return tiles

    def download_all_micah_asmspotter_orders(
        root="data/downloads/SPOT", collection_id="9c9041d4-e9d1-46bb-89de-6d029784340b", metadata_name=None
    ):
        """ Downloads all the orders from the Micah/ASMSpotter collection, which is 22.5 km², as 2.5 km² tiles.

        Parameters
        ----------
        root : str, optional
            The root path where to save the downloaded images, by default "data/downloads/SPOT".
        collection_id : str, optional
            The collection id, by default "9c9041d4-e9d1-46bb-89de-6d029784340b" (ASMSpotter).
        """
        metadata = pd.DataFrame()
        metadata_name = 'asmspotter_micah_metadata.csv' if metadata_name is None else metadata_name
        orders = SPOTOrder.get_all_orders_for_collection(
            collection_id, status="DONE"
        )
        for _, order in tqdm(orders.iterrows(), total=len(orders), desc='Downloading'):
            tiles = SPOTDownloader.download_micah_asmspotter_order(order, root=root)
            metadata = pd.concat([metadata, tiles])
        metadata.to_csv(str(Path(root, metadata_name)), index=False)

