import pandas as pd
from datetime import datetime
from sentinelhub import BBox, CRS, DataCollection
import concurrent
from concurrent.futures import as_completed
from dataset_generation.SentinelHub import SentinelHub
from dataset_generation.AOIGenerator import AOIGenerator
from tqdm.auto import tqdm
import dateutil
from shapely.geometry.polygon import Polygon
from pyproj import Geod
from dataset_generation.Visualiser import Visualiser


class SentinelCatalogue:
    """ Searches the Sentinel Hub catalogue for available Sentinel-2 revisits for each AOI within the given time range,
    and allows for filtering by a maximum cloud cover percentage.
    """

    def __init__(
        self,
        aois,
        start_date=datetime(2017, 1, 1),
        end_date=datetime.now(),
        crs=CRS.WGS84,
        max_cloud_percentage=100,
        multithread=True,
    ):
        """ Initialises the SentinelCatalogue class.

        Parameters
        ----------
        aois : pandas.DataFrame
            A dataframe containing the bounding boxes of the areas of interest.
        start_date : datetime.datetime
            The start date of the search, by default 2017-01-01.
        end_date : datetime.datetime
            The end date of the search, by default the current date.
        crs : sentinelhub.CRS
            The coordinate reference system of the AOIs, by default WGS84.
        max_cloud_percentage : int, optional
            The maximum percentage of cloud cover allowed in the Sentinel-2 catalogue data, by default 100.
        multithread : bool, optional
            Whether to use multithreading to search the Sentinel Hub catalogue, by default True.
        """
        self.check_aois(aois)
        self.aois = aois
        self.data_collection = DataCollection.SENTINEL2_L2A
        self.start_date = start_date
        self.end_date = end_date
        self.crs = crs
        self.max_cloud_percentage = max_cloud_percentage
        self.multithread = multithread
        self.download_sentinel2_catalogue()
        self.visualiser = Visualiser(None)

    def check_aois(self, aois):
        """ Checks that the AOIs are valid.
        Valid AOIs should be a Pandas DataFrame with:
        - A column named "bounds" containing the bounding boxes of the areas of interest.
        - A column named "name" containing the name of the area of interest.

        Parameters
        ----------
        aois : pandas.DataFrame
            A dataframe containing the bounding boxes of the areas of interest.

        Raises
        ------
        Exception
            If the AOIs are not a Pandas DataFrame, as returned by the AOIGenerator.
        Exception
            If the AOIs do not contain a column named "bounds".
        Exception
            If the AOIs do not contain a column named "name".
        """
        if not isinstance(aois, pd.DataFrame):
            raise Exception("AOIs must be a dataframe, as returned by AOIGenerator.")
        if "name" not in aois.columns:
            raise Exception(
                "AOIs must have a name column. You can use the AOIGenerator class to generate AOIs from POIs."
            )
        if "bounds" not in aois.columns:
            raise Exception(
                "AOIs must have a bounds column. You can use the AOIGenerator class to generate AOIs from POIs."
            )

    def download_sentinel2_catalogue(self):
        """ Searches the Sentinel Hub catalogue for available Sentinel-2 revisits for each AOI (row) in the dataframe 
        and downloads the catalogue data for each AOI (row). """
        print(
            f"Searching Sentinel-2 catalogue from {self.start_date} to {self.end_date}"
        )
        print(f"This can take a while if there are a lot of revisits available.")
        self.search_sentinel2_catalogue_for_aois()
        self.sentinel2_catalogue["name"] = self.sentinel2_catalogue.index
        self.sentinel2_catalogue.drop_duplicates(
            ["name", "datetime"], keep="first", inplace=True
        )
        self.sentinel2_catalogue.drop("name", axis=1, inplace=True)

    def visualise_catalog_coverage(self):
        """ Visualises the overlap of the Sentinel Hub catalogue coverage for each AOI and the AOI itself on a map.

        Returns
        -------
        matplotlib.pyplot.figure
            A figure containing the map.
        """
        aois_and_catalogue_merged = self.aois.merge(
            self.sentinel2_catalogue, left_index=True, right_index=True
        )
        self.visualiser.update_data(aois_and_catalogue_merged)
        return self.visualiser.visualise_aoi_and_revisit_coverage()

    def search_sentinel2_catalogue_for_aois(self):
        """Searches the Sentinel Hub catalogue for available Sentinel-2 revisits for each AOI (row) in the dataframe.
        Wrapper function that uses the default Sentinel Hub catalogue search parameters and allows easy multithreading switching.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (row).
        """

        if self.multithread:
            return self.search_sentinel2_catalogue_for_aois_multithreaded()
        else:
            return self.search_sentinel2_catalogue_for_aois_singlethreaded()

    def search_sentinel2_catalogue_for_aois_multithreaded(self):
        """Searches the Sentinel Hub catalogue for available Sentinel-2 revisits for each AOI (row) in the dataframe using multiple threads.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            A dataframe containing the bounding boxes of the areas of interest.
        data_collection : string, optional
            The Sentinel Hub data collection to search for, by default DataCollection.SENTINEL2_L2A.
        start_date : datetime, optional
            The start date of the search, by default datetime(2017, 1, 1).
        end_date : datetime, optional
            The end date of the search, by default datetime.now().
        crs : sentinelhub.CRS, optional
            The coordinate reference system of the bounding boxes, by default CRS.WGS84.
        max_cloud_pc : int, optional
            The maximum cloud cover percentage, by default 100.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (row).
        """
        with concurrent.futures.ThreadPoolExecutor() as executor_thread:
            res = [
                executor_thread.submit(
                    self.search_sentinel2_catalogue_for_aoi, aoi["bounds"], index
                )
                for index, aoi in self.aois.iterrows()
            ]
            for task in tqdm(
                as_completed(res), total=len(res), desc="Searching Sentinel2 catalogue"
            ):
                pass
        catalogue = pd.concat([revisits.result() for revisits in res])
        self.sentinel2_catalogue = catalogue

    def search_sentinel2_catalogue_for_aois_singlethreaded(self):
        """Searches the Sentinel Hub catalogue for available Sentinel-2 revisits for each AOI (row) in the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            A dataframe containing the bounding boxes of the areas of interest.
        data_collection : string, optional
            The Sentinel Hub data collection to search for, by default DataCollection.SENTINEL2_L2A.
        start_date : datetime, optional
            The start date of the search, by default datetime(2017, 1, 1).
        end_date : datetime, optional
            The end date of the search, by default datetime.now().
        crs : sentinelhub.CRS, optional
            The coordinate reference system of the bounding boxes, by default CRS.WGS84.
        max_cloud_pc : int, optional
            The maximum cloud cover percentage, by default 100.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (row).
        """
        catalogue = pd.DataFrame()
        for index, aoi in tqdm(
            self.aois.iterrows(),
            total=len(self.aois),
            desc="Searching Sentinel2 catalogue",
        ):
            aoi_revisits = self.search_sentinel2_catalogue_for_aoi(aoi["bounds"], index)
            aoi_revisits["aoi_index"] = index
            aoi_revisits.set_index("aoi_index", inplace=True)
            catalogue = pd.concat([catalogue, aoi_revisits])
        self.sentinel2_catalogue = catalogue

    def search_sentinel2_catalogue_for_aoi(self, bounds, index):
        """Searches the Sentinel Hub catalogue for available Sentinel 2 data for the given AOI (bounds).

        Parameters
        ----------
        bounds : list
            The AOI to search for, in the format [lon_min, lat_min, lon_max, lat_max].
        start_date : datetime.datetime
            The start date of the search, by default 2017-01-01.
        end_date : datetime.datetime
            The end date of the search, by default now.
        crs : sentinelhub.CRS
            The coordinate reference system of the AOI, by default WGS84.
        max_cloud_pc : int, optional
            The maximum cloud percentage, by default 100.
        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (bounds),
            an empty dataframe if no data is available.
        """
        bounding_box = BBox(bounds, self.crs)
        search_from_to_dates = (self.start_date, self.end_date)
        search_iterator = self.generate_search_iterator(
            bounding_box, search_from_to_dates
        )
        results = list(search_iterator)
        revisits = self.format_sentinel_catalogue_results(results)
        revisits["aoi_index"] = index
        revisits.set_index("aoi_index", inplace=True)
        return revisits

    def generate_search_iterator(self, bounding_box, time_interval):
        """Generates a SentinelHubCatalog iterator for the Sentinel Hub catalogue search.

        Parameters
        ----------
        data_collection : sentinelhub.DataCollection
            The SentinelHub DataCollection whose catalogue will be searched.
        bounding_box : sentinelhub.BBox
            The AOI to search for, as returned by sentinelhub.BBox when given the format [lon_min, lat_min, lon_max, lat_max].
        time_interval : tuple
            The start and end date of the search.
        max_cloud_pc : int
            The maximum cloud percentage.

        Returns
        -------
        SentinelHubCatalog iterator
            An iterator of the available catalogue data.
        """
        catalog = SentinelHub.get_sentinel_hub_catalog()

        return catalog.search(
            self.data_collection,
            bbox=bounding_box,
            time=time_interval,
            # distinct="date"
            filter=f"eo:cloud_cover<{self.max_cloud_percentage}",
            fields={
                "include": [
                    "id",
                    "geometry",
                    "properties.datetime",
                    "properties.eo:cloud_cover",
                ],
                "exclude": [],
            },
        )

    def calculate_revisit_intersection_area_for_aois_and_catalogue_revisits(self):
        """Calculates the intersection area between the AOIs and the catalogue revisits a dataframe of revisits and AOIs.

        Parameters
        ----------
        aois : pandas.DataFrame
            DataFrame contaning the AOIs.
        catalogue : pandas.DataFrame
            DataFrame containing the catalogue data (revisits) for the AOIs.

        Returns
        -------
        pandas.DataFrame
            The catalogue data with the intersection area added.
        """
        tqdm.pandas(
            desc="Calculating intersection area for AOIs and catalogue revisits"
        )
        self.sentinel2_catalogue["area"] = self.sentinel2_catalogue.progress_apply(
            lambda row: self.calculate_revisit_intersection_area_for_row(row), axis=1
        )

    def calculate_revisit_intersection_area_for_row(self, row):
        """Calculates the intersection area between the AOI and the catalogue revisit for a single revisit row.

        Parameters
        ----------
        aois : pandas.DataFrame
            DataFrame contaning the AOIs.
        row : pandas.Series
            A single row of the catalogue data (revisit).

        Returns
        -------
        float
            The area of the intersection between the catalogue revisit and its AOI.
        """
        sentinel2_revisit_polygon = Polygon(row["coordinates"])
        aoi_bounds = self.aois.loc[row.name]["bounds"]
        aoi_bounding_box = AOIGenerator.bounds_to_bounding_box(*aoi_bounds, closed=True)
        aoi_polygon = Polygon(aoi_bounding_box)
        intersection = sentinel2_revisit_polygon.intersection(aoi_polygon)
        geod = Geod(ellps="WGS84")
        area = abs(geod.geometry_area_perimeter(intersection)[0]) / 10 ** 6
        return round(area, 2)

    def format_sentinel_catalogue_results(self, results):
        """Formats the Sentinel Hub catalogue results into a pandas dataframe.

        Parameters
        ----------
        results : list
            A list of dictionaries containing the available catalogue data.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data.
        """
        results = pd.json_normalize(results)
        results["source"] = "Sentinel2"
        results["cloud_cover"] = results["properties.eo:cloud_cover"]
        results["coordinates"] = results["geometry.coordinates"].apply(
            lambda x: x[0][0]
        )
        results["datetime"] = results["properties.datetime"].apply(
            lambda x: dateutil.parser.isoparse(x)
        )
        results.drop(
            columns=[
                "properties.datetime",
                "properties.eo:cloud_cover",
                "geometry.type",
                "geometry.crs.type",
                "geometry.crs.properties.name",
                "geometry.coordinates",
            ],
            inplace=True,
        )

        results.rename(columns={"id": "product_id"}, inplace=True)

        return results.reindex(
            columns=["product_id", "source", "datetime", "coordinates", "cloud_cover"]
        )

    def calculate_delta_for_sentinel2_revisits(self):
        """Calculates the difference in days between the Sentinel-2 revisits and their AOIs' target dates.

        Parameters
        ----------
        catalogue : pandas.DataFrame
            DataFrame containing the catalogue data (S2 revisits) for the AOIs.
        aois : pandas.DataFrame
            DataFrame contaning the AOIs.

        Returns
        -------
        pandas.DataFrame
            The catalogue data with the delta (difference in days, +/-) added.
        """
        self.aois["target_date"] = self.aois["target_date"].apply(
            lambda x: pd.to_datetime(x).date
        )
        self.sentinel2_catalogue["datetime"] = self.sentinel2_catalogue[
            "datetime"
        ].apply(lambda x: pd.to_datetime(x).date)
        tqdm.pandas(
            desc="Calculating delta between target date and Sentinel2 catalogue revisit"
        )
        self.sentinel2_catalogue["delta"] = self.sentinel2_catalogue.progress_apply(
            lambda row: (row["datetime"] - self.aois.loc[row.name]["target_date"]).days,
            axis=1,
        )

    def filter_by_intersection_area(self):
        """ Filters the catalogue data by the intersection area between the AOI and the catalogue revisit.
        The intersection has to be within 0.01% of the target AOI area in order to be included.
        """
        self.sentinel2_catalogue = self.sentinel2_catalogue.loc[
            self.sentinel2_catalogue["area"] > self.aoi_area * 0.999
        ]

