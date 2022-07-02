from dataset_generation.SentinelHub import SentinelHub
import pandas as pd
from datetime import date
from dateutil import parser
from sentinelhub import CRS, serialize_time
import json
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import requests
import dateutil

import concurrent
from concurrent.futures import as_completed
from tqdm.auto import tqdm
from dataset_generation.Visualiser import Visualiser


class SPOTCatalogue:
    """ Searches the Sentinel Hub catalogue for available SPOT revisits for AOIs. """

    def __init__(
        self,
        aois,
        start_date=datetime(2017, 1, 1),
        end_date=datetime.now(),
        aoi_area=None,
        crs=CRS.WGS84,
        max_cloud_percentage=100,
        max_incidence_angle=90,
        multithread=True,
    ):
        """ Initialises the SPOT catalogue searcher.

        Parameters
        ----------
        aois : pandas.DataFrame
            A dataframe containing the bounding boxes of the areas of interest.
        start_date : datetime, optional
            The start date of the search, by default datetime(2017, 1, 1).
        end_date : datetime, optional
            The end date of the search, by default datetime.now().
        aoi_area : float, optional
            The area of the AOI, by default None.
        crs : sentinelhub.CRS, optional
            The coordinate reference system of the bounding boxes, by default CRS.WGS84.
        max_cloud_percentage : int, optional
            The maximum cloud cover percentage, by default 100.
        max_incidence_angle : int, optional
            The maximum incidence angle, by default 90.
        multithread : bool, optional
            Whether to multithread API calls, by default True.
        """
        self.check_aois(aois)
        self.aois = aois
        self.start_date = start_date
        self.end_date = end_date
        self.set_aoi_area(aoi_area)
        self.crs = crs
        self.max_cloud_percentage = max_cloud_percentage
        self.max_incidence_angle = max_incidence_angle
        self.multithread = multithread
        self.download_spot_catalogue()
        self.visualiser = Visualiser(None)

    def download_spot_catalogue(self):
        """ Downloads the entire SPOT catalogue for the specified AOIs and date range. """
        print(f"Searching SPOT catalogue from {self.start_date} to {self.end_date}")
        print(f"This can take a while if there are a lot of revisits available.")
        self.search_spot_catalogue_for_aois()
        self.spot_catalogue["name"] = self.spot_catalogue.index
        self.spot_catalogue.drop_duplicates(
            ["name", "datetime"], keep="first", inplace=True
        )
        self.spot_catalogue.drop("name", axis=1, inplace=True)

    def visualise_catalog_coverage(self):
        """ Visualises the coverage of the SPOT catalogue for each AOI on a world map.

        Returns
        -------
        folium.Map
            A world map visualising the coverage of the SPOT catalogue for each AOI.
        """
        aois_and_catalogue_merged = self.aois.merge(
            self.spot_catalogue, left_index=True, right_index=True
        )
        self.visualiser.update_data(aois_and_catalogue_merged)
        return self.visualiser.visualise_aoi_and_revisit_coverage()

    def search_spot_catalogue_for_aois(self):
        """Searches the Sentinel Hub catalogue for available SPOT revisits for each AOI (row) in the dataframe.
        Wrapper function that uses the default Sentinel Hub catalogue search parameters and allows easy multithreading switching.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (row).
        """

        if self.multithread:
            return self.search_spot_catalogue_for_aois_multithreaded()
        else:
            raise NotImplementedError("Single threaded version not implemented yet.")

    def search_spot_catalogue_for_aois_multithreaded(self):
        """Searches the Sentinel Hub catalogue for available SPOT revisits for each AOI (row) in the dataframe using multiple threads.

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
                    self.search_spot_catalogue_for_aoi, aoi["bounds"], index
                )
                for index, aoi in self.aois.iterrows()
            ]
            for task in tqdm(
                as_completed(res), total=len(res), desc="Searching SPOT catalogue"
            ):
                pass
        catalogue = pd.concat([revisits.result() for revisits in res])
        self.spot_catalogue = catalogue

    def search_spot_catalogue_for_aoi(self, bounds, aoi_index):
        """Searches the Sentinel Hub catalogue for aoi_index SPOT data for the given AOI (bounds).

        Parameters
        ----------
        bounds : list
            The AOI to search for, in the format [lon_min, lat_min, lon_max, lat_max].
        end_date : datetime.datetime
            The end date of the search, by default now.
        max_cloud_pc : int, optional
            The maximum cloud percentage, by default 100.
        max_incidence : int, optional
            The maximum incidence angle, by default 90.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the available catalogue data for each AOI (bounds),
            an empty dataframe if no data is available.
        """

        payload = self.generate_spot_payload(bounds)
        results = self.query_sentinel_hub_for_spot_revisits(payload)
        try:
            revisits = self.format_spot_catalogue_results(results)
            revisits["aoi_index"] = aoi_index
            revisits.set_index("aoi_index", inplace=True)
            return revisits
        except:
            print(
                f"WARNING: No SPOT revisits found for {bounds} between {self.start_date} and {self.end_date} with max_cloud_pc <= {self.max_cloud_percentage}."
            )
            return pd.DataFrame()

    def generate_spot_payload(self, bounds):
        """Generates the payload defining the SPOT catalogue query for the Sentinel Hub API.

        Parameters
        ----------
        bounds : list
            The AOI to search for, in the format [lon_min, lat_min, lon_max, lat_max].
        start_date : datetime.datetime
            The start date of the search.
        end_date : datetime.datetime
            The end date of the search.
        max_cloud_pc : int
            The maximum cloud percentage.
        max_incidence : int
            The maximum incidence angle.

        Returns
        -------
        str
            The payload to send to the API, containing the AOI bounds, start and end date.
        """
        payload = {
            "provider": "AIRBUS",
            "bounds": {"bbox": bounds},
            "data": [
                {
                    "constellation": "SPOT",
                    "dataFilter": {
                        "maxCloudCoverage": self.max_cloud_percentage,
                        "maxIncidenceAngle": self.max_incidence_angle,
                        "timeRange": {
                            # "2020-05-13T00:00:00Z",
                            "from": serialize_time(self.start_date, use_tz=True),
                            # "2021-06-13T23:59:59Z"
                            "to": serialize_time(self.end_date, use_tz=True),
                        },
                    },
                }
            ],
        }
        return json.dumps(payload)

    def query_sentinel_hub_for_spot_revisits(self, payload):
        """Queries the Sentinel Hub API for SPOT data.

        Parameters
        ----------
        payload : dict
            The payload to send to the API, containing the AOI bounds, start and end date.
        sentinel_hub_session : sentinelhub.SentinelHubSession
            The Sentinel Hub session to use for the query.

        Returns
        -------
        list
            A list of dictionaries containing the available SPOT revisits.
        """
        # Reference: https://docs.sentinel-hub.com/api/latest/reference/#tag/dataimport_search
        SH_SEARCH_URL = "https://services.sentinel-hub.com/api/v1/dataimport/search"
        url = SH_SEARCH_URL
        features = []
        done = False
        while not done:
            response = self.post_request_to_sentinel_hub(url, payload)
            response_features, url, done = self.clear_unused_fields_in_response(
                response
            )
            features.extend(response_features)
        return features

    def post_request_to_sentinel_hub(self, url, payload):
        """Sends a POST request to the Sentinel Hub API. 
        Uses an HTTP adapter that retries the request if it fails due to a server error, using a backoff factor of 1.
        If the request fails more than 5 times, the request is aborted and the exception is raised.

        Parameters
        ----------
        url : str
            The URL to send the POST request to.
        payload : dict
            The payload to send to the API.
        sentinel_hub_session : sentinelhub.SentinelHubSession
            The Sentinel Hub session to use for the query.

        Returns
        -------
        dict
            The response from the API.
        Raises
        ------
        e
            The exception raised by the request, if the request fails more than 5 times.
        """
        headers = {"Content-Type": "application/json"}

        # Refresh token to be safe
        headers.update(SentinelHub.generate_sentinel_hub_session().session_headers)
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        # Any request that has an URL with this prefix (https://) will use this adapter
        s.mount("https://", HTTPAdapter(max_retries=retries))
        try:
            r = s.post(url, data=payload, headers=headers)
        except Exception as e:
            print(f"ERROR: {e}")
            raise e

        r.raise_for_status()

        rj = json.loads(r.content)
        return rj

    def clear_unused_fields_in_response(self, response):
        """Clears the unused fields in the response from the Sentinel Hub API.

        Parameters
        ----------
        response : dict
            The response from the API.

        Returns
        -------
        dict, str, bool
            The response from the API, with unused fields removed, the link to the next page of results, if any, and a boolean indicating whether there are more results.
        """
        # Clear unused fields
        done = False
        for f in response["features"]:
            for image in ["quicklook", "thumbnail"]:
                f[image + ".href"] = f["_links"][image]["href"]
                f[image + ".link"] = '<a href="{}">{}</a>'.format(
                    f[image + ".href"], image
                )
            f.pop("_links")

        # Have we reached the last page?
        if ("links" not in response) or ("next" not in response["links"]):
            done = True
            return response["features"], None, done
        url = response["links"]["next"]
        return response["features"], url, done

    def format_spot_catalogue_results(self, results):
        """Formats the SPOT catalogue results into a pandas dataframe, sets the source to SPOT, parses the datetime and coordinates and renames several columns.
        Drops a lot of processing-related fields.
        Parameters
        ----------
        results : dict
            A dictionary containing the available catalogue data response from the API.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the parsed and reformatted catalogue data.
        """
        # Airbus format explained here: http://www.geoapi-airbusds.com/api-catalog/oneatlas-data/index.html#tag/Search/paths/~1api~1v1~1items~1{catalogItemId}/get
        results = pd.json_normalize(results)
        results["source"] = "SPOT"

        results["coordinates"] = results["geometry.coordinates"].apply(lambda x: x[0])
        results["datetime"] = results["properties.acquisitionDate"].apply(
            lambda x: dateutil.parser.isoparse(x)
        )
        results.rename(
            columns={
                "properties.id": "product_id",
                "properties.cloudCover": "cloud_cover",
                "properties.azimuthAngle": "view_azimuth",
                "properties.incidenceAngle": "incidence",
                "properties.illuminationAzimuthAngle": "sun_azimuth",
                "properties.illuminationElevationAngle": "sun_elevation",
            },
            inplace=True,
        )

        # We drop a lot of info, in particular processing type.
        # Gergga says we can safely ignore SENSOR vs ALBUM value in processing type
        results.drop(
            columns=[
                "properties.acquisitionDate",
                "geometry.coordinates",
                "type",
                "geometry.type",
                "properties.acquisitionIdentifier",
                "properties.acquisitionStation",
                "properties.activityId",
                "properties.archivingCenter",
                "properties.constellation",
                "properties.correlationId",
                "properties.expirationDate",
                "properties.format",
                "properties.incidenceAngleAcrossTrack",
                "properties.incidenceAngleAlongTrack",
                "properties.lastUpdateDate",
                "properties.organisationName",
                "properties.parentIdentifier",
                "properties.platform",
                "properties.processingCenter",
                "properties.processingDate",
                "properties.processingLevel",
                "properties.processorName",
                "properties.productCategory",
                "properties.productType",
                "properties.productionStatus",
                "properties.publicationDate",
                "properties.qualified",
                "properties.resolution",
                "properties.sensorType",
                "properties.snowCover",
                "properties.sourceIdentifier",
                "properties.spectralRange",
                "properties.workspaceId",
                "properties.workspaceName",
                "properties.workspaceTitle",
                "properties.comment",
                "properties.commercialReference",
                "properties.customerReference",
                "properties.acquisitionAnonymousIdentifier",
                "properties.archiveCorrelationId",
                "properties.title",
                "properties.dataUri",
            ],
            inplace=True,
            errors="ignore",
        )

        return results.reindex(
            columns=[
                "product_id",
                "source",
                "datetime",
                "coordinates",
                "cloud_cover",
                "incidence",
                "resolution",
                "view_azimuth",
                "sun_azimuth",
                "sun_elevation",
                "quicklook.href",
                "quicklook.link",
                "thumbnail.href",
                "thumbnail.link",
            ]
        )

    def set_target_date(self, target_date, overwrite=False):
        """ Sets the target date for the Sentinel Hub API query.

        Parameters
        ----------
        target_date : datetime.date
            The target date for the query.
        overwrite : bool, optional
            Whether to overwrite the current target date. The default is False.
        """
        if self.target_date is not None and overwrite is False:
            print(
                f"Target date is already set to {self.target_date}. Use overwrite=True to overwrite."
            )
            return
        if str(target_date) in self.aois.columns:
            self.aois["target_date"] = pd.to_datetime(self.aois[target_date])
            return
        if type(target_date) is type(date.today()):
            target_date = pd.to_datetime(target_date)
        else:
            try:
                target_date = pd.to_datetime(parser.parse(target_date)).tz_localize(
                    "UTC"
                )
            except:
                print(
                    f"{target_date} is not a valid date. Please use a valid date, date string or a column name."
                )
        self.target_date = target_date.tz_convert("UTC")
        self.aois["target_date"] = target_date
        print(f"Target date set to: {target_date}.")

    def set_aoi_area(self, aoi_area):
        """ Sets the target AOI area for all AOIs.

        Parameters
        ----------
        aoi_area : float
            The target AOI area for all AOIs.
        """
        if aoi_area is not None:
            self.aoi_area = aoi_area
            print(f"AOI area set to {self.aoi_area}.")
        else:
            self.aoi_area = self.aois.head(1)["area"].values[0]
            print(
                f"AOI area wasn't explicitly passed, sampling one AOI and taking its area ({self.aoi_area}) and assuming all AOIs have the same area."
            )

    def check_aois(self, aois):
        """ Checks if the AOIs passed to the class are valid.
        Valid AOIs are a pandas.DataFrame with a name and bounds column.

        The AOIGenerator class can be used to generate AOIs, 
        and the DataLoader class can be used to load existing  AOIs from a file.

        Parameters
        ----------
        aois : pandas.DataFrame
            A pandas.DataFrame containing the AOIs to be queried.

        Raises
        ------
        Exception
            If the AOIs passed to the class are not a pandas.DataFrame.
        Exception
            If the AOIs passed to the class do not have a name column.
        Exception
            If the AOIs passed to the class do not have a bounds column.
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

