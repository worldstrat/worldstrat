import os
import math
from geographiclib.geodesic import Geodesic
from dataset_generation.DataLoader import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from folium.plugins import MarkerCluster
import folium
from shapely.geometry.polygon import Polygon


class AOIGenerator:
    """ Generates areas of interest (AOIs) around a set of points of interest (POIs).
    The POIs are assumed to be in a dataframe with columns 'name', 'lat' and 'lon', or 
    a KML, XML, CSV, GeoJSON or JSON file with a 'name' and 'lat', 'lon' columns.
    """

    def __init__(self, pois, aoi_area_sqkm, aoi_name_prefix=None):
        """ Initialises the AOIGenerator class.

        Parameters
        ----------
        pois : str or pd.DataFrame
            The path to a file containing the points of interest, or a dataframe with columns 'name', 'lat' and 'lon'.
        aoi_area_sqkm : float
            The area of each AOI in square kilometers.
        aoi_name_prefix : str, optional
            If a name isn't provided for each AOI, this prefix will be added to the AOI's index and used as the name.
            By default None.
        """
        self.load_pois(pois, aoi_name_prefix)
        self.aoi_area = aoi_area_sqkm
        self.aois = AOIGenerator.generate_aois_from_pois(self.pois, self.aoi_area)
        self.aois["area"] = self.aoi_area

    def load_pois(self, pois, aoi_name_prefix=None):
        """ Loads the points of interest into a Pandas DataFrame.

        Parameters
        ----------
        pois : str or pd.DataFrame
            The path to a file containing the points of interest, or a dataframe with columns 'name', 'lat' and 'lon'.
        aoi_name_prefix : str, optional
            If a name isn't provided for each AOI, this prefix will be added to the AOI's index and used as the name.
            By default None.
        """
        self.load_pois_if_its_a_file(pois, aoi_name_prefix)
        self.load_pois_if_its_a_dataframe(pois, aoi_name_prefix)

    def load_pois_if_its_a_file(self, pois, aoi_name_prefix=None):
        """ Loads the points of interest from a file using the DataLoader class.

        Parameters
        ----------
        pois : str
            The path to a file containing the points of interest.
        aoi_name_prefix : str, optional
            If a name isn't provided for each AOI, this prefix will be added to the AOI's index and used as the name.
        """
        if isinstance(pois, str):
            if os.path.isfile(pois):
                self.pois = DataLoader.load_to_dataframe(
                    pois, aoi_name_prefix=aoi_name_prefix
                )

    def load_pois_if_its_a_dataframe(self, pois, aoi_name_prefix=None):
        """ Loads the points of interest from a Pandas DataFrame.

        Parameters
        ----------
        pois : pd.DataFrame
            The dataframe with columns 'name', 'lat' and 'lon'.
        aoi_name_prefix : str, optional
            If a name isn't provided for each AOI, this prefix will be added to the AOI's index and used as the name.

        Raises
        ------
        Exception
            If the dataframe doesn't have the required columns.
        Exception
            If the dataframe doesn't have a 'name' column.
        """
        if isinstance(pois, pd.DataFrame):
            if "name" not in pois.columns and aoi_name_prefix is not None:
                if not aoi_name_prefix.endswith("-"):
                    aoi_name_prefix += "-"
                pois["name"] = aoi_name_prefix + pois.index.astype(str)
            elif "name" not in pois.columns:
                raise Exception("POIs must have a name column")

            if "lat" in pois.columns and "lon" in pois.columns:
                self.pois = pois
            elif "latitude" in pois.columns and "longitude" in pois.columns:
                self.pois = pois.rename(columns={"latitude": "lat", "longitude": "lon"})
            else:
                raise Exception("POIs must have lat and lon columns")

    def visualise_generated_aois(self):
        """ Visualises the generated AOIs on a map.

        Returns
        -------
        folium.Map
            The map with the generated AOIs.
        """
        map = folium.Map([0, 0], zoom_start=3, tiles="CartoDB dark_matter")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="""Tiles &copy; Esri &mdash; 
            Source: Esri, i-cubed, USDA, USGS, AEX, 
            GeoEye, Getmapping, Aerogrid, 
            IGN, IGP, UPR-EGP, and the GIS User Community""",
        ).add_to(map)

        aoi_style = {"fillColor": "#e74c3c", "color": "#c0392b"}
        marker_cluster = MarkerCluster().add_to(map)

        for _, aoi in self.aois.iterrows():
            folium.Marker((aoi["lat"], aoi["lon"])).add_to(marker_cluster)
            aoi_polygon = Polygon(AOIGenerator.bounds_to_bounding_box(*aoi["bounds"]))
            aoi_geojson = folium.GeoJson(
                aoi_polygon, style_function=lambda x: aoi_style
            )
            aoi_tooltip = folium.Tooltip(
                f"""<strong>AOI:</strong> {aoi['name']} <br> 
                <strong>IPCC:</strong> {aoi['IPCC']} <br> 
                <strong>LCCS:</strong> {aoi['LCCS']} <br> 
                <strong>SMOD:</strong> {aoi['SMOD']}"""
            )
            aoi_tooltip.add_to(aoi_geojson)
            aoi_geojson.add_to(map)
        folium.LatLngPopup().add_to(map)
        return map

    @staticmethod
    def generate_aois_from_pois(pois, aoi_sqkm):
        """Generates a dataframe of AOIs from a list of POIs and the target area in sqkm.

        Parameters
        ----------
        pois : pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        aoi_area : int
            Area of interest in square kilometers.

        Returns
        -------
        pd.DataFrame
            DataFrame of AOIs.
        """
        if "bounds" in pois.columns:
            return pois

        return AOIGenerator.generate_bounds_for_pois_dataframe(pois, aoi_sqkm)

    @staticmethod
    def generate_bounds_for_pois_dataframe(pois_dataframe, side_km):
        """Generate bounding box around each POI in a dataframe with a given side length.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the POIs, defined by a latitude and longitude column.
        side_km : float
            Side length of the bounding box in kilometres.

        Returns
        -------
        pandas.DataFrame
            Dataframe with the same columns as the input dataframe, plus a bounding box column.
        """
        tqdm.pandas(desc="Generating AOIs")
        pois_dataframe["bounds"] = pois_dataframe.progress_apply(
            lambda row: AOIGenerator.generate_bounds_around_poi(
                row["lat"], row["lon"], side_km
            ),
            axis=1,
        )
        return pois_dataframe

    @staticmethod
    def generate_bounds_around_poi(lat, lon, aoi_area, lonlat=True, return_type="list"):
        """Accurate computation of degree of lat needed for a given distance.
        Uses bisection search on geopy's implementation of Karney algorithm, on WGS-84 ellipsoid.

        Parameters
        ----------
        lat : float
            The center point latitude around which the bounding box is made.
        lon : float
            The center point longitude around which the bounding box is made.
        side_km : float
            The size of a bounding side in kilometres.

        Returns
        -------
        dict
            Bounds defined by the maximum and minimum latitude and longitude boundaries.
        """
        if return_type not in ["list", "dict"]:
            raise ValueError("return_type must be either dict or list")

        side_km = math.sqrt(aoi_area)
        geod = Geodesic.WGS84
        side_m = side_km * 1e3

        # Find the point side_m/1000 km
        north = geod.Direct(lat, lon, 0, side_m / 2)
        east = geod.Direct(lat, lon, 90, side_m / 2)
        south = geod.Direct(lat, lon, 180, side_m / 2)
        west = geod.Direct(lat, lon, 270, side_m / 2)

        bounds = [west["lon2"], south["lat2"], east["lon2"], north["lat2"]]
        keys_bounds = ["lon_min", "lat_min", "lon_max", "lat_max"]

        if not lonlat:
            # Swap lat and lon
            bounds[1:4:2], bounds[0:4:2] = bounds[0:4:2], bounds[1:4:2]
            keys_bounds[1:4:2], keys_bounds[0:4:2] = (
                keys_bounds[0:4:2],
                keys_bounds[1:4:2],
            )

        return dict(zip(keys_bounds, bounds)) if return_type == "dict" else bounds

    @staticmethod
    def bounds_to_bounding_box(
        lat_min, lon_min, lat_max, lon_max, closed=False, lonlat=False
    ):
        """Convert bounds to a bounding box (square).

        Parameters
        ----------
        lat_min : float
            Minimum latitude boundary.
        lon_min : float
            Minimum longitude boundary.
        lat_max : float
            Maximum latitude boundary.
        lon_max : float
            Maximum longitude boundary.
        closed : bool, optional
            Closes back to the first corner if True, by default False.
        lonlat : bool, optional
            Points in the bb are in the [longitude, latitude] order if True, in [latitude, longitude] if False. Default False.

        Returns
        -------
        list
            Four corners defining a bounding box, goes clockwise from SouthWest.
        """
        if lonlat:
            bounding_box = [
                [lon_min, lat_min],
                [lon_min, lat_max],
                [lon_max, lat_max],
                [lon_max, lat_min],
            ]
            if closed:
                bounding_box.append([lon_min, lat_min])
        else:
            bounding_box = [
                [lat_min, lon_min],
                [lat_min, lon_max],
                [lat_max, lon_max],
                [lat_max, lon_min],
            ]

            if closed:
                bounding_box.append([lat_min, lon_min])

        return bounding_box

    @staticmethod
    def check_aois(aois):
        """ Checks if the AOIs are valid.
        Valid AOIs are a pandas.DataFrame with name and bounds columns.

        Parameters
        ----------
        aois : pandas.DataFrame
            Dataframe containing the AOIs.

        Raises
        ------
        Exception
            If the AOIs aren't a pandas.DataFrame, as returned by AOIGenerator.
        Exception
            If the AOIs don't have the name column.
        Exception
            If the AOIs don't have the bounds column.
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

    @staticmethod
    def bounds_to_poi(bounds, lon_lat=False):
        """ Convert a bounding box of an AOI back to a POI.
        Returns the center point of the bounding box.

        Parameters
        ----------
        bounds : list
            Four corners defining a bounding box, goes clockwise from SouthWest.
        lon_lat : bool, optional
            Points in the bb are in the [longitude, latitude] order if True, 
            in [latitude, longitude] if False. Default False.

        Returns
        -------
        tuple of float
            The center point of the bounding box.
        """
        lon_min, lat_min, lon_max, lat_max = bounds
        lat_center = (lat_max - lat_min) / 2 + lat_min
        lon_center = (lon_max - lon_min) / 2 + lon_min
        return (lon_center, lat_center) if lon_lat else (lat_center, lon_center)
