import pandas as pd
from dataset_generation.StratificationDatasets import (
    StratificationDatasets,
    LCCS_classes,
    LCCS_TO_IPCC,
    SMOD_classes,
)
from tqdm.auto import tqdm
import pyproj
import math

from dataset_generation.Visualiser import Visualiser


class Stratifier:
    """ Stratifies POIs into SMOD, LCCS and IPCC classes. """

    def __init__(self, points):
        """ Initialises the Stratifier class.

        Parameters
        ----------
        points : pandas.DataFrame
            Dataframe containing the points to be stratified.
        """
        self.points = points
        self.check_points()
        self.visualiser = Visualiser(self.points)
        datasets = StratificationDatasets()
        self.smod_dataset = datasets.smod_dataset
        self.landcover_dataset = datasets.landcover_dataset
        self.mollweide_projection = pyproj.proj.Proj(
            r"+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        self.stratify_points()

    def check_points(self):
        """Checks that the points are a valid dataframe (i.e. contains the correct column: lon, lat).
        If the points are not a dataframe, offers a way to load them into a dataframe or randomly sample points on the planet.

        Raises
        ------
        TypeError
            If the points are not a dataframe.
        ValueError
            If the points do not contain the correct column: lon, lat.
        ValueError
            If the points do not contain the correct column: lon, lat.
        """
        help_string = "\nYou can use the PlanetSampler class to generate a dataframe of random points or the DataLoader class to load a CSV, KML, GeoJSON, XML or KML file."
        columns = list(self.points.columns)
        if not isinstance(self.points, pd.DataFrame):
            raise TypeError(f"Points must be a pandas dataframe.{help_string}")
        if not "longitude" in columns and not "lon" in columns:
            raise ValueError(
                f"Points dataframe must contain a longitude/lon column. No column named 'longitude' or 'lon' found in {columns}.{help_string}"
            )
        if not "latitude" in columns and not "lat" in columns:
            raise ValueError(
                f"Points dataframe must contain a latitude/lat column. No column named 'latitude' or 'lat' found in {columns}.{help_string}"
            )

    def stratify_points(self):
        """Stratifies the points into SMOD, LCCS and IPCC classes.
        """
        self.determine_landcoveres()
        self.determine_smod_classes()

    def determine_landcoveres(self):
        """Determines the ESA CCI landcover class for each point.
        """
        tqdm.pandas(desc="Determining landcover classes")
        self.points["landcover_id"] = self.points.progress_apply(
            lambda row: self.get_landcover(row.lon, row.lat), axis=1
        )
        self.points["LCCS"] = self.points.apply(
            lambda row: self.landcover_to_lccs(row.landcover_id), axis=1
        )
        self.points["IPCC"] = self.points.apply(
            lambda row: self.landcover_to_ipcc(row.landcover_id), axis=1
        )

    def determine_smod_classes(self):
        """Determines the SMOD class for each point.
        """
        tqdm.pandas(desc="Determining SMOD classes")
        self.points["SMOD_id"] = self.points.progress_apply(
            lambda row: self.get_smod(row.lon, row.lat), axis=1
        )
        self.points["SMOD"] = self.points.apply(
            lambda row: SMOD_classes[row.SMOD_id], axis=1
        )

    def get_smod(self, lon, lat):
        """Determines the SMOD class for a point.

        Parameters
        ----------
        lon : float
            Longitude of the point.
        lat : float
            Latitude of the point.

        Returns
        -------
        int
            SMOD class ID.
        """
        x, y = Stratifier.lon_lat_to_mollweide(lon, lat)
        bottom_left = self.smod_dataset.index(x, y, op=math.ceil)
        upper_right = self.smod_dataset.index(x, y, op=math.floor)
        rows = (bottom_left[0], upper_right[0] + 2)
        cols = (bottom_left[1], upper_right[1] + 2)
        classes = self.smod_dataset.read(1, window=(rows, cols))
        try:
            return int(classes[0][0])
        except:
            return 0

    def get_landcover(self, lon, lat):
        """Determines the ESA CCI landcover class for a point.

        Parameters
        ----------
        lon : float
            Longitude of the point.
        lat : float
            Latitude of the point.

        Returns
        -------
        int
            ESA CCI landcover class ID.
        """
        x, y = self.landcover_dataset.index(lon, lat, op=math.ceil)
        rows = (x, x + 1)
        cols = (y, y + 1)

        ESA_CCI_LC_BAND = 1
        classes = self.landcover_dataset.read(ESA_CCI_LC_BAND, window=(rows, cols))
        try:
            classes = int(classes[0][0])
        except:
            print(f"No class found for {lon}, {lat}")
            classes = None
        return classes

    def visualise_SMOD_distribution(self):
        """ Visualises the SMOD distribution within the sampled points.

        Returns
        -------
        plotnine.ggplot
            Plot of the SMOD distribution.
        """
        self.visualiser.update_data(self.points)
        return self.visualiser.visualise_distribution(
            "SMOD",
            title="Distribution of GHSL SMOD classes within the sampled points.",
            x_label="GHSL SMOD class",
        )

    def visualise_LCCS_distribution(self):
        """ Visualises the LCCS distribution within the sampled points.

        Returns
        -------
        plotnine.ggplot
            Plot of the LCCS distribution.
        """
        self.visualiser.update_data(self.points)
        return self.visualiser.visualise_distribution(
            "LCCS",
            "Distribution of LCCS classes within the sampled points.",
            x_label="ESA CCI LCCS class",
        )

    def visualise_IPCC_distribution(self):
        """ Visualises the IPCC distribution within the sampled points.

        Returns
        -------
        plotnine.ggplot
            Plot of the IPCC distribution.
        """        
        self.visualiser.update_data(self.points)
        return self.visualiser.visualise_distribution(
            "IPCC",
            "Distribution of IPCC classes within the sampled points.",
            x_label="IPCC class",
        )

    def visualise_SMOD_conditionally_on_IPCC(self):
        """ Visualises the SMOD distribution conditioned on the IPCC class.

        Returns
        -------
        plotnine.ggplot
            Plot of the SMOD distribution.
        """        
        self.visualiser.update_data(self.points)
        return self.visualiser.visualise_conditional_distribution(
            "IPCC", "SMOD", "SMOD class conditioned on IPCC class"
        )

    def visualise_IPCC_conditionally_on_SMOD(self):
        """ Visualises the IPCC distribution conditioned on the SMOD class.

        Returns
        -------
        plotnine.ggplot
            Plot of the IPCC distribution.
        """        
        self.visualiser.update_data(self.points)
        return self.visualiser.visualise_conditional_distribution(
            "SMOD", "IPCC", "IPCC class conditioned on SMOD class"
        )

    def to_csv(self, path):
        """Writes the points to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        """
        self.points.to_csv(path)

    @staticmethod
    def from_csv(path):
        """Loads points from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        Stratifier
            Stratifier instantiated with the points from the CSV file.
        """
        points = pd.read_csv(path, index_col=0)
        return Stratifier(points)

    @staticmethod
    def lon_lat_to_mollweide(lon, lat):
        """Static method that converts longitude and latitude to Mollweide coordinates.

        Parameters
        ----------
        lon : float
            Longitude of the point.
        lat : float
            Latitude of the point.

        Returns
        -------
        float
            Mollweide x, y coordinates.
        """
        mollweide_projection = pyproj.proj.Proj(
            r"+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        return mollweide_projection(lon, lat)

    @staticmethod
    def get_smod_for_point(lon, lat):
        """Loads the SMOD dataset and determines the SMOD class for a single point.
        Meant to be used to determine the SMOD class for a single point without instantiating the stratifier and for testing purposes.

        Parameters
        ----------
        lon : float
            Longitude of the point.
        lat : float
            Latitude of the point.

        Returns
        -------
        int
            SMOD class ID.
        """
        smod_dataset = StratificationDatasets.load_smod_dataset()
        x, y = Stratifier.lon_lat_to_mollweide(lon, lat)
        bottom_left = smod_dataset.index(x, y, op=math.ceil)
        upper_right = smod_dataset.index(x, y, op=math.floor)
        rows = (bottom_left[0], upper_right[0] + 2)
        cols = (bottom_left[1], upper_right[1] + 2)
        classes = smod_dataset.read(1, window=(rows, cols))
        try:
            return int(classes[0][0])
        except:
            return None

    @staticmethod
    def get_cci_lc_for_point(lon, lat):
        """Loads the ESA CCI landcover dataset and determines the ESA CCI landcover class for a single point.
        Meant to be used to determine the ESA CCI landcover class for a single point without instantiating the stratifier and for testing purposes.

        Parameters
        ----------
        lon : float
            Longitude of the point.
        lat : float
            Latitude of the point.

        Returns
        -------
        int
            ESA CCI landcover class ID.
        """
        classes = None
        landcover_dataset = StratificationDatasets.load_landcover_dataset()
        x, y = landcover_dataset.index(lon, lat, op=math.ceil)
        rows = (x, x + 1)
        cols = (y, y + 1)

        ESA_CCI_LC_BAND = 1
        classes = landcover_dataset.read(ESA_CCI_LC_BAND, window=(rows, cols))
        try:
            classes = int(classes[0][0])
        except:
            print(f"No class found for {lon}, {lat}")
            classes = None
        return classes

    @staticmethod
    def landcover_to_ipcc(landcover):
        """Translates the ESA CCI landcover class to the appropriate IPCC class.

        Parameters
        ----------
        landcover : int
            ESA CCI landcover class ID.

        Returns
        -------
        str
            IPCC class description.
        """
        return LCCS_TO_IPCC[landcover]

    @staticmethod
    def landcover_to_lccs(landcover):
        """Translates the ESA CCI landcover class id to the appropriate LCCS class name.

        Parameters
        ----------
        landcover : int
            ESA CCI landcover class ID.

        Returns
        -------
        str
            LCCS class name.
        """
        return LCCS_classes[landcover]
