import pandas as pd
import geopandas as gpd
import os


class DataLoader:
    def __init__(self, path, name_prefix=None, check_name=True, index_col=None):
        """ Initializes the DataLoader class.

        Parameters
        ----------
        path : string
            Path to the file containing the POIs.
        name_prefix : string
            Prefix to add to the POI names.
        check_name : bool, optional
            Whether to check if the dataframe contains a name column.
        """
        self.path = path
        self.name_prefix = name_prefix
        self.check_name = check_name
        self.index_col = index_col
        self.dataframe = self.load_file()

    def load_csv_to_dataframe(self, path):
        """Loads a CSV file containing a list of longitude and latitude pairs into a pandas dataframe.

        Parameters
        ----------
        path : string
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        """
        return pd.read_csv(path) if self.index_col is None else pd.read_csv(path, index_col=self.index_col)

    def load_kml_to_dataframe(self, path):
        """Loads a KML file containing a list of longitude and latitude pairs into a pandas dataframe.

        Parameters
        ----------
        path : string
            Path to the KML file.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        """
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        return gpd.read_file(path, driver="KML")

    def check_file_exists(self, path):
        """Checks if a file exists at the given path.

        Parameters
        ----------
        path : string
            Path to the file.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        return os.path.exists(path)

    def check_if_dataframe_contains_lon_lat(self, dataframe):
        """Checks if a dataframe contains the longitude and latitude columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to check.

        Returns
        -------
        bool
            True if the dataframe contains the longitude and latitude columns, False otherwise.
        """

        long_name = "longitude" in dataframe.columns and "latitude" in dataframe.columns
        short_name = "lon" in dataframe.columns and "lat" in dataframe.columns
        return long_name or short_name

    def check_if_dataframe_contains_name(self, dataframe):
        """Checks if a dataframe contains a name column.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to check.

        Returns
        -------
        bool
            True if the dataframe contains the name column, False otherwise.
        """
        self.name_key = None
        if "name" in dataframe.columns:
            self.name_key = "name"
        if "id" in dataframe.columns:
            self.name_key = "id"
        if "poi_name" in dataframe.columns:
            self.name_key = "poi_name"
        if "poi_id" in dataframe.columns:
            self.name_key = "poi_id"
        return not self.name_key is None

    def load_geojson_to_dataframe(self, path):
        """Loads a GeoJSON file containing a list of longitude and latitude pairs into a pandas dataframe.

        Parameters
        ----------
        path : string
            Path to the geojson file.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        """

        dataframe = gpd.read_file(path)
        return dataframe

    def load_json_to_dataframe(self, path):
        """Loads a JSON file containing a list of longitude and latitude pairs into a pandas dataframe.

        Parameters
        ----------
        path : string
            Path to the JSON file.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        """
        return pd.read_json(path)

    def load_xml_to_dataframe(self, path):
        """Loads an XML file containing a list of longitude and latitude pairs into a pandas dataframe.

        Parameters
        ----------
        path : string
            Path to the XML file.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.
        """
        return pd.read_xml(path)

    def extract_lon_lat_from_geometry(self, dataframe):

        try:
            dataframe["lon"] = dataframe["geometry"].apply(lambda x: x.x)
            dataframe["lat"] = dataframe["geometry"].apply(lambda x: x.y)
            return dataframe
        except:
            raise ValueError("Could not extract lon/lat from geometry.")

    def add_name_to_dataframe(self, dataframe):
        """Adds a name column to a dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to add the name column to.

        Returns
        -------
        pd.DataFrame
            Dataframe with the name column added.
        """
        dataframe["name"] = [
            f"{self.name_prefix}-{i}" for i in range(1, len(dataframe) + 1)
        ]
        return dataframe

    def load_file(self):
        """Determines the file type and loads the file into a pandas dataframe.
        Checks if the file exists and if the dataframe contains the longitude and latitude columns.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the longitude and latitude columns.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the dataframe does not contain the longitude and latitude columns.
        AttributeError
            If the file type is not supported.
        """
        path = self.path
        dataframe = None

        if type(path) == pd.DataFrame:
            dataframe = path
        elif not self.check_file_exists(path):
            raise FileNotFoundError("File not found: {}".format(path))
        elif path.endswith(".csv"):
            dataframe = self.load_csv_to_dataframe(path)
        elif path.endswith(".kml"):
            dataframe = self.load_kml_to_dataframe(path)
        elif path.endswith(".geojson"):
            dataframe = self.load_geojson_to_dataframe(path)
        elif path.endswith(".json"):
            dataframe = self.load_json_to_dataframe(path)
        elif path.endswith(".xml"):
            dataframe = self.load_xml_to_dataframe(path)
        else:
            raise ValueError("Filetype not supported.")
        if not self.check_if_dataframe_contains_lon_lat(dataframe):
            if "geometry" in dataframe.columns:
                self.extract_lon_lat_from_geometry(dataframe)
        if self.name_prefix is not None:
            dataframe = self.add_name_to_dataframe(dataframe)
        if not self.check_if_dataframe_contains_lon_lat(dataframe):
            raise AttributeError(
                "Dataframe does not contain longitude and latitude columns."
            )

        if self.check_name and not self.check_if_dataframe_contains_name(dataframe):
            raise AttributeError("Dataframe does not contain name column.")
        elif self.check_if_dataframe_contains_name(dataframe):
            dataframe.set_index(self.name_key, inplace=True)
        return dataframe

    @staticmethod
    def load_to_dataframe(path, name_prefix=None):
        """ Loads a file containing AOIs into a Pandas DataFrame.

        Parameters
        ----------
        path : string
            Path to the file.
        name_prefix : string, optional
            Prefix to add to the name column.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the POIs.
        """
        loader = DataLoader(path, name_prefix)
        return loader.dataframe
