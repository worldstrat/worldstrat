import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

from datetime import date
from dateutil import parser

from shapely.geometry.polygon import Polygon
from pyproj import Geod
from dataset_generation.AOIGenerator import AOIGenerator
from dataset_generation.Visualiser import Visualiser
import matplotlib

pd.options.mode.chained_assignment = None


class CatalogueFilter:
    """ Filters AOIs based on:
    - The number of revisits that are targeted/required.
    - The maximum cloud cover percentage that is allowed (if set).
    - The maximum number of days away from the target date (delta days) that is allowed (if set).
    - The relative importance between the cloud cover percentage and the delta days 
      (having a revisit closer to the target date vs. having a revisit with smaller cloud cover percentage).
    """

    def __init__(
        self,
        aois,
        catalogue,
        target_date=None,
        aoi_area=None,
        cloud_cover_importance=1,
        delta_importance=1,
        n_target_revisits=8,
        max_cloud_cover_percentage=None,
        max_delta_days=None,
    ):
        """
        Initialises the CatalogueFilter class.

        Parameters
        ----------
        cloud_cover_importance : int
            The importance of the cloud cover percentage. The higher the value, the more important the cloud cover percentage is.
        delta_importance : int
            The importance of the delta days. The higher the value, the more important the delta days is.
        n_target_revisits : int, optional
            The number of revisits that are required to be included in the filtered AOIs. The default is 8.
        max_cloud_cover_percentage : float, optional
            The maximum cloud cover percentage that is allowed. The default is None.
        max_delta_days : int, optional
            The maximum delta days (absolute value) that is allowed. The default is None.
        """
        AOIGenerator.check_aois(aois)
        self.aois = aois.copy()
        self.set_aoi_area(aoi_area)
        self.cloud_cover_importance = cloud_cover_importance
        self.delta_importance = delta_importance
        self.n_target_revisits = n_target_revisits
        self.max_cloud_cover_percentage = max_cloud_cover_percentage
        self.max_delta_days = max_delta_days
        self.catalogue = catalogue.copy()
        self.set_target_date(target_date)
        self.calculate_filter_values()
        self.filter_aois()
        self.check_parameters()
        self.check_filtered()
        self.visualiser = Visualiser(self.filtered_catalogue)

    def check_parameters(self):
        """ Checks if there are enough revisits that satisfy the specified filter parameters. 
        If not, returns the minimum required cloud cover and delta needed to satisfy the number of targeted/required revisits.
        """

        minimum_recommended_parameters = {
            "n_target_revisits": None,
            "max_cloud_cover_percentage": None,
            "max_delta_days": None,
        }

        aoi_parameters = {}
        for aoi in set(self.catalogue.index):
            aoi_parameters[aoi] = minimum_recommended_parameters.copy()
            revisits = self.catalogue.loc[aoi]
            if len(revisits) < self.n_target_revisits:
                aoi_parameters[aoi]["n_target_revisits"] = self.n_target_revisits
            minimum_needed_delta = CatalogueFilter.check_minimum_required_delta(
                revisits, self.max_delta_days, self.n_target_revisits
            )
            minimum_needed_cloud_cover = CatalogueFilter.check_minimum_required_cloud_cover(
                revisits, self.max_cloud_cover_percentage, self.n_target_revisits
            )
            (
                combination_delta,
                combination_cloud_cover,
            ) = CatalogueFilter.check_combination_delta_cloud_cover(
                revisits,
                self.max_delta_days,
                self.max_cloud_cover_percentage,
                self.n_target_revisits,
            )

            if minimum_needed_delta:
                aoi_parameters[aoi]["max_delta_days"] = minimum_needed_delta

            if minimum_needed_cloud_cover:
                aoi_parameters[aoi][
                    "max_cloud_cover_percentage"
                ] = minimum_needed_cloud_cover

            if combination_delta or combination_cloud_cover:
                aoi_parameters[aoi][
                    "max_delta_days"
                ] = f"{combination_delta} if using {self.maximum_cloud_cover_percentage}"
                aoi_parameters[aoi][
                    "max_cloud_cover_percentage"
                ] = f"{combination_cloud_cover} if using {self.maximum_delta_days}"

        for aoi, parameters in aoi_parameters.items():
            if parameters != minimum_recommended_parameters:
                recommended = {k: v for k, v in parameters.items() if v is not None}
                print(
                    f"WARNING: AOI {aoi} won't have the targeted {self.n_target_revisits} unless the following minimum parameters are used: {recommended}\n"
                )

    def visualise_filtered_catalogue(self):
        """ Visualises the filtered catalogue on a map.

        Returns
        -------
        matplotlib.pyplot.figure
            The figure containing the filtered catalogue visualised on a map.
        """
        merged_aois_filtered_catalogue = self.filtered_catalogue.merge(
            self.aois[["IPCC", "SMOD", "LCCS"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        self.visualiser.update_data(merged_aois_filtered_catalogue)
        return self.visualiser.visualise_aoi_and_revisit_coverage()

    def visualise_delta_days(self):
        """ Visualises the time difference between the revisits and the target date through a histogram.

        Returns
        -------
        matplotlib.pyplot.figure
            The figure containing the histogram.
        """
        matplotlib.style.use("ggplot")
        return self.filtered_catalogue["delta"].plot.hist(
            figsize=(8, 8), title="Number of days away from the target date (delta)"
        )

    def visualise_cloud_cover_percentage(self):
        """ Visualises the cloud cover percentage of the revisits through a histogram.

        Returns
        -------
        matplotlib.pyplot.figure
            The figure containing the histogram.
        """
        matplotlib.style.use("ggplot")
        return self.filtered_catalogue["cloud_cover"].plot.hist(
            figsize=(8, 8), title="Cloud cover percentage"
        )

    @staticmethod
    def check_minimum_required_delta(revisits, max_delta_days, n_target_revisits):
        """ Checks if it's possible to satisfy the number of target revisits with the given maximum delta days filter.
        
        If there aren't enough revisits, returns the minimum delta days needed 
        to satisfy the number of targeted/required revisits.

        If there are enough revisits, returns None.

        Parameters
        ----------
        revisits : pandas.Series
            The available revisits for the AOI.    
        max_delta_days : int
            The maximum delta days (absolute value) that is allowed.
        n_target_revisits : int
            The number of target revisits.

        Returns
        -------
        int or None
            The minimum delta days needed to satisfy the number of targeted/required revisits.
        """
        if max_delta_days is not None:
            ranked_by_delta = revisits.sort_values(by="delta", ascending=True, key=abs)
            minimum_needed_delta = abs(
                ranked_by_delta.head(n_target_revisits).iloc[-1]["delta"]
            )
            if minimum_needed_delta > max_delta_days:
                return minimum_needed_delta
        return None

    @staticmethod
    def check_minimum_required_cloud_cover(
        revisits, max_cloud_cover_percentage, n_target_revisits
    ):
        """ Checks if it's possible to satisfy the number of target revisits with the given maximum cloud cover filter.

        If there aren't enough revisits, returns the minimum cloud cover percentage needed
        to satisfy the number of targeted/required revisits.

        If there are enough revisits, returns None.

        Parameters
        ----------
        revisits : pandas.Series
            The available revisits for the AOI.
        max_cloud_cover_percentage : int
            The maximum cloud cover percentage that is allowed.
        n_target_revisits : int
            The number of target revisits.

        Returns
        -------
        int or None
            The minimum cloud cover percentage needed to satisfy the number of targeted/required revisits.
        """
        if max_cloud_cover_percentage is not None:
            ranked_by_cloud_cover = revisits.sort_values(
                by="cloud_cover", ascending=True
            )
            minimum_needed_cloud_cover = ranked_by_cloud_cover.head(
                n_target_revisits
            ).iloc[-1]["cloud_cover"]
            if minimum_needed_cloud_cover > max_cloud_cover_percentage:
                return minimum_needed_cloud_cover

    @staticmethod
    def check_combination_delta_cloud_cover(
        revisits, max_delta_days, max_cloud_cover_percentage, n_target_revisits
    ):
        """ Checks if it's possible to satisfy the number of target revisits 
        with the given combination of delta days and cloud cover filters.

        If there aren't enough revisits, returns the minimum delta days and
        cloud cover percentage needed to satisfy the number of targeted/required revisits.

        If it's impossible to find enough revisits with any combination of delta days and cloud cover filters,
        prints out a warning and returns None.


        If there are enough revisits or there is no filter set, returns None.


        Parameters
        ----------
        revisits : pandas.Series
            The available revisits for the AOI.
        max_delta_days : int
            The maximum delta days (absolute value) that is allowed.
        max_cloud_cover_percentage : int
            The maximum cloud cover percentage that is allowed.
        n_target_revisits : int
            The number of target revisits.

        Returns
        -------
        tuple of int or tuple of None
            The minimum delta days and cloud cover percentage needed to satisfy the number of targeted/required revisits.
        """
        if max_cloud_cover_percentage is not None and max_delta_days is not None:
            minimum_needed_cloud_cover = CatalogueFilter.check_minimum_required_cloud_cover(
                revisits, max_cloud_cover_percentage, n_target_revisits
            )
            minimum_needed_delta = CatalogueFilter.check_minimum_required_delta(
                revisits, max_delta_days, n_target_revisits
            )

            combination_cloud_delta = revisits[
                revisits["cloud_cover"] <= minimum_needed_cloud_cover
            ]
            combination_delta_cloud = revisits[
                abs(revisits["delta"]) <= minimum_needed_delta
            ]
            if (
                len(combination_cloud_delta) < n_target_revisits
                or len(combination_delta_cloud) < n_target_revisits
            ):
                print(
                    f"WARNING: Impossible to find enough revisits that satisfy the minimum cloud cover percentage of {max_cloud_cover_percentage} and the minimum delta days of {max_delta_days}"
                )
                return None, None
            else:
                minimum_combination_cloud_delta = CatalogueFilter.check_minimum_required_delta(
                    combination_cloud_delta, max_delta_days, n_target_revisits
                )
                minimum_combination_delta_cloud = CatalogueFilter.check_minimum_required_cloud_cover(
                    combination_delta_cloud,
                    max_cloud_cover_percentage,
                    n_target_revisits,
                )
                return minimum_combination_cloud_delta, minimum_combination_delta_cloud
        else:
            return None, None

    def calculate_filter_values(self):
        """ Calculates the column values by which the catalogue can be filtered:
        - The intersection area of the defined AOI and the actual available revisits.
        - The time difference in days between the available revisits and the target date.
        
        Automatically filters the revisits that don't cover the entire AOI.
        """
        self.calculate_revisit_intersection_area_for_aois_and_catalogue_revisits()
        self.calculate_delta_for_revisits()
        self.filter_by_intersection_area()

    def check_filtered(self):
        """
        Checks the filtered AOIs.

        Raises
        ------
        ValueError
            If the filtered AOIs are empty.
        """
        if len(self.filtered_catalogue) == 0:
            raise ValueError(
                f"The filtered AOIs are empty. Please check the parameters."
            )
        for aoi in set(self.catalogue.index):
            if aoi not in self.filtered_catalogue.index:
                print(
                    f"WARNING: no revisits selected for {aoi}, parameters may be too restrictive."
                )
            elif len(self.filtered_catalogue.loc[aoi]) < self.n_target_revisits:
                print(
                    f"WARNING: AOI {aoi} has less than {self.n_target_revisits} revisits."
                )

    def set_target_date(self, target_date):
        """ Sets the target date around which the revisits are searched for.

        Parameters
        ----------
        target_date : datetime.date, string or column name
            The target date around which the revisits are searched for.

        Raises
        ------
        ValueError
            If the target date is not a valid date.
        """
        if str(target_date) in self.aois.columns:
            self.aois["target_date"] = pd.to_datetime(self.aois[target_date])
            return
        elif "target_date" in self.aois.columns:
            self.aois["target_date"] = pd.to_datetime(self.aois["target_date"])
            return

        elif target_date is None:
            raise ValueError(
                f"The AOIs do not have a per-AOI target date defined, and a global target date was not passed."
            )
        elif isinstance(target_date, date):
            self.target_date = target_date
            self.aois["target_date"] = target_date
        else:
            try:
                self.target_date = parser.parse(target_date).date()
                self.target_date = target_date
                self.aois["target_date"] = target_date
            except:
                print(
                    f"{target_date} is not a valid date. Please use a valid date, date string or a column name."
                )
        print(
            f"Per-AOI target date not found, setting the target date to be {self.target_date} for all AOIs."
        )
        self.aois["target_date"] = target_date

    def set_aoi_area(self, aoi_area):
        """ Sets the target area of the AOIs.
        If the target area is not defined, prints out a warning, 
        selects the first AOI in the dataframe and uses its area as the target area.

        Parameters
        ----------
        aoi_area : float
            The target area of the AOIs.
        """
        if aoi_area is not None:
            self.aoi_area = aoi_area
            print(f"AOI area set to {self.aoi_area}.")
        else:
            self.aoi_area = self.aois.head(1)["area"].values[0]
            print(
                f"AOI area wasn't explicitly passed, sampling one AOI and taking its area ({self.aoi_area}) and assuming all AOIs have the same area."
            )

    def filter_aois(self):
        """
        Filters the AOIs based on the initialised parameters.

        Parameters
        ----------
        aois : pandas.DataFrame
            AOIs to be filtered, containing the cloud cover percentage and the delta days.

        Returns
        -------
        pandas.DataFrame
            Filtered AOIs.
        """
        print(f"Filtering AOIs based on the following parameters:")
        print(f"Cloud cover importance: {self.cloud_cover_importance}")
        print(f"Delta importance: {self.delta_importance}")
        print(f"Number of target revisits: {self.n_target_revisits}")
        print(f"Maximum cloud cover percentage: {self.max_cloud_cover_percentage}")
        print(f"Maximum delta days: {self.max_delta_days}")

        self.filtered_catalogue = pd.DataFrame()
        self.filtered_catalogue = CatalogueFilter.select_revisits_by_delta_and_cloud_cover(
            self.catalogue,
            self.n_target_revisits,
            self.delta_importance,
            self.cloud_cover_importance,
            self.max_delta_days,
            self.max_cloud_cover_percentage,
        )
        self.filtered_catalogue["n"] = (
            self.filtered_catalogue.groupby(self.filtered_catalogue.index).cumcount()
            + 1
        )
        self.filtered_catalogue["name"] = self.filtered_catalogue.apply(
            lambda x: self.aois.loc[x.name]["name"], axis=1
        )
        self.filtered_catalogue["bounds"] = self.filtered_catalogue.apply(
            lambda x: self.aois.loc[x.name]["bounds"], axis=1
        )
        self.filtered_catalogue["target_date"] = self.filtered_catalogue.apply(
            lambda x: self.aois.loc[x.name]["target_date"], axis=1
        )
        print(f"Total revisits: {len(self.filtered_catalogue)}")

    def calculate_delta_for_revisits(self):
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
        tqdm.pandas(
            desc="Calculating delta between target date and Sentinel2 catalogue revisit"
        )
        self.catalogue["delta"] = self.catalogue.progress_apply(
            lambda row: (
                row["datetime"].date() - pd.to_datetime(self.aois.loc[row.name]["target_date"]).date()
            ).days,
            axis=1,
        )

    def filter_by_intersection_area(self):
        self.catalogue = self.catalogue.loc[
            self.catalogue["area"] > self.aoi_area * 0.999
        ]

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
        self.catalogue["area"] = self.catalogue.progress_apply(
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

    @staticmethod
    def select_revisits_by_delta_and_cloud_cover(
        revisits,
        target_revisits,
        delta_importance=1,
        cloud_cover_importance=1,
        max_delta=None,
        max_cloud_cover=None,
    ):
        """Selects the revisits that are closest to the target_revisits.

        Parameters
        ----------
        revisits : pandas.DataFrame
            DataFrame with the revisits.
        target_revisits : int
            Number of revisits that should be selected.
        delta_importance : int, optional
            Importance of the delta. The higher the value, the more important the delta is, by default 1.
        cloud_cover_importance : int, optional
            Importance of the cloud cover. The higher the value, the more important the cloud cover is, by default 1.
        max_delta : _type_, optional
            The maximum delta days (absolute value) that is allowed. The default is None.
        max_cloud_cover : _type_, optional
            The maximum cloud cover percentage that is allowed. The default is None.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the revisits that are closest to the target_revisits.
        """
        result = pd.DataFrame()
        if max_delta is not None:
            revisits = revisits[abs(revisits["delta"]) <= max_delta]
        if max_cloud_cover is not None:
            revisits = revisits[revisits["cloud_cover"] <= max_cloud_cover]
        for index in tqdm(set(revisits.index)):

            if isinstance(revisits.loc[index], pd.Series):
                revisits.loc[index, "rank"] = 1
                selected = pd.DataFrame([revisits.loc[index]])
            else:
                revisits.loc[index, "rank"] = (
                    abs(revisits.loc[index].delta).rank() * delta_importance
                    + revisits.loc[index].cloud_cover.rank() * cloud_cover_importance
                )
                selected = (
                    revisits.loc[index]
                    .sort_values("rank", ascending=True)
                    .head(target_revisits)
                )
            if len(selected) < target_revisits:
                print(
                    f"{index} has only {len(selected)} revisits for the specified criteria."
                )
            result = pd.concat([result, selected])
        return result

    @staticmethod
    def filter_by_days(revisits, date, days):
        """Filters the revisits by keeping only the revisits that are within the specified days of the specified date.

        Parameters
        ----------
        revisits : pandas.DataFrame
            The revisits to filter.
        date : datetime.datetime
            The date to filter the revisits by.
        days : int
            The number of days to filter the revisits by.

        Returns
        -------
        pandas.DataFrame
            The filtered revisits.
        """
        # Convert the date to a datetime object
        date = datetime.strptime(date, "%Y-%m-%d")
        # Create a list of datetime objects
        date_list = [date + datetime.timedelta(days=x) for x in range(days)]
        # Convert the list of datetime objects to a list of strings
        date_list = [datetime.strftime(x, "%Y-%m-%d") for x in date_list]
        # Filter the dataframe by the date_list
        return revisits[revisits["date"].isin(date_list)]

    @staticmethod
    def filter_by_cloud_cover(revisits, maximum_cloud_cover):
        """Filter out all the revisits in the Pandas dataframe that have a cloud cover percentage greater than the maximum_cloud_cover.

        Parameters
        ----------
        revisits : pandas.DataFrame
            The dataframe containing the revisits.
        maximum_cloud_cover : float
            The maximum cloud cover percentage (0.-100.).

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the revisits that have a cloud cover percentage less than the maximum_cloud_cover.
        """
        pois_before = set(revisits.index)
        filtered = revisits[revisits["cloud_cover"] <= maximum_cloud_cover]
        pois_after = set(filtered.index)
        if len(pois_before - pois_after) != 0:
            print(
                f"Warning, POIs {pois_before - pois_after} do not have any revisits with cloud cover less than {maximum_cloud_cover}."
            )
        return filtered

    @staticmethod
    def filter_by_column(revisits, column, value, filter_type):
        """Filter out all the revisits in the dataframe by a specific value in a specific column with a numerical value.
        Available filter types are 'equal', 'less_than', 'greater_than', 'less_than_or_equal', 'greater_than_or_equal', or their symbolic equivalents.

        Parameters
        ----------
        revisits : pandas.DataFrame
            The dataframe containing the revisits.
        column : str
            The column to filter by.
        value : float or int
            The value to filter by.
        filter_type : str
            The type of filter to apply.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the rows who satisfy the filter.
        """
        if filter_type == ">" or filter_type == "greater_than":
            return revisits[revisits[column] > value]
        elif filter_type == "<" or filter_type == "less_than":
            return revisits[revisits[column] < value]
        elif filter_type == ">=" or filter_type == "greater_than_or_equal":
            return revisits[revisits[column] >= value]
        elif filter_type == "<=" or filter_type == "less_than_or_equal":
            return revisits[revisits[column] <= value]
        elif filter_type == "=" or filter_type == "equal":
            return revisits[revisits[column] == value]
        else:
            raise ValueError(f"The filter_type {filter_type} is not supported.")

    @staticmethod
    def filter_revisits_by_intersection_area(
        revisits, minimum_wanted_revisits=8, target_area=2.5
    ):
        """Filter out all the revisits in the Pandas dataframe that have an intersection area less than the target_area.

        Parameters
        ----------
        revisits : pandas.DataFrame
            The dataframe containing the revisits.
        minimum_wanted_revisits : int, optional
            The minimum number of revisits that are required to be included in the output, by default 8.
        target_area : float, optional
            The target area in square kilometers, by default 2.5.

        Returns
        -------
        pandas.DataFrame
            The dataframe containing the revisits that have an intersection area greater than the target_area.
        """
        pois_before = set(revisits.index)
        filtered = revisits[revisits["area"] >= target_area]
        pois_after = set(filtered.index)
        if len(pois_before - pois_after) != 0:
            print(
                f"Warning, POIs {pois_before - pois_after} do not have any revisits with an intersection area greater than {target_area}."
            )
        for poi_index, group_size in (
            filtered.groupby(filtered.index).size().iteritems()
        ):
            if group_size < minimum_wanted_revisits:
                print(
                    f"Warning, POI {poi_index} has less than {minimum_wanted_revisits} revisits after filtering."
                )
        return filtered

    @staticmethod
    def verify_intersection_area(revisits, target_area=2.5):
        """Verify that all the revisits in the Pandas dataframe have an intersection area greater than the target_area.

        Parameters
        ----------
        revisits : pandas.DataFrame
            The dataframe containing the revisits.
        target_area : float, optional
            The target area in km2, by default 2.5.

        Returns
        -------
        bool
            True if all the revisits have an intersection area greater than the target_area.
        """
        return (
            "All good."
            if len(revisits[revisits["area"] >= target_area]) == len(revisits)
            else f"WARNING: not all revisits have an area larger than {target_area}"
        )

