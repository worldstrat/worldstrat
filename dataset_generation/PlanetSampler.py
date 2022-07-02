from tqdm.autonotebook import tqdm
import pandas as pd
import global_land_mask as globe
from tqdm import tqdm
import numpy as np
from dataset_generation.Visualiser import Visualiser


class PlanetSampler:
    """ Samples points uniformly from the Earth.
    It can exculde oceans or sample uniformly on the globe.
    """

    def __init__(self, sampling_strategy="exclude oceans", number_of_points=None):
        """Initialises the PlanetSampler object.

        Parameters
        ----------
        sampling_strategy : str, optional
            Sampling strategy. Available strategies are 'exclude oceans' and 'uniform on globe'. 
            The default is 'exclude oceans'.
        number_of_points : int, optional
            Number of points to sample. If set, the points will be sampled upon initialisation. The default is None.
        """
        self.sampling_strategy = sampling_strategy
        self.number_of_points = number_of_points
        self.sampled_points = None
        if self.number_of_points is not None:
            self.sampled_points = self.sample_points()
        self.visualiser = Visualiser(self.sampled_points)

    def sample_points(self):
        """Samples points uniformly from the Earth. The possible sampling strategies are: 'exclude oceans', 'uniform on globe'.
        The 'exclude oceans' strategy samples points uniformly from the Earth, excluding the oceans, while keeping land water bodies. 
        The 'uniform on globe' strategy samples points uniformly from the Earth, including the oceans.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the sampled points.

        Raises
        ------
        ValueError
            If the sampling strategy is not supported.
        ValueError
            If the number of points is not set.
        """
        if self.number_of_points is None:
            raise ValueError("Number of points is not set.")
        if self.sampling_strategy == "uniform on globe":
            return PlanetSampler.uniform_on_sphere(self.number_of_points)
        elif self.sampling_strategy == "exclude oceans":
            return PlanetSampler.uniform_on_sphere_without_oceans(self.number_of_points)
        else:
            raise ValueError("The sampling strategy is not valid.")

    def export_to_csv(self, filename):
        """Exports the sampled points to a CSV file.

        Parameters
        ----------
        filename : str
            Name of the output CSV file.

        Raises
        ------
        ValueError
            If the number of points is not set.
        """
        if self.sampled_points is None:
            raise ValueError("The sampled points are not set.")
        self.sampled_points.to_csv(filename)

    def visualise(self):
        self.visualiser.update_data(self.sampled_points)
        return self.visualiser.visualise_points_on_map()

    @staticmethod
    def uniform_on_sphere(n):
        """ Sample lon, lat uniformly on the sphere.

        Simply follow http://corysimon.github.io/articles/uniformdistn-on-sphere/ 
        Note: this is not perfect as it does not take into account the ellipsoid nature of earth
        but is better than sampling in cartesians, which would over-sample the poles and under-sample
        the equator in terms of surface density.

        Parameters
        ------
        n : int
            Number of POIs to sample.

        Returns:
        ------
        pandas.DataFrame
            Dataframe containing the longitudes and latitudes, size (N_POI, 2), first column is lon, second column is lat.
        """

        lon_lat = np.random.random_sample(size=(n, 2))
        # Longitude between -180 and 180
        lon_lat[:, 0] = 360 * (lon_lat[:, 0] - 0.5)
        # Arccos is [0,180] with 0 at pole. We want latitude between -90 and 90 with 0 at equator.
        lon_lat[:, 1] = np.degrees(np.arccos(1 - 2 * lon_lat[:, 1])) - 90

        return pd.DataFrame(lon_lat, columns=["lon", "lat"])

    @staticmethod
    def uniform_on_sphere_without_oceans(n):
        """Sample lon, lat uniformly on the sphere, excluding oceans.
        Naively samples N points, then checks if any are in the ocean. If they are, sample again until a non-ocean point is found.

        Parameters
        ----------
        n : int
            Number of POIs to sample.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the longitudes and latitudes, size (N_POI, 2), first column is lon, second column is lat.
        """
        sampled_points = PlanetSampler.uniform_on_sphere(n)
        selected_points = pd.DataFrame()
        for lon, lat in tqdm(
            sampled_points.itertuples(index=False),
            total=n,
            desc="Sampling points on the planet",
        ):
            sample = {}
            while not globe.is_land(lat, lon):
                lon, lat = PlanetSampler.uniform_on_sphere(1).values[0]

            sample["lon"] = lon
            sample["lat"] = lat
            sample = pd.DataFrame([sample])
            selected_points = pd.concat([selected_points, sample])
        return selected_points
