from dataset_generation.PlanetSampler import PlanetSampler
from dataset_generation.Stratifier import Stratifier
from dataset_generation.Visualiser import Visualiser
import os
import pandas as pd
import numpy as np


class Rebalancer:
    """ Rebalances the AOIs by sampling a large number of points to more precisely capture the underlying distribution,
    applying a rebalancing function to the distribution (e.g. sqrt, cbrt) and then rebalancing the AOIs 
    to match the new distribution.
    """

    def __init__(
        self,
        number_of_points_to_sample,
        number_of_points_calculate_the_distributions=1000000,
        save_randomly_sampled_points=True,
    ):
        """ Initialises the Rebalancer class.

        Parameters
        ----------
        number_of_points_to_sample : int
            The number of points to sample that will be rebalanced.
        number_of_points_calculate_the_distributions : int, optional
            The number of points to sample to calculate the underlying distribution with, by default 1000000
        save_randomly_sampled_points : bool, optional
            Whether to save the randomly sampled points to a CSV file, by default True
        """
        self.number_of_points_to_sample = number_of_points_to_sample
        self.number_of_points_calculate_the_distributions = (
            number_of_points_calculate_the_distributions
        )
        self.save_randomly_sampled_points = save_randomly_sampled_points
        self.sample_distribution_points()
        self.stratify_sampled_distribution_points()
        self.calculate_original_distributions()
        self.visualiser = Visualiser(self.stratified_distribution_points)
        self.rebalanced_distributions = {}

    def sample_distribution_points(self):
        """Samples a large number of points using the PlanetSampler class in order to calculate the original distributions.
        """
        print(
            f"Sampling {self.number_of_points_calculate_the_distributions} points to calculate the distribution with."
        )
        if self.number_of_points_calculate_the_distributions > 100000:
            print("This will take a while.")
        self.sampler = PlanetSampler(
            number_of_points=self.number_of_points_calculate_the_distributions
        )
        self.sampled_distribution_points = self.sampler.sampled_points

    def stratify_sampled_distribution_points(self):
        """Stratifies the large number of sampled points using the Stratifier class.
        """
        print("Stratifying the sampled distribution points.")
        if self.number_of_points_calculate_the_distributions > 100000:
            print("This will take a while.")
        self.Stratifier = Stratifier(self.sampled_distribution_points)
        self.stratified_distribution_points = self.Stratifier.points
        if self.save_randomly_sampled_points:
            self.save_sampled_points()

    def save_sampled_points(self, directory_path="data/sampled_points"):
        """Saves the sampled points to a CSV file.

        Parameters
        ----------
        directory_path : str, optional
            The directory path to save the sampled points to, by default 'sampled_points'
        """
        csv_path = f"{directory_path}/sampled_points.csv"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self.Stratifier.to_csv(csv_path)

    def calculate_original_distributions(
        self, stratification_datasets=["IPCC", "LCCS", "SMOD"]
    ):
        """Calculates the original distributions for the given stratification datasets within the large number of randomly sampled points.

        Parameters
        ----------
        classes : list, optional
            The classes to calculate the original distributions for, by default ['IPCC', 'LCCS', 'SMOD']
        """
        print("Calculating the original distributions.")
        self.original_distributions = {
            stratification_dataset: self.calculate_original_distribution_for_stratification_dataset(
                stratification_dataset
            )
            for stratification_dataset in stratification_datasets
        }

    def calculate_original_distribution_for_stratification_dataset(
        self, stratification_dataset
    ):
        """Calculates the original distribution for the given stratification dataset.

        Parameters
        ----------
        class_name : str
            The name of the class to calculate the original distribution for.

        Returns
        -------
        pandas.DataFrame
            The original distribution (a class and its frequency) for the given stratification dataset.
        """
        class_distribution = pd.DataFrame()
        class_distribution["count"] = self.stratified_distribution_points.groupby(
            stratification_dataset
        ).agg(({stratification_dataset: "count"}))
        return class_distribution.reset_index()

    def calculate_original_distribution_for_subclass(
        self, subclass_points, stratification_dataset
    ):
        class_distribution = pd.DataFrame()
        class_distribution["count"] = subclass_points.groupby(
            stratification_dataset
        ).agg(({stratification_dataset: "count"}))
        return class_distribution.reset_index()

    def get_original_distribution(self, stratification_dataset):
        """Gets the original distribution for the given stratification dataset.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to get the original distribution for.

        Returns
        -------
        pandas.DataFrame
            The original distribution (a class and its frequency) for the given stratification dataset.
        """
        return self.original_distributions[stratification_dataset]

    def rebalance_within_class(
        self,
        selection_dataset,
        class_name,
        rebalancing_dataset,
        rebalancing_function=np.cbrt,
        target_number_of_points=None,
    ):
        """ Rebalances the AOIs within a given class using a rebalancing function.

        Parameters
        ----------
        selection_dataset : str
            The name of the selection dataset to rebalance.
        class_name : str
            The name of the class to rebalance.
        rebalancing_dataset : str
            The name of the dataset of points providing the underlying distribution.
        rebalancing_function : function, optional
            The rebalancing function to use, by default np.cbrt
        target_number_of_points : int, optional
            The target number of points to rebalance to, by default None
        """
        subdistribution_name = f"{selection_dataset}_{class_name}_{rebalancing_dataset}"
        subclass_points = self.stratified_distribution_points.loc[
            self.stratified_distribution_points[selection_dataset] == class_name
        ].copy()
        self.original_distributions[
            subdistribution_name
        ] = self.calculate_original_distribution_for_subclass(
            subclass_points, rebalancing_dataset
        )
        self.rebalanced_distributions[
            subdistribution_name
        ] = self.rebalance_original_distribution_for_stratification_dataset(
            subdistribution_name, rebalancing_function, target_number_of_points
        )

    def rebalance_original_distributions(self, rebalancing_function=np.cbrt):
        """Rebalances the original distributions for the stratification datasets using the given rebalancing function.

        Parameters
        ----------
        rebalancing_function : function, optional
            The function to rebalance the original distributions with, by default np.cbrt
        """
        for stratification_dataset in self.original_distributions.keys():
            self.rebalanced_distributions[
                stratification_dataset
            ] = self.rebalance_original_distribution_for_stratification_dataset(
                stratification_dataset, rebalancing_function
            )

    def rebalance_original_distribution_for_stratification_dataset(
        self,
        stratification_dataset,
        rebalancing_function=np.cbrt,
        target_number_of_points=None,
    ):
        """Rebalances the original distribution for the given stratification dataset using the given rebalanced function.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to rebalance the original distribution for.
        rebalancing_function : function, optional
            The function to rebalance the original distribution with, by default np.cbrt

        Returns
        -------
        pandas.DataFrame
            The rebalanced distribution (a class and its adjusted frequency) for the given stratification dataset.
        """
        rebalanced_distribution = self.original_distributions[
            stratification_dataset
        ].copy()
        rebalanced_distribution["count"] = rebalanced_distribution["count"].apply(
            lambda x: rebalancing_function(x)
        )
        rebalanced_distribution["percentage"] = rebalanced_distribution.apply(
            lambda x: x["count"] / sum(rebalanced_distribution["count"]), axis=1
        )
        # TODO (ori): check for small numbers
        number_of_pois = self.distribute_pois_using_largest_remainder(
            rebalanced_distribution, target_number_of_points
        )
        rebalanced_distribution = self.original_distributions[
            stratification_dataset
        ].copy()
        rebalanced_distribution["count"] = number_of_pois
        return rebalanced_distribution

    def distribute_pois_using_largest_remainder(
        self, rebalanced_distribution, target_number_of_points=None
    ):
        """Distributes the targeted number of POIs using the largest remainder method.

        Parameters
        ----------
        rebalanced_distribution : pandas.DataFrame
            The rebalanced distribution (a class and its adjusted frequency) for the given stratification dataset.

        Returns
        -------
        pandas.Series
            The number of POIs assigned to each class using the rebalanced distribution.
        """
        target_number_of_points = (
            target_number_of_points
            if target_number_of_points is not None
            else self.number_of_points_to_sample
        )
        remainders = (
            rebalanced_distribution["percentage"]
            .apply(lambda x: (x * 100) % 1)
            .sort_values(ascending=False)
            .index
        )

        rebalanced_distribution["rounded_percentage"] = rebalanced_distribution[
            "percentage"
        ].apply(lambda x: round(x, 2))

        rebalanced_distribution["number_of_pois"] = rebalanced_distribution[
            "rounded_percentage"
        ].apply(lambda x: x * target_number_of_points)
        remainder = target_number_of_points - sum(
            rebalanced_distribution["number_of_pois"]
        )

        current_index = 0
        while remainder > 0:
            rebalanced_distribution.loc[
                remainders[current_index % len(remainders)], "number_of_pois"
            ] += 1
            current_index += 1
            remainder -= 1

        return rebalanced_distribution["number_of_pois"].apply(int)

    def check_total_number_of_rebalanced_pois(self, stratification_dataset):
        """Checks the difference between the targeted number of POIs and the total number of POIs after rebalancing for the given stratification dataset.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to check the total number of rebalanced POIs for.
        """
        print(
            f"The target number of points to sample is {self.number_of_points_to_sample}."
        )
        print(
            f"The total number of points after rebalancing on {stratification_dataset} is {sum(self.stratified_distributions[stratification_dataset]['number_of_pois'])}."
        )
        if (
            sum(self.rebalanced_distributions[stratification_dataset]["number_of_pois"])
            == self.number_of_points_to_sample
        ):
            print(
                "The total number of points after rebalancing is equal to the target number of points to sample."
            )
        else:
            print(
                "WARNING: The total number of points after rebalancing is not equal to the target number of points to sample. That's okay if you're rebalancing a fraction of the original number of points."
            )

    def visualise_original_distribution(self, stratification_dataset):
        """Visualises the original distribution for the given stratification dataset.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to visualise the original distribution for.

        Returns
        -------
        matplotlib.pyplot.Figure
            The visualised distribution for the given stratification dataset.
        """
        if self.original_distributions[stratification_dataset] is None:
            raise ValueError(
                f"The original distribution for {stratification_dataset} is not available."
            )
        self.visualiser.update_data(self.original_distributions[stratification_dataset])
        return self.visualiser.visualise_distribution(
            stratification_dataset,
            title=f"Original distribution of {stratification_dataset}",
        )

    def visualise_rebalanced_distribution(self, stratification_dataset):
        """Visualises the rebalanced distribution for the given stratification dataset.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to visualise the rebalanced distribution for.

        Returns
        -------
        matplotlib.pyplot.Figure
            The visualised distribution for the given stratification dataset.
        """
        if self.rebalanced_distributions[stratification_dataset] is None:
            raise ValueError(
                f"The rebalanced distribution for {stratification_dataset} is None."
            )
        self.visualiser.update_data(
            self.rebalanced_distributions[stratification_dataset]
        )
        return self.visualiser.visualise_distribution(
            stratification_dataset,
            title=f"Rebalanced distribution of {stratification_dataset}",
        )

    def visualise_sampled_distribution(self, stratification_dataset):
        """ Visualises the sampled distribution for the given stratification dataset.

        Parameters
        ----------
        stratification_dataset : str
            The name of the stratification dataset to visualise the sampled distribution for.

        Returns
        -------
        matplotlib.pyplot.Figure
            The visualised distribution for the given stratification dataset.

        Raises
        ------
        ValueError
            If the sampled distribution for the given stratification dataset is not set.
        """
        if self.sample_distribution is None:
            raise ValueError(
                f"The sample distribution for {stratification_dataset} is not set."
            )

        self.visualiser.update_data(
            self.calculate_original_distribution_for_subclass(
                self.sampled_points, stratification_dataset
            )
        )
        return self.visualiser.visualise_distribution(
            stratification_dataset,
            title=f"Sampled distribution of {stratification_dataset}",
        )

    def sample_distribution(self, stratification_dataset, distribution):
        """Samples the given distribution.

        Parameters
        ----------
        distribution : pandas.DataFrame
            The distribution to sample.

        Returns
        -------
        pandas.Series
            The sampled distribution.
        """
        self.sampled_points = pd.DataFrame()
        for class_name, count in distribution.itertuples(index=False):
            available_points = self.stratified_distribution_points[
                self.stratified_distribution_points[stratification_dataset]
                == class_name
            ]
            while len(available_points) < count:
                additional_points = self.sampler.sample_points(count * 2)
                additional_points = Stratifier(additional_points).points
                additional_points = additional_points[
                    additional_points[stratification_dataset] == class_name
                ]
                available_points = pd.concat([available_points, additional_points])
            self.sampled_points = pd.concat(
                [self.sampled_points, available_points.sample(count)]
            )
        return self.sampled_points
