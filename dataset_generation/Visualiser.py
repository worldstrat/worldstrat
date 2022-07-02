from folium.plugins import MarkerCluster
import folium
from shapely.geometry.polygon import Polygon
import plotnine as p9
from plotnine import ggplot, geom_bar, aes, ggtitle, xlab, ylab, facet_wrap

from dataset_generation.AOIGenerator import AOIGenerator
import tifffile as tiff
from PIL import Image
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import imageio
from IPython.display import Image as IPythonImage
import os
from tqdm.notebook import tqdm
import pandas as pd

class Visualiser:
    """ Provides visualisations for all classes. """

    def __init__(self, data):
        """ Initialises the visualiser with the data to be visualised. 

        Parameters
        ----------
        data : pandas.DataFrame
            The data to be visualised.
        """
        self.data = data

    def update_data(self, data):
        """ Updates the data to be visualised.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to be visualised.
        """
        self.data = data

    def visualise_points_on_map(self):
        """ Visualises points on a world map.

        Returns
        -------
        folium.Map
            The world map with the points visualised.
        """
        map = folium.Map([0, 0], zoom_start=3, tiles="CartoDB dark_matter")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        ).add_to(map)

        marker_cluster = MarkerCluster().add_to(map)
        for _, point in tqdm(self.data.iterrows(), total=len(self.data), desc="Visualising points on a map"):
            folium.Marker((point["lat"], point["lon"])).add_to(marker_cluster)
        folium.LatLngPopup().add_to(map)
        return map

    def visualise_aois_on_map(self):
        """ Visualises AOIs on a world map.

        Returns
        -------
        folium.Map
            The world map with the AOIs visualised.
        """
        map = folium.Map([0, 0], zoom_start=3, tiles="CartoDB dark_matter")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        ).add_to(map)
        aoi_style = {"fillColor": "#e74c3c", "color": "#c0392b"}
        marker_cluster = MarkerCluster().add_to(map)
        use_tooltip = pd.Series(['IPCC', 'LCCS', 'SMOD']).isin(self.data.columns).all() 
        for _, aoi in tqdm(self.data.iterrows(), total=len(self.data), desc='Visualising AOIs on map'):
            folium.Marker((aoi["lat"], aoi["lon"])).add_to(marker_cluster)
            aoi_polygon = Polygon(AOIGenerator.bounds_to_bounding_box(*aoi["bounds"]))
            aoi_geojson = folium.GeoJson(
                aoi_polygon, style_function=lambda x: aoi_style
            )
            if use_tooltip:
                aoi_tooltip = folium.Tooltip(
                    f"<strong>AOI:</strong> {aoi['name']} <br> <strong>IPCC:</strong> {aoi['IPCC']} <br> <strong>LCCS:</strong> {aoi['LCCS']} <br> <strong>SMOD:</strong> {aoi['SMOD']}"
                )
                aoi_tooltip.add_to(aoi_geojson)
            aoi_geojson.add_to(map)
        folium.LatLngPopup().add_to(map)
        return map

    def visualise_distribution(
        self, distribution_class_name, title=None, x_label=None, y_label=None
    ):
        """ Visualises the distribution of a given class within the data.

        Parameters
        ----------
        distribution_class_name : str
            The name of the class whose distribution is being visualised.
        title : str, optional
            The title of the plot. The default is None.
        x_label : str, optional
            The x-axis label of the plot. The default is None.
        y_label : str, optional
            The y-axis label of the plot. The default is None.

        Returns
        -------
        plotnine.ggplot
            The plot of the distribution.
        """
        columns = list(self.data.columns)
        title = f"Distribution of {distribution_class_name}" if title is None else title
        y_label = f"Frequency" if y_label is None else y_label
        plot = ggplot(self.data)
        if "count" not in columns:
            x_label = (
                f'Total number of points: {sum(self.data["number_of_pois"])}'
                if x_label is None
                else x_label
            )
            plot += aes(distribution_class_name, fill=distribution_class_name)
        elif "_" in distribution_class_name:
            title = f'Distribution of {distribution_class_name.split("_")[-1]} within {distribution_class_name.split("_")[0]}: {distribution_class_name.split("_")[1]}'
            distribution_class_name = distribution_class_name.split("_")[-1]
            x_label = (
                f'Total number of points: {sum(self.data["count"])}'
                if x_label is None
                else x_label
            )
            plot += aes(
                weight="count", x=distribution_class_name, fill=distribution_class_name
            )
        else:
            x_label = (
                f'Total number of points: {sum(self.data["count"])}'
                if x_label is None
                else x_label
            )
            plot += aes(
                weight="count", x=distribution_class_name, fill=distribution_class_name
            )
        plot += geom_bar()
        plot += p9.theme(
            subplots_adjust={"wspace": 0.25},
            figure_size=(16, 8),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
        )

        plot += ggtitle(title)
        plot += xlab(x_label)
        plot += ylab(y_label)
        return plot

    def visualise_conditional_distribution(
        self,
        conditional_class_name,
        distribution_class_name,
        title=None,
        x_label=None,
        y_label=None,
        ncol=2,
    ):
        """ Visualises the conditional distribution of a class given another class within the data.

        Parameters
        ----------
        conditional_class_name : str
            The name of the class on which the visualised class is being conditioned.
        distribution_class_name : str
            The name of the class whose distribution is being visualised.
        title : str, optional
            The title of the plot. The default is None.
        x_label : str, optional
            The x-axis label of the plot. The default is None.
        y_label : str, optional
            The y-axis label of the plot. The default is None.
        ncol : int, optional
            The number of columns in the plot. The default is 2.

        Returns
        -------
        plotnine.ggplot
            The plot of the conditional distribution.
        """
        title = (
            f"Distribution of {distribution_class_name} given {conditional_class_name}"
            if title is None
            else title
        )
        x_label = f"{distribution_class_name}" if x_label is None else x_label
        y_label = f"Frequency" if y_label is None else y_label
        plot = ggplot(self.data)
        plot += aes(distribution_class_name, fill=distribution_class_name)
        plot += geom_bar()
        plot += facet_wrap(
            conditional_class_name, labeller="label_both", scales="free_y", ncol=ncol
        )

        plot += p9.theme(
            subplots_adjust={"wspace": 0.25},
            figure_size=(16, 8),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
        )

        plot += ggtitle(title)
        plot += xlab(x_label)
        plot += ylab(y_label)
        return plot

    def visualise_aoi_and_revisit_coverage(self):
        """ Visualises AOIs and their available coverage on a world map.

        Returns
        -------
        plotnine.ggplot
            The plot of the AOIs and their available coverage.
        """
        map = folium.Map([0, 0], zoom_start=3, tiles="CartoDB dark_matter")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        ).add_to(map)

        aoi_style = {"fillColor": "#d32f2f", "color": "#b71c1c"}
        revisit_coverage_style = {"fillColor": "#88e66c", "color": "#88e66c"}
        if isinstance(self.data["bounds"].sample(1), str):
            self.data["bounds"] = self.data["bounds"].apply(eval)
        if len(self.data) > 1000:
            print(
                f"Coverage for {len(self.data)} revisits is being visualised. This may take a while."
            )
        for _, revisit in self.data.iterrows():
            revisit_coverage_polygon = Polygon(revisit["coordinates"])
            aoi_polygon = Polygon(
                AOIGenerator.bounds_to_bounding_box(*revisit["bounds"])
            )
            folium.GeoJson(
                revisit_coverage_polygon,
                style_function=lambda x: revisit_coverage_style,
            ).add_to(map)
            aoi_geojson = folium.GeoJson(
                aoi_polygon, style_function=lambda x: aoi_style
            )
            aoi_tooltip = folium.Tooltip(
                f"<strong>AOI:</strong> {revisit['name']} <br> <strong>IPCC:</strong> {revisit['IPCC']} <br> <strong>LCCS:</strong> {revisit['LCCS']} <br> <strong>SMOD:</strong> {revisit['SMOD']}"
            )
            aoi_tooltip.add_to(aoi_geojson)
            aoi_geojson.add_to(map)
        folium.LatLngPopup().add_to(map)
        return map

    def visualise_delta_days(self):
        """ Visualises the difference in days between revisits and the target date (delta days).

        Returns
        -------
        plotnine.ggplot
            The plot of the delta days of the revisits.
        """
        return self.data.plot

    def visualise_downloaded_revisits(self, revisits, save_gif=False):
        """ Visualises downloaded revisits as a gif or a set of images.

        Parameters
        ----------
        revisits : list
            The paths of downloaded revisits.
        save_gif : bool, optional
            Whether to save the visualisation to a gif and return it, or to return a list of images, by default False.
        """

        def get_revisit_rgb(revisit):
            image = tiff.imread(revisits[revisit])
            rgb = image[:, :, 1:4][:, :, ::-1]
            rgb = rgb / np.max(rgb)
            rgb = np.uint8(255 * rgb)
            return Image.fromarray(rgb).resize((500, 500), Image.BICUBIC)

        if save_gif:
            if not os.path.exists("data/visualisations"):
                os.makedirs("data/visualisations")
            images_scaled = [
                get_revisit_rgb(revisit) for revisit in range(len(revisits))
            ]
            imageio.mimsave(
                "data/visualisations/s2_revisits.gif",
                images_scaled,
                format="gif",
                fps=8,
            )
            return IPythonImage(url="data/visualisations/s2_revisits.gif")

        return interact(
            get_revisit_rgb,
            revisit=widgets.IntSlider(min=0, max=len(revisits) - 1, step=1, value=0),
        )

    def visualise_downloaded_SPOT_revisits(self, revisits, save_gif=False):
        """ Visualises downloaded SPOT revisits as a gif or a set of images.

        Parameters
        ----------
        revisits : list
            The paths of downloaded revisits.
        save_gif : bool, optional
            Whether to save the visualisation to a gif and return it, or to return a list of images, by default False.
        """

        def get_revisit_rgb(revisit):
            image = tiff.imread(revisits[revisit])
            rgb = image[:, :, 0:3]
            rgb = rgb / np.max(rgb)
            rgb = np.uint8(255 * rgb)
            return Image.fromarray(rgb).resize((500, 500), Image.BICUBIC)

        if save_gif:
            if not os.path.exists("data/visualisations"):
                os.makedirs("data/visualisations")
            images_scaled = [
                get_revisit_rgb(revisit) for revisit in range(len(revisits))
            ]
            imageio.mimsave(
                "data/visualisations/spot_revisits.gif",
                images_scaled,
                format="gif",
                fps=3,
            )
            return IPythonImage(url="data/visualisations/spot_revisits.gif")

        return interact(
            get_revisit_rgb,
            revisit=widgets.IntSlider(min=0, max=len(revisits) - 1, step=1, value=0),
        )

    def visualise_orders_on_map(self):
        """ Visualises orders created on SentinelHub on a world map.

        Returns
        -------
        plotnine.ggplot
            The plot of the orders on a world map.
        """
        map = folium.Map([0, 0], zoom_start=3, tiles="CartoDB dark_matter")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        ).add_to(map)
        aoi_style = {"fillColor": "#e74c3c", "color": "#c0392b"}
        for _, order in self.data.iterrows():
            order_bounds = order["input.bounds.bbox"]
            aoi_polygon = Polygon(AOIGenerator.bounds_to_bounding_box(*order_bounds))
            aoi_geojson = folium.GeoJson(
                aoi_polygon, style_function=lambda x: aoi_style
            )

            folium.Marker(AOIGenerator.bounds_to_poi(order_bounds)).add_to(map)
            aoi_tooltip = folium.Tooltip(
                f"<strong>Order:</strong> {order['name']} <br> <strong>Status:</strong> {order['status']} <br> <strong>Created:</strong> {order['created']} <br> <strong>Area:</strong> {order['sqkm']} km²."
            )
            aoi_tooltip.add_to(aoi_geojson)
            aoi_geojson.add_to(map)
        folium.LatLngPopup().add_to(map)
        return map
