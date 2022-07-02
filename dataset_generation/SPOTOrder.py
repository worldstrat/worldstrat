from sentinelhub import SentinelHubBYOC, ByocCollection
from dataset_generation.SentinelHub import SentinelHub
import pandas as pd


from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import json
import concurrent
from concurrent.futures import as_completed
import requests
from tqdm.auto import tqdm
from dataset_generation.Visualiser import Visualiser


class SPOTOrder:
    """ Creates SentinelHub orders for SPOT imagery. """

    def __init__(self, selected_spot_revisits, collection_name=None):
        """ Initialises the SPOTOrder class.

        Parameters
        ----------
        selected_spot_revisits : pandas.DataFrame
            A dataframe containing the selected SPOT imagery from the SPOTCatalogue.
        collection_name : str, optional
            The name of the collection to be used for the orders, by default None.
        """
        self.selected_spot_revisits = selected_spot_revisits
        self.check_collection(collection_name)
        print(f"Checking available quota:")
        self.check_quota()
        self.all_orders = None
        self.orders = None
        self.visualiser = Visualiser(self.orders)

    def get_all_orders(self):
        """ Gets all orders for the current collection.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing all orders for the current collection.
        """
        self.all_orders = SPOTOrder.get_all_orders_for_collection(self.collection_id)
        return self.all_orders

    def confirm_orders(self, actually_confirm_the_orders=False):
        """ Confirms all orders for the current collection.

        Parameters
        ----------
        actually_confirm_the_orders : bool, optional
            Safety variable: Whether to actually confirm the orders, by default False.

        Returns
        -------
        dict
            A dict containing the confirmed orders, as returned by confirm_order.
        """
        self.confirmations = SPOTOrder.confirm_all_orders(
            self.orders, actually_confirm_the_orders=actually_confirm_the_orders
        )
        return self.confirmations

    def create_orders(self):
        """Creates orders for all selected revisit POIs.

        Returns
        -------
        dict
            A dict containing the created orders, as returned by create_order_for_revisit.
        """
        orders = {}
        for revisit_index, revisit in tqdm(
            self.selected_spot_revisits.iterrows(),
            total=len(self.selected_spot_revisits),
            desc="Creating orders for selected SPOT revisits",
        ):
            order = self.create_order_for_revisit(revisit)
            orders[revisit_index] = order
        for order in orders:
            if orders[order]["status"] != "CREATED":
                raise ValueError(f"Order {order} was not created successfully.")
        self.orders = pd.concat(
            [pd.json_normalize(order) for _, order in orders.items()]
        )

    def create_order_for_revisit(self, revisit):
        """ Creates an order for a revisit.

        Parameters
        ----------
        revisit : pandas.Series
            A series containing the SPOT revisit information.

        Returns
        -------
        dict
            A dict containing the created order, as returned by create_order.
        """
        payload = SPOTOrder.create_order_payload(
            revisit["name"],
            revisit["bounds"],
            revisit["product_id"],
            self.collection_id,
        )
        return SPOTOrder.create_order(payload)

    def check_collection(self, collection_name):
        """ Checks if a collection with the given name exists.
        If it does not exist, it is created.
        If None is given, all available collections are listed, and instructions on how to create a new collection are given.

        Parameters
        ----------
        collection_name : str
            The name of the collection to be checked.

        Raises
        ------
        ValueError
            If the collection does not exist and cannot be created.
        """
        available_collections = SPOTOrder.get_collections()
        if collection_name is None:
            if len(available_collections) == 0:
                help_string = "There are no collections available for the SentinelHub account currently being used. You can provide a collection name and one will be created for you."
            if len(available_collections) > 0:
                collection_names = [
                    collection["name"] for collection in available_collections
                ]
                help_string = f"Please provide a collection name. The following collections are available: {collection_names}.\n You can also create a new collection using the 'SPOTOrder.create_collection' function."
            raise ValueError(
                f"""When ordering SPOT imagery from SentinelHub, a collection is used.
                The collection serves as a collection for your orders, both created and completed ones.
                {help_string}"""
            )
        if collection_name is not None:
            if len(available_collections) == 0:
                print(f"Creating a new collection '{collection_name}' for you.")
                self.collection_id = SPOTOrder.create_collection(collection_name)
            else:
                collection_names = [
                    collection["name"] for collection in available_collections
                ]
                if collection_name in collection_names:
                    self.collection_id = available_collections[
                        collection_names.index(collection_name)
                    ]["id"]
                else:
                    print(f"Creating a new collection '{collection_name}' for you.")
                    self.collection_id = SPOTOrder.create_collection(collection_name)

    def visualise_created_orders(self):
        """ Visualises the newly created orders for the current collection.

        Returns
        -------
        folium.Map
            A world map visualising the created orders.
        """
        if self.orders is None:
            return "Please create new orders first."
        self.visualiser.update_data(pd.DataFrame.from_dict(self.orders))
        return self.visualiser.visualise_orders_on_map()

    def visualise_all_orders(self):
        """ Visualises all orders in the current collection.

        Returns
        -------
        folium.Map
            A world map visualising all orders in the current collection.
        """
        if self.all_orders is None or len(self.all_orders) == 0:
            return "There are no orders available for the current collection to be visualised."
        self.visualiser.update_data(self.get_all_orders())
        return self.visualiser.visualise_orders_on_map()

    def create_collection(collection_name, s3_bucket="sh.tpdi.byoc.eu-central-1"):
        """Creates a SentinelHub collection with the given name.

        Parameters
        ----------
        collection_name : str
            The name of the collection to be created.
        s3_bucket : str, optional
            The S3 bucket to store the collection, by default 'sh.tpdi.byoc.eu-central-1'.

        Returns
        -------
        str
            The ID of the created collection.
        """
        byoc = SentinelHubBYOC(config=SentinelHub.get_config())
        new_collection = ByocCollection(name=collection_name, s3_bucket=s3_bucket)
        created_collection = byoc.create_collection(new_collection)
        return created_collection["id"]

    @staticmethod
    def get_collections():
        """Gets all SentinelHub collections.

        Returns
        -------
        list of dicts
            List of all available collections - {'id', 'name', 's3Bucket'}.
        """
        byoc = SentinelHubBYOC(config=SentinelHub.get_config())
        return list(byoc.iter_collections())

    @staticmethod
    def get_collection(collection_id):
        """Gets a SentinelHub collection by its ID.

        Parameters
        ----------
        collection_id : str
            The ID of the collection to be retrieved.

        Returns
        -------
        dict
            The collection, as returned by the SentinelHub API or "Collection not found" if the collection was not found.
        """
        byoc = SentinelHubBYOC(config=SentinelHub.get_config())
        try:
            return byoc.get_collection(collection_id)
        except:
            return "Collection not found"

    @staticmethod
    def rename_collection(collection_id, new_name):
        """Renames a SentinelHub collection.

        Parameters
        ----------
        collection_id : str
            The ID of the collection to be renamed.
        new_name : str
            The new name of the collection.

        Returns
        -------
        str
            The result of the renaming: "Collection renamed" or "Collection not renamed".
        """
        try:
            byoc = SentinelHubBYOC(config=SentinelHub.get_config())
            collection = SPOTOrder.get_collection(collection_id)
            collection["name"] = new_name
            byoc.update_collection(collection)
            return "Collection renamed"
        except:
            return "Collection not renamed"

    @staticmethod
    def delete_collection(collection_id):
        """Deletes a SentinelHub collection.

        Parameters
        ----------
        collection_id : str
            The ID of the collection to be deleted.

        Returns
        -------
        str
            The result of the deletion: "Collection deleted" or "Collection not deleted".
        """
        byoc = SentinelHubBYOC(config=SentinelHub.get_config())
        try:
            byoc.delete_collection(SPOTOrder.get_collection(collection_id))
            return "Collection deleted"
        except Exception as e:
            print(e)
            return "Collection not deleted"

    @staticmethod
    def check_quota():
        """Checks the SentinelHub quota usage.

        Returns
        -------
        list of dicts
            List of all available quotas - {'id', 'collectionId', 'quotaSqkm',  'quotaUsed', 'datasetId'}.
        """
        oauth_session = SentinelHub.generate_oauth_session()
        url = "https://services.sentinel-hub.com/api/v1/dataimport/quotas"

        response = oauth_session.get(url=url)
        response.raise_for_status()

        response = response.json()["data"]
        for provider in response:
            print(
                f"{provider['collectionId']}: {provider['quotaUsed']}/{provider['quotaSqkm']} km² used.\n{round(provider['quotaSqkm'] - provider['quotaUsed'],2)} km² remaining.\n"
            )

    @staticmethod
    def create_order_payload(order_name, bounds, product_id, collection_id):
        """ Creates the payload for a new order.

        Parameters
        ----------
        order_name : str
            The name of the order.
        bounds : list of floats
            The bounds of the order.
        product_id : str
            The ID of the SPOT product to be ordered (as returned by the SPOTCatalogue).
        collection_id : str
            The ID of the collection under which the order will be created.

        Returns
        -------
        dict
            The payload for the new order.
        """
        if type(bounds) is dict:
            bbox = [
                bounds["lon_min"],
                bounds["lat_min"],
                bounds["lon_max"],
                bounds["lat_max"],
            ]
        else:
            bbox = bounds
        payload = {
            "name": order_name,
            "collectionId": collection_id,
            "input": {
                "provider": "AIRBUS",
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{"constellation": "SPOT", "products": [{"id": product_id}]}],
            },
        }
        return payload

    @staticmethod
    def create_order(payload):
        """Create an order on SentinelHub for the provided AOI/bounds, imagery product and collection id.

        Parameters
        ----------
        order_name : string
            The name under which the order will be created/saved.
        bounds : dict
            Dictionary containing the AOI/bounds, defined by the min/max latitude and longitude.
        product_id : string
            Imagery product_id, as returned by search_SPOT.
        collection_id : string
            The SentinelHub collection under which the orders will be stored.

        Returns
        -------
        dict
            SentinelHub API response for the created order.
        """
        order_url = "https://services.sentinel-hub.com/api/v1/dataimport/orders"

        headers = {"Content-Type": "application/json"}
        sentinelhub_session = SentinelHub.generate_sentinel_hub_session()
        headers.update(sentinelhub_session.session_headers)

        s = requests.Session()
        # TODO: Warning, allowing POST as a retried method could cause multiple inserts.
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        # Any request that has an URL with this prefix (https://) will use this adapter
        s.mount("https://", HTTPAdapter(max_retries=retries))
        try:
            response = s.post(order_url, data=json.dumps(payload), headers=headers)
        except Exception as e:
            print(f"ERROR: {e}")
            error_response = {}
            error_response["name"] = payload["name"]
            error_response["error_message"] = e
            return error_response

        if response.status_code == 400:
            response = response.json()
            error_message = json.loads(response["error"]["message"])["message"]
            error_response = {}
            error_response["name"] = payload["name"]
            error_response["error_code"] = int(response["error"]["status"])
            error_response["error_message"] = error_message

            print(f"Error for {payload['name']}:")
            print(error_message)

            return error_response
        else:
            response.raise_for_status()
        return response.json()

    @staticmethod
    def confirm_all_orders(orders, actually_confirm_the_orders=False):
        """Confirms all orders in the parameter dict.

        Parameters
        ----------
        orders : dict
            A dict of created orders, as returned by create_orders_for_list_of_pois.
        actually_confirm_the_order : bool, optional
            Safety variable that stops any order confirmation unless set to True, by default False.

        Returns
        -------
        dict
            Dictionary containing the SentinelHub API response for each POI/AOI, as returned by confirm_order.
        """
        confirmations = pd.DataFrame()
        if not actually_confirm_the_orders:
            print("Safety variable set to False. No orders will be confirmed.")

        for poi_index, order in tqdm(
            orders.iterrows(), total=len(orders), desc="Confirming orders"
        ):
            confirmation = pd.json_normalize(
                SPOTOrder.confirm_order(order, actually_confirm_the_orders)
            )
            confirmation["poi_index"] = poi_index
            confirmations = pd.concat([confirmations, confirmation])
        confirmations.set_index("poi_index", inplace=True)
        return confirmations

    @staticmethod
    def confirm_order(order, actually_confirm_the_order=False):
        """Confirms a created order on SentinelHub.

        Parameters
        ----------
        order : dict
            Dict containing the created order, as returned by create_order.
        actually_confirm_the_order : bool, optional
            Safety variable that stops any order confirmation unless set to True, by default False.

        Returns
        -------
        dict or string
            Dictionary containing the SentinelHub API response for the confirmed order,
            or a string warning that the safety variable is set to False.
        """
        oauth_session = SentinelHub.generate_oauth_session()
        url = "https://services.sentinel-hub.com/api/v1/dataimport/orders/{}/confirm".format(
            order["id"]
        )
        if actually_confirm_the_order:
            response = oauth_session.post(url)
            return response
        return {"order_id": order["id"]}

    @staticmethod
    def check_all_orders_status(orders):
        """Checks the status of all orders in the passed dictionary.

        Parameters
        ----------
        orders : dict
            Dictionary of confirmed orders, as returned by confirm_all_orders.

        Returns
        -------
        string
            String detailing if any and which orders are not yet done.
        """
        order_statuses = pd.DataFrame("poi_index", "order_id", "status")
        for poi_index, order in orders.iterrows():
            status = SPOTOrder.check_order_status(order)
            order_statuses.append([poi_index, order["id"], status])
        if (
            len(order_statuses[order_statuses["status"].isin(["RUNNING", "CREATED"])])
            == 0
        ):
            print("All orders completed.")
        return order_statuses

    @staticmethod
    def check_order_status(order):
        """Checks the status of an order.

        Parameters
        ----------
        order : dict
            A created or confirmed order, as returned by create_order or confirm_order.

        Returns
        -------
        string
            The status of the order: 'CREATED', 'RUNNING', 'DONE'.
        """
        oauth_session = SentinelHub.generate_oauth_session()
        url = (
            f"https://services.sentinel-hub.com/api/v1/dataimport/orders/{order['id']}"
        )

        response = oauth_session.get(url)
        response.raise_for_status()

        order = response.json()
        return order["status"]

    @staticmethod
    def get_orders_from_collection(collection_id, token=None, status=None, count=100):
        """Gets all orders from a collection.
        If a token is provided, it will be used to get the next orders.
        If a status is provided, it will be used to get orders with that status.
        If a count is provided, it will be used to get that number of orders.


        Parameters
        ----------
        collection_id : string
            The SentinelHub collection id.
        token : str, optional
            The token to use to get the next orders, if not provided, the first N orders (N=count) will be returned.
        status : str, optional
            The status of the orders to get, e.g. 'CREATED', 'RUNNING', 'DONE'. If not provided, all orders will be returned.
        count : int, optional
            The number of orders to get, by default 100.

        Returns
        -------
        dict
            Dictionary containing the fetched orders.
        """
        oauth_session = SentinelHub.generate_oauth_session()
        url = f"https://services.sentinel-hub.com/api/v1/dataimport/orders?collectionId={collection_id}"
        if token != None:
            url += f"&viewtoken={token}"
        if status != None:
            url += f"&status={status}"
        if count != None:
            url += f"&count={count}"
        response = oauth_session.get(url)
        response.raise_for_status()

        return response.json()

    @staticmethod
    def get_all_orders_for_collection(collection_id, status=None):
        """Obtains all orders of the given status from a given collection.

        Parameters
        ----------
        collection_id : string, optional
            The SentinelHub collection to query.
        status : str, optional
            The wanted order status, by default 'DONE'.

        Returns
        -------
        list of dicts
            All of the orders inside the given collection with the given status.
        """
        orders = pd.DataFrame()
        response = SPOTOrder.get_orders_from_collection(collection_id, status=status)
        orders = pd.concat([orders, pd.json_normalize(response["data"])])
        while "nextToken" in response["links"]:
            response = SPOTOrder.get_orders_from_collection(
                collection_id, response["links"]["nextToken"]
            )
            orders = pd.concat([orders, pd.json_normalize(response["data"])])
        return orders

    @staticmethod
    def get_order(order_id):
        """Gets an order from the SentinelHub API by its id.

        Parameters
        ----------
        order_id : string
            The id of the order to get.

        Returns
        -------
        pandas.DataFrame
            The order as a DataFrame.
        """
        oauth_session = SentinelHub.generate_oauth_session
        url = f"https://services.sentinel-hub.com/api/v1/dataimport/orders/{order_id}"

        response = oauth_session.get(url)
        return pd.json_normalize(response.json())

    @staticmethod
    def delete_order(order_id):
        """Deletes an order from the SentinelHub API by its id.

        Parameters
        ----------
        order_id : string
            The id of the order to delete.

        Returns
        -------
        string
            The status of the order: 'CREATED', 'RUNNING', 'DONE'.
        """
        oauth_session = SentinelHub.generate_oauth_session
        url = f"https://services.sentinel-hub.com/api/v1/dataimport/orders/{order_id}"
        response = oauth_session.delete(url)
        return True if response.status_code == 204 else False

    @staticmethod
    def get_delivery(order_id):
        """Gets the delivery of an order from the SentinelHub API by its id.

        Parameters
        ----------
        order_id : string
            The id of the order to get the delivery of.

        Returns
        -------
        dict
            The delivery of the order.
        """
        oauth_session = SentinelHub.generate_oauth_session
        url = f"https://services.sentinel-hub.com/api/v1/dataimport/orders/{order_id}/deliveries"
        response = oauth_session.get(url=url)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_tiles_from_collection(collection_id, token=None, status=None, count=100):
        """Gets all tiles from a collection.

        Parameters
        ----------
        collection_id : string
            The SentinelHub collection id.
        token : str, optional
            The token to use to get the next tiles, if not provided, the first N tiles (N=count) will be returned.
        status : str, optional
            The status of the tiles to get, e.g. 'CREATED', 'RUNNING', 'DONE'. If not provided, all tiles will be returned.
        count : int, optional
            The number of tiles to get, by default 100.

        Returns
        -------
        dict
            Dictionary containing the fetched tiles.
        """
        oauth_session = SentinelHub.generate_oauth_session
        url = f"https://services.sentinel-hub.com/api/v1/byoc/collections/{collection_id}/tiles?count={count}"
        if token != None:
            url += f"&viewtoken={token}"
        if status != None:
            url += f"&status={status}"
        response = oauth_session.get(url)
        response.raise_for_status()

        return response.json()

    @staticmethod
    def get_all_tiles(collection_id):
        """Obtains all tiles from the given collection.

        Parameters
        ----------
        collection_id : string, optional
            The id of the SentinelHub collection to query.

        Returns
        -------
        pandas.DataFrame
            All of the tiles inside the given collection.
        """
        tiles = pd.DataFrame()
        response = SPOTOrder.get_tiles_from_collection(collection_id)
        tiles = pd.concat([tiles, pd.json_normalize(response["data"])])
        while "nextToken" in response["links"]:
            response = SPOTOrder.get_tiles_from_collection(
                collection_id, response["links"]["nextToken"]
            )
            tiles = pd.concat([tiles, pd.json_normalize(response["data"])])
        return tiles

    @staticmethod
    def save_metadata_for_orders(collection_id, orders, revisits):
        """Saves the metadata for the given orders and their revisits under the given collection id as a csv file.

        Parameters
        ----------
        collection_id : string
            The id of the SentinelHub collection to save the metadata for.
        orders : pandas.DataFrame
            The orders to save the metadata for.
        revisits : pandas.DataFrame
            The dataframe containing the metadata for the revisits of the given orders.
        """
        metadata = pd.merge(orders, revisits, on="poi_index")
        metadata.to_csv(f"{collection_id}-metadata.csv", index=False)

    @staticmethod
    def get_collection_metadata(collection_id):
        """ Gets the metadata for the given collection.

        Parameters
        ----------
        collection_id : string
            The id of the SentinelHub collection to get the metadata for.

        Returns
        -------
        pandas.DataFrame
            The metadata for the given collection.
        """
        collection_tiles = SPOTOrder.get_all_tiles(collection_id)
        if len(collection_tiles) == 0:
            return "Empty collection."
        collection_tiles["id"] = collection_tiles["path"].apply(
            lambda x: x.split("/")[0]
        )  # Get the order_id from the tile path
        collection_orders = SPOTOrder.get_all_orders(collection_id)
        column_remapping = {
            "created_x": "order_created",
            "status_x": "order_status",
            "created_y": "tile_created",
            "status_y": "tile_status",
            "input.bounds.bbox": "bounds",
        }  # Rename duplicate columns
        return collection_orders.merge(collection_tiles, on="id").rename(
            columns=column_remapping
        )

    @staticmethod
    def get_all_collections_metadata(save_path=None):
        """ Gets the metadata for all collections.

        Parameters
        ----------
        save_path : string, optional
            The path to save the metadata to, by default None.

        Returns
        -------
        pandas.DataFrame
            The metadata for all collections.
        """
        collections = SPOTOrder.get_collections()
        collections_metadata = {}
        for collection in collections:
            collections_metadata[
                collection["name"]
            ] = SPOTOrder.get_collection_metadata(collection["id"])
        if save_path != None:
            for collection_name, collection_metadata in collections_metadata.items():
                if type(collection_metadata) == type(pd.DataFrame()):
                    collection_metadata.to_csv(
                        f"{save_path}/{collection_name}-metadata.csv", index=False
                    )
        return collections_metadata
