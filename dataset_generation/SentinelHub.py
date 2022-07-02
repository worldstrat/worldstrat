from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from sentinelhub import SHConfig, SentinelHubSession, SentinelHubCatalog


class SentinelHub:
    """ Wrapper class for the SentinelHub API. """

    @staticmethod
    def get_config():
        """ Returns the configuration for the SentinelHub API.

        Returns
        -------
        sentinelhub.SHConfig
            The configuration for the SentinelHub API.

        Raises
        ------
        EnvironmentError
            If the credentials (client ID and secret) are not set.
        """
        config = SHConfig()
        if None in [config.sh_client_id, config.sh_client_secret]:
            raise EnvironmentError(
                "Warning! To use Sentinel Hub Catalog API, please set the credentials (client ID and client secret) using sentinelhub.config."
            )
        return config

    @staticmethod
    def generate_sentinel_hub_session(client_id=None, client_secret=None):
        """Generates a SentinelHubSession used to send requests to SentinelHub.

        Returns
        -------
        sentinelhub.SentinelHubSession
            The generated used SentinelHubSession.

        Raises
        ------
        EnvironmentError
            If the credentials (client ID and client secret) are not provided.
        """

        sentinel_hub_session = SentinelHubSession(SentinelHub.get_config())
        return sentinel_hub_session

    @staticmethod
    def get_sentinel_hub_catalog():
        """Generates a SentinelHubCatalog used to search a SentinelHub DataCollection for products.

        Returns
        -------
        sentinelhub.SentinelHubCatalog
            The generated SentinelHubCatalog.

        Raises
        ------
        EnvironmentError
            If the credentials (client ID and client secret) are not provided.
        """
        return SentinelHubCatalog(config=SentinelHub.get_config())

    @staticmethod
    def get_sentinel_hub_config():
        """Generates a SentinelHubConfig used to get the credentials (client ID and client secret) used to send requests to SentinelHub.

        Returns
        -------
        sentinelhub.SentinelHubConfig
            The generated SentinelHubConfig.
        """
        return SHConfig()

    @staticmethod
    def generate_oauth_session():
        """Generates an OAuth2 session used to send requests to SentinelHub.

        Parameters
        ----------
        refresh : bool, optional
            Overwrite/renew the current session, by default False.

        Returns
        -------
        requests_oauthlib.OAuth2Session
            The generated/currently used OAuth2Session.
        """
        sentinel_hub_config = SentinelHub.get_config()
        # Your client credentials
        client_id = sentinel_hub_config.sh_client_id
        client_secret = sentinel_hub_config.sh_client_secret

        # Create a session
        # NOTE: there might be a more elegant way using SentinelHubSession, but for now
        # I am just adapting the example from the SentinelHub documentation
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)

        # Get token for the session
        token = oauth.fetch_token(
            token_url="https://services.sentinel-hub.com/oauth/token",
            client_id=client_id,
            client_secret=client_secret,
        )

        # All requests using this session will have an access token automatically added
        resp = oauth.get("https://services.sentinel-hub.com/oauth/tokeninfo")

        return oauth
