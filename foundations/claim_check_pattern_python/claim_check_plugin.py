import os
from temporalio.client import Plugin, ClientConfig
from temporalio.converter import DataConverter
from temporalio.service import ConnectConfig, ServiceClient

from claim_check_codec import ClaimCheckCodec


class ClaimCheckPlugin(Plugin):
    """Temporal plugin that integrates the Claim Check codec with client configuration."""

    def __init__(self):
        """Initialize the plugin with S3 connection configuration."""
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "temporal-claim-check")
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self._next_plugin = None

    def init_client_plugin(self, next_plugin: Plugin) -> None:
        """Initialize this plugin in the client plugin chain."""
        self._next_plugin = next_plugin

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        """Apply the claim check configuration to the client.
        
        Args:
            config: Temporal client configuration
            
        Returns:
            Updated client configuration with claim check data converter
        """
        # Configure the data converter with claim check codec
        default_converter_class = config["data_converter"].payload_converter_class
        claim_check_codec = ClaimCheckCodec(
            bucket_name=self.bucket_name,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        )
        
        config["data_converter"] = DataConverter(
            payload_converter_class=default_converter_class,
            payload_codec=claim_check_codec
        )
        
        # Delegate to next plugin if it exists
        if self._next_plugin:
            return self._next_plugin.configure_client(config)
        return config

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        """Connect to the Temporal service.
        
        Args:
            config: Service connection configuration
            
        Returns:
            Connected service client
        """
        # Delegate to next plugin if it exists
        if self._next_plugin:
            return await self._next_plugin.connect_service_client(config)
        
        # If no next plugin, use default connection
        from temporalio.service import ServiceClient
        return await ServiceClient.connect(config)
