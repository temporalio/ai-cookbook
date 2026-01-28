import os
from temporalio.plugin import SimplePlugin
from temporalio.converter import DataConverter

from .claim_check import ClaimCheckCodec


class ClaimCheckPlugin(SimplePlugin):
    """Temporal plugin that integrates the Claim Check codec with client configuration."""

    def __init__(self):
        """Initialize the plugin with S3 connection configuration."""
        super().__init__(
            name="claim-check",
            data_converter=DataConverter(
                payload_codec=ClaimCheckCodec(
                    bucket_name=os.getenv("S3_BUCKET_NAME", "temporal-claim-check"),
                    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                ),
            ),
        )
