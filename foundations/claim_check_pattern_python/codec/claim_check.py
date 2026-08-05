import uuid
import logging
from typing import Iterable, List
import aioboto3
from botocore.exceptions import ClientError

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec

logger = logging.getLogger(__name__)


class ClaimCheckCodec(PayloadCodec):
    """PayloadCodec that implements the Claim Check pattern using S3 storage.
    
    This codec stores large payloads in S3 and replaces them with unique keys,
    allowing Temporal workflows to operate with lightweight references instead
    of large payload data.
    """

    def __init__(
        self,
        bucket_name: str = "temporal-claim-check",
        endpoint_url: str = None,
        region_name: str = "us-east-1",
        max_inline_bytes: int = 20 * 1024,
    ):
        """Initialize the claim check codec with S3 connection details.

        Args:
            bucket_name: S3 bucket name for storing claim check data
            endpoint_url: S3 endpoint URL (for MinIO or other S3-compatible services)
            region_name: AWS region name
            max_inline_bytes: Payloads up to this size will be left inline
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.max_inline_bytes = max_inline_bytes
        self.session = aioboto3.Session()

        self._bucket_created = False

    async def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, creating it if necessary."""
        if self._bucket_created:
            return
            
        async with self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        ) as s3_client:
            try:
                await s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['404', 'NoSuchBucket']:
                    try:
                        await s3_client.create_bucket(Bucket=self.bucket_name)
                    except ClientError as create_error:
                        # Handle bucket already exists race condition
                        if create_error.response['Error']['Code'] not in ['BucketAlreadyExists', 'BucketAlreadyOwnedByYou']:
                            raise create_error
                elif error_code not in ['403', 'Forbidden']:
                    raise e
        
        self._bucket_created = True

    async def encode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Replace large payloads with keys and store original data in S3.
        
        Args:
            payloads: Iterable of payloads to encode
            
        Returns:
            List of encoded payloads (keys for claim-checked payloads)
        """
        await self._ensure_bucket_exists()
        
        out: List[Payload] = []
        for payload in payloads:
            # Leave small payloads inline to improve debuggability and avoid unnecessary indirection
            data_size = len(payload.data or b"")
            if data_size <= self.max_inline_bytes:
                out.append(payload)
                continue

            encoded = await self.encode_payload(payload)
            out.append(encoded)
        return out

    async def decode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Retrieve original payloads from S3 using stored keys.
        
        Args:
            payloads: Iterable of payloads to decode
            
        Returns:
            List of decoded payloads (original data retrieved from S3)
            
        Raises:
            ValueError: If a claim check key is not found in S3
        """
        await self._ensure_bucket_exists()
        
        out: List[Payload] = []
        for payload in payloads:
            if payload.metadata.get("temporal.io/claim-check-codec", b"").decode() != "v1":
                # Not a claim-checked payload, pass through unchanged
                out.append(payload)
                continue

            s3_key = payload.data.decode("utf-8")
            stored_data = await self.get_payload_from_s3(s3_key)
            if stored_data is None:
                raise ValueError(f"Claim check key not found in S3: {s3_key}")
            
            original_payload = Payload.FromString(stored_data)
            out.append(original_payload)
        return out

    async def encode_payload(self, payload: Payload) -> Payload:
        """Store payload in S3 and return a key-based payload.
        
        Args:
            payload: Original payload to store
            
        Returns:
            Payload containing only the S3 key
        """
        await self._ensure_bucket_exists()
        
        key = str(uuid.uuid4())
        serialized_data = payload.SerializeToString()
        
        # Store the original payload data in S3
        async with self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        ) as s3_client:
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=serialized_data
            )
        
        # Return a lightweight payload containing only the key
        return Payload(
            metadata={
                "encoding": b"claim-checked",
                "temporal.io/claim-check-codec": b"v1",
            },
            data=key.encode("utf-8"),
        )

    async def get_payload_from_s3(self, s3_key: str) -> bytes:
        """Retrieve payload data from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Raw payload data bytes, or None if not found
        """
        try:
            async with self.session.client(
                's3',
                endpoint_url=self.endpoint_url,
                region_name=self.region_name
            ) as s3_client:
                response = await s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                return await response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise e