import uuid
import redis.asyncio as redis
from typing import Iterable, List

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec


class ClaimCheckCodec(PayloadCodec):
    """PayloadCodec that implements the Claim Check pattern using Redis storage.
    
    This codec stores large payloads in Redis and replaces them with unique keys,
    allowing Temporal workflows to operate with lightweight references instead
    of large payload data.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize the claim check codec with Redis connection details.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)

    async def encode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Replace large payloads with keys and store original data in Redis.
        
        Args:
            payloads: Iterable of payloads to encode
            
        Returns:
            List of encoded payloads (keys for claim-checked payloads)
        """
        out: List[Payload] = []
        for payload in payloads:
            encoded = await self.encode_payload(payload)
            out.append(encoded)
        return out

    async def decode(self, payloads: Iterable[Payload]) -> List[Payload]:
        """Retrieve original payloads from Redis using stored keys.
        
        Args:
            payloads: Iterable of payloads to decode
            
        Returns:
            List of decoded payloads (original data retrieved from Redis)
            
        Raises:
            ValueError: If a claim check key is not found in Redis
        """
        out: List[Payload] = []
        for payload in payloads:
            if payload.metadata.get("temporal.io/claim-check-codec", b"").decode() != "v1":
                # Not a claim-checked payload, pass through unchanged
                out.append(payload)
                continue

            redis_key = payload.data.decode("utf-8")
            stored_data = await self.redis_client.get(redis_key)
            if stored_data is None:
                raise ValueError(f"Claim check key not found in Redis: {redis_key}")
            
            original_payload = Payload.FromString(stored_data)
            out.append(original_payload)
        return out

    async def encode_payload(self, payload: Payload) -> Payload:
        """Store payload in Redis and return a key-based payload.
        
        Args:
            payload: Original payload to store
            
        Returns:
            Payload containing only the Redis key
        """
        key = str(uuid.uuid4())
        serialized_data = payload.SerializeToString()
        
        # Store the original payload data in Redis
        await self.redis_client.set(key, serialized_data)
        
        # Return a lightweight payload containing only the key
        return Payload(
            metadata={
                "encoding": b"claim-checked",
                "temporal.io/claim-check-codec": b"v1",
            },
            data=key.encode("utf-8"),
        )
