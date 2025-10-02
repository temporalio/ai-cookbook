from temporalio import activity
from openai import AsyncOpenAI
from typing import Optional, List, cast, Any, TypeVar, Generic
from typing_extensions import Annotated
from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator
from pydantic.functional_serializers import PlainSerializer

import importlib

T = TypeVar("T", bound=BaseModel)


def _coerce_class(v: Any) -> type[Any]:
    """Pydantic validator: convert string path to class during deserialization."""
    if isinstance(v, str):
        mod_path, sep, qual = v.partition(":")
        if not sep:  # support "package.module.Class"
            mod_path, _, qual = v.rpartition(".")
        module = importlib.import_module(mod_path)
        obj = module
        for attr in qual.split("."):
            obj = getattr(obj, attr)
        return cast(type[Any], obj)
    elif isinstance(v, type):
        return v
    else:
        raise ValueError(f"Cannot coerce {v} to class")


def _dump_class(t: type[Any]) -> str:
    """Pydantic serializer: convert class to string path during serialization."""
    return f"{t.__module__}:{t.__qualname__}"


# Custom type that automatically handles class <-> string conversion in Pydantic serialization
ClassReference = Annotated[
    type[T],
    BeforeValidator(_coerce_class),
    PlainSerializer(_dump_class, return_type=str),
]


class InvokeModelRequest(BaseModel, Generic[T]):
    model: str
    instructions: str
    input: str
    response_format: Optional[ClassReference[T]] = None
    tools: Optional[List[dict]] = None


class InvokeModelResponse(BaseModel, Generic[T]):
    # response_format records the type of the response model
    response_format: Optional[ClassReference[T]] = None
    response_model: Any

    @property
    def response(self) -> T:
        """Reconstruct the original response type if response_format was provided."""
        if self.response_format:
            model_cls = self.response_format
            return model_cls.model_validate(self.response_model)
        return self.response_model


@activity.defn
async def invoke_model(request: InvokeModelRequest[T]) -> InvokeModelResponse[T]:
    client = AsyncOpenAI(max_retries=0)

    kwargs: dict[str, Any] = {
        "model": request.model,
        "instructions": request.instructions,
        "input": request.input,
    }

    if request.response_format:
        kwargs["text_format"] = request.response_format

    if request.tools:
        kwargs["tools"] = request.tools

    # Use responses API consistently
    resp = await client.responses.parse(**kwargs)

    if request.response_format:
        # Convert structured response to dict for managed serialization.
        # This allows us to reconstruct the original response type while maintaining type safety.
        parsed_model = cast(BaseModel, resp.output_parsed)
        return InvokeModelResponse(
            response_model=parsed_model.model_dump(),
            response_format=request.response_format,
        )
    else:
        return InvokeModelResponse(
            response_model=resp.output_text, response_format=None
        )
