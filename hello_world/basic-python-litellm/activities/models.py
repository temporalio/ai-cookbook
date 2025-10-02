from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


@dataclass
class LiteLLMRequest:
    """Lightweight container for the handful of LiteLLM options this sample surfaces."""

    model: str
    messages: List[Dict[str, Any]]

    # Optional knobs: limited to the most common tweaks.
    temperature: Optional[float] = None              # Controls response creativity without extra ceremony.
    max_tokens: Optional[int] = None                 # Caps response length/cost.
    timeout: Optional[Union[float, int]] = None      # Lets callers bound slow provider responses.
    response_format: Optional[Union[dict, Type[Any]]] = None  # Hook for JSON/object style responses.

    # Escape hatch for advanced parameters we do not model explicitly.
    extra_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_acompletion_kwargs(self) -> Dict[str, Any]:
        """Convert this request to kwargs suitable for litellm.acompletion()."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
        }
        
        optional_values = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "response_format": self.response_format,
        }

        for key, value in optional_values.items():
            if value is not None:
                kwargs[key] = value

        if self.extra_options:
            kwargs.update(self.extra_options)
        
        return kwargs
