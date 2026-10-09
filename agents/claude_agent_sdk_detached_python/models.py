from dataclasses import dataclass


@dataclass
class RunRequest:
    prompt: str
    effect: str
    effect_delay_seconds: float = 0


@dataclass
class RunResult:
    response: str
    session_id: str
    activity_attempt: int
