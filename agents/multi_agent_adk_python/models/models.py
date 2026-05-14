from pydantic import BaseModel


class AssignmentInput(BaseModel):
    """A new order to be assigned to a driver."""

    order_id: str
    hotel: str
    priority: str
    servings: int
    deadline_minutes: int
    delivery_lat: float
    delivery_lng: float


class AssignmentOutput(BaseModel):
    """The dispatch decision returned by the agent pipeline."""

    driver_id: str
    reasoning_summary: str
