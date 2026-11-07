"""Shared wiring: the Temporal client with the OpenAI Agents plugin, for the worker and the operator scripts."""

from __future__ import annotations

import os
from datetime import timedelta

from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

from provider import nim_model_provider

TASK_QUEUE = os.environ.get("TASK_QUEUE", "openshell-durable-approvals")


async def connect(*, register_model_activities: bool = False) -> Client:
    plugin = OpenAIAgentsPlugin(
        # Every model call is an activity: a 15 s heartbeat notices a killed worker quickly, and Temporal retries the call.
        model_params=ModelActivityParameters(start_to_close_timeout=timedelta(minutes=2), heartbeat_timeout=timedelta(seconds=15),
                                             retry_policy=RetryPolicy(maximum_attempts=5, initial_interval=timedelta(seconds=2), backoff_coefficient=2.0)),
        model_provider=nim_model_provider(),
        register_activities=register_model_activities,
    )
    kwargs: dict = {}
    if os.environ.get("TEMPORAL_API_KEY"):  # Temporal Cloud
        kwargs = {"api_key": os.environ["TEMPORAL_API_KEY"], "tls": True}
    return await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"), namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
                                plugins=[plugin], **kwargs)
