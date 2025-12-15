<!--
description: A durable MCP server that uses Temporal workflows for reliable execution of weather tools.
tags: [mcp, python, workflows]
priority: 775
-->
# Durable MCP Weather Server

This example demonstrates how to build a durable MCP (Model Context Protocol) server using Temporal Workflows for Durable Execution. The server exposes weather tools that fetch alerts and forecasts from the National Weather Service API.

MCP tools are "actions" that the MCP server can perform. Within a given MCP tool, there are often multiple steps (API calls, functions, etc.) that must happen in a certain order to complete an action. For example, the `get_forecast` tool performs the following steps:
- Call the National Weather Service API to find which region corresponds to the given latitude and longitude coordinates
- Call the National Weather Service API again to retrieve the forecast for that region
- Format and return the response to the user

In this one tool alone, we are taking several steps to complete a given action. We implement these steps in a Temporal Workflow, which ensures durability out-of-the-box. This means that whenever your MCP tool is called, it kicks off the Temporal Workflow, and every step (API call, function) is executed reliably and all the way to completion.

We use [FastMCP](https://github.com/jlowin/fastmcp) to implement the MCP Server and create tools using the decorator `@mcp.tool`.

> [!NOTE]
> External API calls are made within Temporal Activities. This ensures that network requests are retried appropriately and failures are handled gracefully.

This recipe highlights the following key design decisions:
- **Separation of concerns**: MCP tools act as thin wrappers that start Temporal Workflows. All business logic lives in workflows, ensuring durability and reliability.
- **Durable Execution**: By moving multi-step operations into Temporal Workflows, we guarantee that operations complete even in the face of failures, network issues, or process restarts.
- **Activity-based external calls**: All external API calls (like NWS API requests) are made within Temporal Activities, which provides automatic retries and proper error handling.
- **Retry policies**: Workflows use configurable retry policies to handle transient failures gracefully.

Also see this foundational [recipe for basic tool calling](https://docs.temporal.io/ai-cookbook/tool-calling-python) using the same weather tools.

## Application Components

This example includes the following components:
- The [MCP server](#create-the-mcp-server) (mcp_server.py) that exposes tools via FastMCP and starts Temporal workflows
- The [Workflows](#create-the-workflows) (weather_workflows.py) that orchestrate the multi-step weather operations
- The [Activity](#create-the-activity) (weather_activities.py) for making external API calls to the National Weather Service
- The [Worker](#create-the-worker) (worker.py) (that manages the Workflows and Activities)
- [Config for Claude Desktop](#configure-claude-desktop) (claude_desktop_config.json) for connecting the MCP server to Claude Desktop

## Create the MCP Server

The MCP server is implemented using FastMCP and exposes tools via the `@mcp.tool` decorator. Each tool is a thin wrapper that starts a Temporal Workflow and waits for the result. This design ensures that all business logic lives in durable Workflows.

*File: mcp_servers/weather.py*

```python
from temporalio.client import Client
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Temporal client setup (do this once, then reuse)
temporal_client = None

async def get_temporal_client():
    global temporal_client
    if not temporal_client:
        config = ClientConfig.load_client_connect_config()
        config.setdefault("target_host", "localhost:7233")
        temporal_client = await Client.connect(**config)
    return temporal_client

@mcp.tool
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    # The business logic has been moved into the Temporal Workflow, the MCP tool kicks off the Workflow
    client = await get_temporal_client()
    handle = await client.start_workflow(
        "GetAlerts",
        state,
        id=f"alerts-{state.lower()}",
        task_queue="weather-task-queue"
    )
    return await handle.result()

@mcp.tool
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a US location.

    Args:
        latitude: Latitude of the location (must be within the US)
        longitude: Longitude of the location (must be within the US)
    """
    # The business logic has been moved into the Temporal Workflow, the MCP tool kicks off the Workflow
    client = await get_temporal_client()
    handle = await client.start_workflow(
        workflow="GetForecast",
        args=[latitude, longitude],
        id=f"forecast-{latitude}-{longitude}",
        task_queue="weather-task-queue",
    )
    return await handle.result()

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
```

## Create the Workflows

The Workflows contain the business logic for fetching weather data. They orchestrate multiple steps, including API calls and data formatting. By implementing this logic in workflows, we ensure that operations complete reliably even if there are failures or interruptions.

### GetAlerts Workflow

The `GetAlerts` workflow fetches active weather alerts for a US state.

*File: workflows/weather_workflows.py*

```python
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

retry_policy = RetryPolicy(
    maximum_attempts=0,  # Infinite retries
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(minutes=1),
    backoff_coefficient=1.0,
)

# Constants
NWS_API_BASE = "https://api.weather.gov"

# Import Activities, passing them through the sandbox
with workflow.unsafe.imports_passed_through():
    from activities.weather_activities import make_nws_request

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@workflow.defn
class GetAlerts:
    @workflow.run
    async def get_alerts(self, state: str) -> str:
        """Get weather alerts for a US state.

        Args:
            state: Two-letter US state code (e.g. CA, NY)
        """
        url = f"{NWS_API_BASE}/alerts/active/area/{state}"
        data = await workflow.execute_activity(
            make_nws_request,
            url,
            schedule_to_close_timeout=timedelta(seconds=40),
            retry_policy=retry_policy,
        )

        if not data or "features" not in data:
            return "Unable to fetch alerts or no alerts found."

        if not data["features"]:
            return "No active alerts for this state."

        alerts = [format_alert(feature) for feature in data["features"]]
        return "\n---\n".join(alerts)
```

### GetForecast Workflow

The `GetForecast` workflow demonstrates a multi-step operation: it first fetches the forecast grid endpoint for a location, then uses that information to fetch the detailed forecast. 

*File: workflows/weather_workflows.py*

```python
@workflow.defn
class GetForecast:
    @workflow.run
    async def get_forecast(self, latitude: float, longitude: float) -> str:
        """Get weather forecast for a US location.

        Args:
            latitude: Latitude of the location (must be within the US)
            longitude: Longitude of the location (must be within the US)
        """
        # First get the forecast grid endpoint
        points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
        points_data = await workflow.execute_activity(
            make_nws_request,
            points_url,
            schedule_to_close_timeout=timedelta(seconds=40),
            retry_policy=retry_policy,
        )

        if not points_data:
            return "Unable to fetch forecast data for this location."

        # Get the forecast URL from the points response
        forecast_url = points_data["properties"]["forecast"]
        forecast_data = await workflow.execute_activity(
            make_nws_request,
            forecast_url,
            schedule_to_close_timeout=timedelta(seconds=40),
            retry_policy=retry_policy,
        )
        if not forecast_data:
            return "Unable to fetch detailed forecast."

        # Format the periods into a readable forecast
        periods = forecast_data["properties"]["periods"]
        forecasts = []
        for period in periods[:5]:  # Only show next 5 periods
            forecast = f"""
    {period['name']}:
    Temperature: {period['temperature']}°{period['temperatureUnit']}
    Wind: {period['windSpeed']} {period['windDirection']}
    Forecast: {period['detailedForecast']}
    """
            forecasts.append(forecast)

        return "\n---\n".join(forecasts)
```

## Create the Activity

We create an Activity for making HTTP requests to the National Weather Service API. All external API calls happen within Activities, which provides automatic retries and proper error handling through Temporal's retry mechanisms.

*File: activities/weather_activities.py*

```python
from typing import Any
from temporalio import activity
import httpx

USER_AGENT = "weather-app/1.0"

# External calls happen via Activities
@activity.defn
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return response.json()
```

## Create the Worker

The Worker is the process that excutes Activities and Workflows. 

*File: worker.py*

```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflows.weather_workflows import GetAlerts, GetForecast
from activities.weather_activities import make_nws_request

async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
        data_converter=pydantic_data_converter,
    )

    # Register both Workflows and the Activity 
    worker = Worker(
        client,
        task_queue="weather-task-queue",
        workflows=[GetAlerts, GetForecast],
        activities=[make_nws_request],
    )
    print("Worker started. Listening for workflows...")
    await worker.run()

# Start worker with both Workflows and Activities
if __name__ == "__main__":
    asyncio.run(main())
```

## Configure Claude Desktop

For this example, we are using Claude Desktop as the MCP Client. To use this MCP server with Claude Desktop, you need to configure it in your Claude Desktop configuration file. The config file tells Claude Desktop how to start the MCP server.

*File: claude_desktop_config.json*

```json
{
    "mcpServers": {
        "weather": {
        "command": "uv",
        "args": [
            "--directory",
            "<full path to the directory containing the weather.py>",
            "run",
            "mcp_servers/weather.py"
        ]
        }
    }
}
```

Replace `<full path to the directory containing the weather.py>` with the absolute path to the `hello_world_durable_mcp_server` directory.

## Configuration

This recipe uses Temporal's environment configuration system to connect to Temporal. By default, it connects to a local Temporal server. To use Temporal Cloud:

1. Set the `TEMPORAL_PROFILE` environment variable to use the cloud profile:
   ```bash
   export TEMPORAL_PROFILE=cloud
   ```

2. Configure the cloud profile using the Temporal CLI:
   ```bash
   temporal config set --profile cloud --prop address --value "<your temporal cloud endpoint>"
   temporal config set --profile cloud --prop namespace --value "<your temporal cloud namespace>"
   temporal config set --profile cloud --prop api_key --value "<your temporal cloud api key>"
   ```

   For TLS certificate authentication instead of API key, refer to the [Temporal environment configuration documentation](https://docs.temporal.io/develop/environment-configuration) for details.

## Running the MCP Server

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Start a Temporal server:
   ```bash
   # Using Temporal CLI
   temporal server start-dev
   ```

3. Start the worker in one terminal:
   ```bash
   uv run python worker.py
   ```

4. Configure Claude Desktop by adding the configuration from `claude_desktop_config.json` to your Claude Desktop config file (typically located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS).

5. Restart Claude Desktop to load the MCP server.

Once configured, you should see the tool appear under the slider icon underneath the Claude Desktop chat input box.

You can now ask Claude something like `What is the weather like in San Francisco, CA?`. Claude Desktop will understand that it needs to use the `get_forecast` tool in the Weather MCP server that you just configured.

> [!NOTE]
> The National Weather Service API only supports US locations. Asking about weather in non-US locations (e.g., "What is the weather in London?") will result in a 404 error from the API. 

After tool execution, Claude Desktop will send the result over to the LLM (with other context) for human formating, and then returns that result to the user. You can see these and other MCP-related actions in the `mcp_server.log`.