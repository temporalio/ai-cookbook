import asyncio
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from workflows.weather_workflows import GetAlerts, GetForecast
from activities.weather_activities import make_nws_request

async def main():
    # Connect to Temporal server 
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
