# weather_worker.py

import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflows.weather_workflows import GetAlerts, GetForecast
from activities.weather_activities import make_nws_request

async def main():
    # Connect to Temporal server (change address if using Temporal Cloud)
    client = await Client.connect("localhost:7233")

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
