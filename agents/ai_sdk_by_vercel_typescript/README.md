<!--
description: Build a durable AI agent with the AI SDK by Vercel and Temporal that chooses tools to answer user questions.
tags: [agents, typescript, openai]
priority: 750
-->

# Durable agent with tools using the AI SDK by Vercel

In this example, we show you how to build a durable agent using the [AI SDK by Vercel](https://docs.temporal.io/develop/typescript/integrations/ai-sdk#provide-your-durable-agent-with-tools). The agent calls tools backed by Temporal Activities to answer user questions, and it can determine which tools to use based on the input it receives.

This recipe highlights key implementation patterns:

- **AI SDK client integration**: The Workflow uses `generateText` from `ai` and `temporalProvider` from `@temporalio/ai-sdk/workflow`. This automatically wraps the LLM invocation as an Activity, so it's retried and tracked like any other durable step. `temporalProvider` is configured for `gpt-4o-mini` here, but you can point it at any model the AI SDK supports.
- **Tools-as-Activities**: `proxyActivities` wires the `getWeather` and `calculateAreaOfCircle` Activities into the Workflow so `toolsAgent` can offer tool schemas to the model, wait for results durably, and retry a tool call if it fails.

Unlike some other Temporal AI integrations — for example, the OpenAI Agents SDK's `activity_as_tool` helper, which generates a tool schema from a Python function's type hints — the Vercel AI SDK's `tool()` has no equivalent auto-generation from a TypeScript function signature. Each tool's `inputSchema` is written by hand as a Zod schema.

## Create the Activity

Temporal Activities provide the tools that `toolsAgent` can call. `getWeather` demonstrates an Activity that wraps an unreliable external call: it geocodes the city name (via the free [Open-Meteo geocoding API](https://open-meteo.com/en/docs/geocoding-api)) and then queries the [National Weather Service API](https://www.weather.gov/documentation/services-web-api), both of which are free and require no API key. Because the NWS API only covers the United States, use a US city in your prompt. `calculateAreaOfCircle` shows the opposite case — a tool that runs entirely locally with no external I/O.

*File: src/activities.ts*

```ts
const USER_AGENT = '(temporal-ai-cookbook, cookbook@temporal.io)';

async function geocode(location: string): Promise<{ name: string; latitude: number; longitude: number }> {
  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&countryCode=US`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Geocoding request failed for "${location}": ${response.status}`);
  }
  const data = (await response.json()) as {
    results?: Array<{ name: string; latitude: number; longitude: number }>;
  };
  const match = data.results?.[0];
  if (!match) {
    throw new Error(`No location found for "${location}". The National Weather Service only covers the US.`);
  }
  return match;
}

// The National Weather Service API only covers the US and requires a two-step
// lookup: resolve coordinates to a forecast grid, then fetch that grid's forecast.
export async function getWeather(input: {
  location: string;
}): Promise<{ city: string; temperatureRange: string; conditions: string }> {
  const { name, latitude, longitude } = await geocode(input.location);

  const pointsResponse = await fetch(`https://api.weather.gov/points/${latitude.toFixed(4)},${longitude.toFixed(4)}`, {
    headers: { 'User-Agent': USER_AGENT },
  });
  if (!pointsResponse.ok) {
    throw new Error(`National Weather Service points lookup failed: ${pointsResponse.status}`);
  }
  const points = (await pointsResponse.json()) as { properties: { forecast: string } };

  const forecastResponse = await fetch(points.properties.forecast, { headers: { 'User-Agent': USER_AGENT } });
  if (!forecastResponse.ok) {
    throw new Error(`National Weather Service forecast lookup failed: ${forecastResponse.status}`);
  }
  const forecast = (await forecastResponse.json()) as {
    properties: { periods: Array<{ temperature: number; temperatureUnit: string; shortForecast: string }> };
  };
  const current = forecast.properties.periods[0];

  return {
    city: name,
    temperatureRange: `${current.temperature}${current.temperatureUnit}`,
    conditions: current.shortForecast,
  };
}

export async function calculateAreaOfCircle(input: { radius: number }): Promise<{ area: number }> {
  return { area: Math.PI * input.radius * input.radius };
}
```

## Create the Workflow

The Workflow registers both Activities as tools with a Zod schema so the model can call them when appropriate.

*File: src/workflows.ts*

```ts
import type * as activities from './activities';
import { generateText, stepCountIs, tool } from 'ai';
import { temporalProvider } from '@temporalio/ai-sdk/workflow';
import { proxyActivities } from '@temporalio/workflow';
import z from 'zod';

const { getWeather, calculateAreaOfCircle } = proxyActivities<typeof activities>({
  startToCloseTimeout: '1 minute',
  retry: {
    maximumAttempts: 3,
  },
});

export async function toolsAgent(question: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt: question,
    system: 'You are a helpful agent.',
    tools: {
      getWeather: tool({
        description: 'Get the weather for a given city',
        inputSchema: z.object({
          location: z.string().describe('The location to get the weather for'),
        }),
        execute: getWeather,
      }),
      calculateCircleArea: tool({
        description: 'Calculate the area of a circle',
        inputSchema: z.object({
          radius: z.number().describe('The radius of the circle'),
        }),
        execute: calculateAreaOfCircle,
      }),
    },
    stopWhen: stepCountIs(5),
  });
  return result.text;
}
```

## Create the Worker

Create the process for executing Activities and Workflows. The Worker uses `AiSdkPlugin` to configure the OpenAI provider and to keep Workflow code isolated from the Activity environment.

*File: src/worker.ts*

```ts
import { NativeConnection, Worker } from '@temporalio/worker';
import * as activities from './activities';
import { AiSdkPlugin } from '@temporalio/ai-sdk';
import { openai } from '@ai-sdk/openai';

async function run() {
  const connection = await NativeConnection.connect({ address: 'localhost:7233' });
  const worker = await Worker.create({
    plugins: [new AiSdkPlugin({ modelProvider: openai })],
    connection,
    namespace: 'default',
    taskQueue: 'ai-sdk',
    workflowsPath: require.resolve('./workflows'),
    activities,
  });
  await worker.run();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

## Create the Workflow Starter

The starter (`src/client.ts`) takes the question to ask as a command-line argument, spins up a Temporal client, and starts `toolsAgent` with a new Workflow Id.

*File: src/client.ts*

```ts
import { Connection, Client } from '@temporalio/client';
import { loadClientConnectConfig } from '@temporalio/envconfig';
import { toolsAgent } from './workflows';
import { nanoid } from 'nanoid';

async function run() {
  const question = process.argv.slice(2).join(' ') || 'What is the weather in Seattle right now?';

  const config = loadClientConnectConfig();
  const connection = await Connection.connect(config.connectionOptions);
  const client = new Client({ connection });

  const handle = await client.workflow.start(toolsAgent, {
    taskQueue: 'ai-sdk',
    args: [question],
    workflowId: 'workflow-' + nanoid(),
  });

  console.log(`Started workflow ${handle.workflowId}`);
  console.log(await handle.result());
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

## Running

Start the Temporal Dev Server:
```bash
temporal server start-dev
```

Install all dependencies:
```bash
npm install
```

Set your API Key based on your preferred model provider:
```bash
export OPENAI_API_KEY=<KEY>
```

Run the worker:
```bash
npm run start.watch
```

Start execution with the default question, or supply your own:
```bash
npm run workflow
npm run workflow "What is the weather in Chicago?"
npm run workflow "Calculate the area of a circle with radius 5"
```

## Example interactions

Try asking the agent questions like:

- "What is the weather in Seattle right now?"
- "Calculate the area of a circle with radius 5"
- "What is the weather in Chicago and calculate the area of a circle with radius 3"

The agent decides which tools to use. Open the [Temporal UI](http://localhost:8233) to see the `invokeModel`, `getWeather`, and `calculateAreaOfCircle` Activities recorded in the Event History.
