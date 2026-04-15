<!--
description: Build a durable AI agent with the AI SDK by Vercel (Typescript) and Temporal that can intelligently choose tools to answer user questions
tags: [agents, typescript, tools]
priority: 5
-->

# Agent with Tools - AI SDK by Vercel

In this example, we show you how to build a Durable Agent using the [AI SDK by Vercel](https://docs.temporal.io/develop/typescript/ai-sdk#provide-your-durable-agent-with-tools). The AI agent we build call tools backed by Temporal Activities, and stream responses with the AI SDK. The agent can determine which tools to use based on the user's input and execute them as needed.

## Key implementation patterns

- **AI SDK client integration**: The Workflows use `generateText` from `ai` and imports `temporalProvider` from `@temporalio/ai-sdk`. This:
    - Automatically wraps the LLM invocation as an activity
    - Configures `temporalProvider` to point to `gpt-4o-mini` but you can use any model of your choice that is compatible with the AI SDK here.
- **Tools-as-Activities**: `proxyActivities` wires the `getWeather` and `calculateCircleArea` Activities into the Workflows so `toolsAgent` can offer tool schemas to the AI model, wait for results durably, and retry requests if needed.
- **Minimal orchestration**: The Workflow starter (`src/client.ts`) picks a Workflow name (`haiku` or `tools`) and runs it through a connected Temporal cluster, keeping the CLI code simple.
- **Worker plugin**: The `AiSdkPlugin` configures the Worker to use the OpenAI provider and keep Workflow definitions isolated from the Activity environment.

## Create the Activity

Temporal Activities provide the tools that `toolsAgent` can call. The sample keeps things simple with 2 tools: weather lookup and calculate the area of a circle.

*File: src/activities.ts*

```ts
export async function getWeather(input: { location: string }): Promise<{
  city: string;
  temperatureRange: string;
  conditions: string;
}> {
  console.log('Activity execution');
  return {
    city: input.location,
    temperatureRange: '14-20C',
    conditions: 'Sunny with wind.',
  };
}

export async function calculateCircleArea(radius: number): Promise<number> {
  return Math.PI * radius * radius;
}
```

## Create the Workflows

The Workflows show two ways of invoking the AI SDK: `haikuAgent` shows a basic agent with defined system instructions, in this case - generate a haiku, and `toolsAgent` registers tools with a Zod schema so the model can request it when appropriate.

*File: src/workflows.ts*

```ts
import { generateText, stepCountIs, tool } from 'ai';
import { temporalProvider } from '@temporalio/ai-sdk';
import { proxyActivities } from '@temporalio/workflow';
import z from 'zod';

const { getWeather, calculateCircleArea } = proxyActivities<typeof activities>({
  startToCloseTimeout: '1 minute',
});

export async function haikuAgent(prompt: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt,
    system: 'You only respond in haikus.',
  });
  return result.text;
}

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
        description: 'Calculate the area of a circle given its radius',
        inputSchema: z.object({
          radius: z.number().describe('The radius of the circle'),
        }),
        execute: calculateCircleArea,
      }),
    },
    stopWhen: stepCountIs(5),
  });
  return result.text;
}
```

## Create the Worker

Create the process for executing Activities and Workflows. The Worker instantiates `AiSdkPlugin` to enable the AI SDK integration.

*File: src/worker.ts*

```ts
import { NativeConnection, Worker } from '@temporalio/worker';
import * as activities from './activities';
import { AiSdkPlugin } from '@temporalio/ai-sdk';
import { openai } from '@ai-sdk/openai';

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
```

## Create the Workflow Starter

The starter (`src/client.ts`) accepts a Workflow name argument, spins up a Temporal client, and invokes the requested Workflow (`haiku` or `tools`) with a new Workflow ID.

*File: src/client.ts*

```ts
import { Connection, Client } from '@temporalio/client';
import { loadClientConnectConfig } from '@temporalio/envconfig';
import { haikuAgent, toolsAgent } from './workflows';
import { nanoid } from 'nanoid';

const args = process.argv;
const workflow = args[2] ?? 'haiku';

const config = loadClientConnectConfig();
const connection = await Connection.connect(config.connectionOptions);
const client = new Client({ connection });

let handle;
switch (workflow) {
  case 'tools':
    handle = await client.workflow.start(toolsAgent, {
      taskQueue: 'ai-sdk',
      args: ['What is the weather in Tokyo?'],
      workflowId: 'workflow-' + nanoid(),
    });
    break;
  case 'haiku':
  default:
    handle = await client.workflow.start(haikuAgent, {
      taskQueue: 'ai-sdk',
      args: ['Temporal'],
      workflowId: 'workflow-' + nanoid(),
    });
}

console.log(await handle.result());
```

## Running

Start the Temporal Dev Server:
```
temporal server start-dev
```

Install all dependencies:
```
npm install
```

Set your API Key based on your preferred model provider:
```
export OPENAI_API_KEY=<KEY>
```

Run the worker:
```
npm run start.watch
```

Start execution:
```
npm run workflow haiku
npm run workflow tools
```

## Example prompts

- `"Temporal"` → `haikuAgent` will respond with a haiku about Temporal
- `"What is the weather in Tokyo?"` → `toolsAgent` will call `getWeather` and weave the result into the final reply.
- `"Calculate the area of a circle with radius 5"` → `toolsAgent` will invoke `calculateCircleArea` and include the numeric result.
