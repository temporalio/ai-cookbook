import '@temporalio/ai-sdk/lib/load-polyfills';
import type * as activities from './activities';
import { generateText, stepCountIs, tool } from 'ai';
import { temporalProvider } from '@temporalio/ai-sdk';
import { proxyActivities } from '@temporalio/workflow';
import z from 'zod';

const { getWeather, calculateAreaOfCircle } = proxyActivities<typeof activities>({
    startToCloseTimeout: '1 minute',
    retry: {
      maximumAttempts: 3,
    },
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