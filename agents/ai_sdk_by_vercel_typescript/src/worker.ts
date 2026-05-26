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

