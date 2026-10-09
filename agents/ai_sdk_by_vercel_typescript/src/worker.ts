import { NativeConnection, Worker } from '@temporalio/worker';
import { loadClientConnectConfig } from '@temporalio/envconfig';
import * as activities from './activities';
import { AiSdkPlugin } from '@temporalio/ai-sdk';
import { openai } from '@ai-sdk/openai';

async function run() {
  const config = loadClientConnectConfig();
  const connection = await NativeConnection.connect(config.connectionOptions);
  const worker = await Worker.create({
    plugins: [new AiSdkPlugin({ modelProvider: openai })],
    connection,
    namespace: config.namespace,
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

