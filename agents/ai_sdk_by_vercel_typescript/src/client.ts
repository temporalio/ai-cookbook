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
