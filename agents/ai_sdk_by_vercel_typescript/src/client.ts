import { Connection, Client } from '@temporalio/client';
import { loadClientConnectConfig } from '@temporalio/envconfig';
import { haikuAgent, toolsAgent } from './workflows';
import { nanoid } from 'nanoid';

async function run() {
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

  console.log(`Started workflow ${handle.workflowId}`);
  console.log(await handle.result());
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});

