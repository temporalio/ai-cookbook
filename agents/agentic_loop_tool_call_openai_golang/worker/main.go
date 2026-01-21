package main

import (
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/activity"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/workflow"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/log"
	"go.temporal.io/sdk/worker"
)

const TaskQueue = "agentic-loop-openai-golang-task-queue"

func main() {
	// Verify OPENAI_API_KEY is set
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
	}

	// Create a logger that only shows WARN and above (suppresses DEBUG and INFO)
	slogger := slog.New(slog.NewTextHandler(io.Discard, nil))
	logger := log.NewStructuredLogger(slogger)

	// Create Temporal client with quiet logger
	c, err := client.Dial(client.Options{
		HostPort: getEnvOrDefault("TEMPORAL_HOST", "localhost:7233"),
		Logger:   logger,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to create Temporal client: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	// Create worker
	w := worker.New(c, TaskQueue, worker.Options{})

	// Register workflow
	w.RegisterWorkflow(workflow.AgentWorkflow)

	// Register activities
	w.RegisterActivity(activity.OpenAIResponsesCreate)
	w.RegisterActivity(activity.ToolExecutor)

	fmt.Printf("Starting worker on task queue: %s\n", TaskQueue)
	fmt.Println("Press Ctrl+C to exit")

	// Start listening
	err = w.Run(worker.InterruptCh())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to start worker: %v\n", err)
		os.Exit(1)
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
