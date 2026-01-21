package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/google/uuid"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/workflow"
	"go.temporal.io/sdk/client"
)

const TaskQueue = "agentic-loop-openai-golang-task-queue"

func main() {
	// Get user input from command line arguments
	userInput := "Tell me about recursion"
	if len(os.Args) > 1 {
		userInput = strings.Join(os.Args[1:], " ")
	}

	// Create Temporal client
	c, err := client.Dial(client.Options{
		HostPort: getEnvOrDefault("TEMPORAL_HOST", "localhost:7233"),
	})
	if err != nil {
		log.Fatalf("Unable to create Temporal client: %v", err)
	}
	defer c.Close()

	// Generate workflow ID
	workflowID := fmt.Sprintf("agentic-loop-golang-%s", uuid.New().String())

	// Start workflow options
	workflowOptions := client.StartWorkflowOptions{
		ID:        workflowID,
		TaskQueue: TaskQueue,
	}

	fmt.Printf("Starting workflow with input: %s\n", userInput)
	fmt.Printf("Workflow ID: %s\n\n", workflowID)

	// Execute workflow and wait for result
	we, err := c.ExecuteWorkflow(context.Background(), workflowOptions, workflow.AgentWorkflow, userInput)
	if err != nil {
		log.Fatalf("Unable to execute workflow: %v", err)
	}

	// Wait for workflow completion
	var result string
	err = we.Get(context.Background(), &result)
	if err != nil {
		log.Fatalf("Workflow failed: %v", err)
	}

	fmt.Printf("Result:\n%s\n", result)
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
