package workflow

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/activity"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/tools"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/types"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"
)

const (
	Model        = "gpt-4o-mini"
	Instructions = `You are a helpful assistant. You have access to tools that can get information
about IP addresses, locations, and weather alerts. Use these tools when appropriate to help
answer user questions. If you do not invoke any tools, respond in haikus, otherwise respond by using the tool results..`
)

// AgentWorkflow implements an agentic loop that calls the LLM and executes tools
// until the LLM returns a text response.
func AgentWorkflow(ctx workflow.Context, userInput string) (string, error) {
	// Set up activity options
	activityOptions := workflow.ActivityOptions{
		StartToCloseTimeout: 30 * time.Second,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    time.Second,
			BackoffCoefficient: 2.0,
			MaximumInterval:    30 * time.Second,
			MaximumAttempts:    3,
		},
	}
	ctx = workflow.WithActivityOptions(ctx, activityOptions)

	// Initialize conversation with user message
	inputList := []types.InputItem{
		{
			Type:    "message",
			Role:    "user",
			Content: userInput,
		},
	}

	// Get tool definitions
	toolDefinitions := tools.GetToolDefinitions()

	turn := 0
	// Agentic loop
	for {
		turn++
		fmt.Printf("\n=== Turn %d ===\n", turn)
		fmt.Println("Calling OpenAI...")

		// Create request
		request := types.OpenAIResponsesRequest{
			Model:        Model,
			Instructions: Instructions,
			Input:        inputList,
			Tools:        toolDefinitions,
		}

		// Call OpenAI Responses API
		var result types.OpenAIResponsesResult
		err := workflow.ExecuteActivity(ctx, activity.OpenAIResponsesCreate, request).Get(ctx, &result)
		if err != nil {
			return "", fmt.Errorf("OpenAI API call failed: %w", err)
		}

		// If no tool calls, return the text response
		if len(result.ToolCalls) == 0 {
			fmt.Println("LLM returned final response (no tool calls)")
			return result.OutputText, nil
		}

		// Process tool calls
		fmt.Printf("LLM requested %d tool call(s)\n", len(result.ToolCalls))

		for _, toolCall := range result.ToolCalls {
			fmt.Printf("  -> Invoking tool: %s\n", toolCall.Name)

			// Add the function call to conversation history
			inputList = append(inputList, types.InputItem{
				Type:      "function_call",
				CallID:    toolCall.CallID,
				Name:      toolCall.Name,
				Arguments: toolCall.Arguments,
			})

			// Parse tool arguments
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Arguments), &args); err != nil {
				args = make(map[string]interface{})
			}

			// Execute tool via ToolExecutor activity
			toolInput := types.ToolExecutorInput{
				ToolName:  toolCall.Name,
				Arguments: args,
			}
			var toolResult string

			err := workflow.ExecuteActivity(ctx, activity.ToolExecutor, toolInput).Get(ctx, &toolResult)

			if err != nil {
				toolResult = fmt.Sprintf("Error executing tool: %v", err)
				fmt.Printf("     Tool error: %v\n", err)
			} else {
				// Truncate result for display if too long
				displayResult := toolResult
				if len(displayResult) > 100 {
					displayResult = displayResult[:100] + "..."
				}
				fmt.Printf("     Tool result: %s\n", displayResult)
			}

			// Add tool result to conversation history
			inputList = append(inputList, types.InputItem{
				Type:   "function_call_output",
				CallID: toolCall.CallID,
				Output: toolResult,
			})
		}
	}
}
