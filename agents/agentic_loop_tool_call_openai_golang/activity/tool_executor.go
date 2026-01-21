package activity

import (
	"context"
	"fmt"

	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/tools"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/types"
)

// ToolExecutor executes a tool by name and returns the result.
func ToolExecutor(ctx context.Context, input types.ToolExecutorInput) (string, error) {
	handler, err := tools.GetHandler(input.ToolName)
	if err != nil {
		return "", fmt.Errorf("unknown tool %s: %w", input.ToolName, err)
	}

	result, err := handler(ctx, input.Arguments)
	if err != nil {
		return "", fmt.Errorf("tool %s failed: %w", input.ToolName, err)
	}

	return result, nil
}
