package tools

import (
	"context"
	"fmt"

	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/types"
)

// ToolHandler is a function that executes a tool and returns a result.
type ToolHandler func(ctx context.Context, args map[string]interface{}) (string, error)

// Tool represents a registered tool with its handler and definition.
type Tool struct {
	Name       string
	Handler    ToolHandler
	Definition types.ToolDefinition
}

// Registry maps tool names to their handlers.
var Registry = map[string]Tool{
	"get_ip_address":     GetIPAddressTool,
	"get_location_info":  GetLocationInfoTool,
	"get_weather_alerts": GetWeatherAlertsTool,
}

// GetHandler returns the handler function for a given tool name.
func GetHandler(toolName string) (ToolHandler, error) {
	tool, ok := Registry[toolName]
	if !ok {
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
	return tool.Handler, nil
}

// GetToolDefinitions returns all tool definitions for use with the OpenAI API.
func GetToolDefinitions() []types.ToolDefinition {
	definitions := make([]types.ToolDefinition, 0, len(Registry))
	for _, tool := range Registry {
		definitions = append(definitions, tool.Definition)
	}
	return definitions
}

// CreateToolDefinition is a helper to create a tool definition with proper structure.
func CreateToolDefinition(name, description string, properties map[string]types.ToolParameterField, required []string) types.ToolDefinition {
	if properties == nil {
		properties = make(map[string]types.ToolParameterField)
	}
	if required == nil {
		required = []string{}
	}
	return types.ToolDefinition{
		Type:        "function",
		Name:        name,
		Description: description,
		Parameters: types.ToolParameters{
			Type:                 "object",
			Properties:           properties,
			Required:             required,
			AdditionalProperties: false,
		},
		Strict: true,
	}
}
