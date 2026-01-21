package activity

import (
	"context"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/types"
)

// OpenAIResponsesCreate calls OpenAI's Responses API and returns the result.
func OpenAIResponsesCreate(ctx context.Context, request types.OpenAIResponsesRequest) (*types.OpenAIResponsesResult, error) {
	client := openai.NewClient(
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		option.WithMaxRetries(0), // Temporal handles retries
	)

	// Build input items
	inputItems := make([]responses.ResponseInputItemUnionParam, 0, len(request.Input))
	for _, item := range request.Input {
		switch item.Type {
		case "message":
			inputItems = append(inputItems, responses.ResponseInputItemParamOfMessage(
				item.Content,
				responses.EasyInputMessageRole(item.Role),
			))
		case "function_call":
			inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCall(
				item.Arguments,
				item.CallID,
				item.Name,
			))
		case "function_call_output":
			inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(
				item.CallID,
				item.Output,
			))
		}
	}

	// Build tool definitions
	tools := make([]responses.ToolUnionParam, 0, len(request.Tools))
	for _, tool := range request.Tools {
		// Build properties map for JSON schema
		properties := make(map[string]interface{})
		for name, field := range tool.Parameters.Properties {
			properties[name] = map[string]interface{}{
				"type":        field.Type,
				"description": field.Description,
			}
		}

		params := map[string]interface{}{
			"type":                 tool.Parameters.Type,
			"properties":           properties,
			"required":             tool.Parameters.Required,
			"additionalProperties": tool.Parameters.AdditionalProperties,
		}

		toolParam := responses.ToolParamOfFunction(tool.Name, params, tool.Strict)
		toolParam.OfFunction.Description = openai.String(tool.Description)
		tools = append(tools, toolParam)
	}

	// Create the request
	params := responses.ResponseNewParams{
		Model:        request.Model,
		Instructions: openai.String(request.Instructions),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: inputItems,
		},
		Tools: tools,
	}

	resp, err := client.Responses.New(ctx, params)
	if err != nil {
		return nil, err
	}

	// Process response output
	result := &types.OpenAIResponsesResult{
		OutputText: "",
		ToolCalls:  make([]types.ToolCallInfo, 0),
	}

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				if content.Type == "output_text" {
					result.OutputText += content.Text
				}
			}
		case "function_call":
			result.ToolCalls = append(result.ToolCalls, types.ToolCallInfo{
				CallID:    item.CallID,
				Name:      item.Name,
				Arguments: item.Arguments,
			})
		}
	}

	return result, nil
}
