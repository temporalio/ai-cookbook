package types

// OpenAIResponsesRequest contains the parameters for calling OpenAI's Responses API.
type OpenAIResponsesRequest struct {
	Model        string           `json:"model"`
	Instructions string           `json:"instructions"`
	Input        []InputItem      `json:"input"`
	Tools        []ToolDefinition `json:"tools"`
}

// InputItem represents a single item in the conversation history.
// It can be a user message, assistant message, function call, or function call output.
type InputItem struct {
	Type      string `json:"type"`                 // "message", "function_call", "function_call_output"
	Role      string `json:"role,omitempty"`       // "user", "assistant" (for messages)
	Content   string `json:"content,omitempty"`    // Message content
	CallID    string `json:"call_id,omitempty"`    // For function_call and function_call_output
	Name      string `json:"name,omitempty"`       // For function_call
	Arguments string `json:"arguments,omitempty"`  // For function_call (JSON string)
	Output    string `json:"output,omitempty"`     // For function_call_output
}

// OpenAIResponsesResult contains the result from OpenAI's Responses API.
type OpenAIResponsesResult struct {
	OutputText string         `json:"output_text"`
	ToolCalls  []ToolCallInfo `json:"tool_calls"`
}

// ToolCallInfo contains information about a tool call returned by the LLM.
type ToolCallInfo struct {
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolDefinition defines a tool for OpenAI's Responses API.
type ToolDefinition struct {
	Type        string           `json:"type"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Parameters  ToolParameters   `json:"parameters"`
	Strict      bool             `json:"strict"`
}

// ToolParameters defines the JSON Schema for a tool's parameters.
type ToolParameters struct {
	Type                 string                        `json:"type"`
	Properties           map[string]ToolParameterField `json:"properties"`
	Required             []string                      `json:"required"`
	AdditionalProperties bool                          `json:"additionalProperties"`
}

// ToolParameterField defines a single parameter field in a tool's schema.
type ToolParameterField struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// ToolExecutorInput is the input for the tool executor activity.
type ToolExecutorInput struct {
	ToolName  string                 `json:"tool_name"`
	Arguments map[string]interface{} `json:"arguments"`
}
