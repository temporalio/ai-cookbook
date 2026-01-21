<!--
description: Durable agentic loop with OpenAI Responses API in Go
tags: [agents, golang, openai]
priority: 745
-->

# Durable Agent with Agentic Loop - OpenAI Responses API (Go)

This example demonstrates a durable AI agent implemented in Go using OpenAI's Responses API with Temporal. The agent uses an agentic loop pattern that continues executing tools until the LLM returns a text response.

## Overview

The agent implements a loop-based pattern where:
1. User input is sent to OpenAI's Responses API
2. If the response contains tool calls, tools are executed and results fed back
3. The loop continues until the LLM returns a text response
4. Temporal provides durability, so the workflow survives failures and restarts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AgentWorkflow                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 Agentic Loop                       │ │
│  │  ┌──────────────┐    ┌──────────────────────────┐  │ │
│  │  │   OpenAI     │───▶│   Tool Calls?            │  │ │
│  │  │  Responses   │    │                          │  │ │
│  │  │   Activity   │    │  Yes: Execute tools      │  │ │
│  │  └──────────────┘    │       via ToolExecutor   │  │ │
│  │         ▲            │       activity           │  │ │
│  │         │            │                          │  │ │
│  │         └────────────│  No: Return text         │  │ │
│  │                      └──────────────────────────┘  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
agentic_loop_tool_call_openai_golang/
├── go.mod                      # Module definition
├── README.md                   # This file
├── types/
│   └── types.go                # Shared data structures
├── tools/
│   ├── tools.go                # Tool registry and definitions
│   ├── location.go             # IP address and location tools
│   └── weather.go              # Weather alerts tool
├── activity/
│   ├── openai_responses.go     # OpenAI Responses API wrapper
│   └── tool_executor.go        # Tool execution activity
├── workflow/
│   └── agent.go                # Agentic loop workflow
├── worker/
│   └── main.go                 # Worker entry point
└── starter/
    └── main.go                 # CLI workflow starter
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_ip_address` | Gets the current public IP address |
| `get_location_info` | Gets location data for an IP address |
| `get_weather_alerts` | Gets NWS weather alerts for a US state |

## Prerequisites

- Go 1.23 or later
- Temporal Server (local or cloud)
- OpenAI API key

## Running the Example

### 1. Start Temporal Server

```bash
temporal server start-dev
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY=your-api-key-here
```

### 3. Install Dependencies and Start Worker

```bash
cd agents/agentic_loop_tool_call_openai_golang
go mod tidy
go run ./worker
```

### 4. Run Queries

In a separate terminal:

```bash
# Simple query (no tools needed - responds in haikus)
go run ./starter "tell me a joke"

# Query requiring tool use
go run ./starter "what is my ip address?"

# Multi-tool query
go run ./starter "where am I located?"

# Weather query
go run ./starter "are there weather alerts in CA?"
```

## Key Implementation Details

### Tool Executor Pattern

The project uses a `ToolExecutor` activity that dispatches to the appropriate tool handler based on the tool name:

```go
// Activity receives tool name and arguments
func ToolExecutor(ctx context.Context, input types.ToolExecutorInput) (string, error) {
    handler, err := tools.GetHandler(input.ToolName)
    if err != nil {
        return "", err
    }
    return handler(ctx, input.Arguments)
}

// Workflow calls ToolExecutor with tool info
toolInput := types.ToolExecutorInput{
    ToolName:  toolCall.Name,
    Arguments: args,
}
workflow.ExecuteActivity(ctx, activity.ToolExecutor, toolInput)
```

This pattern keeps the workflow code simple while allowing tools to be added/removed by updating the registry.

### OpenAI Client Configuration

The client disables retries because Temporal handles retry logic:

```go
client := openai.NewClient(
    option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    option.WithMaxRetries(0), // Temporal handles retries
)
```

### Tool Registry Pattern

Tools are registered in a map, allowing easy addition/removal without changing workflow code:

```go
var Registry = map[string]Tool{
    "get_ip_address":     GetIPAddressTool,
    "get_location_info":  GetLocationInfoTool,
    "get_weather_alerts": GetWeatherAlertsTool,
}
```

## Adding New Tools

1. Create a new file in `tools/` (e.g., `tools/my_tool.go`)
2. Define the tool with handler and definition:

```go
var MyTool = Tool{
    Name:    "my_tool",
    Handler: myToolHandler,
    Definition: CreateToolDefinition(
        "my_tool",
        "Description of what the tool does",
        map[string]types.ToolParameterField{
            "param1": {Type: "string", Description: "Parameter description"},
        },
        []string{"param1"},
    ),
}

func myToolHandler(ctx context.Context, args map[string]interface{}) (string, error) {
    // Implementation
    return "result", nil
}
```

3. Add to registry in `tools/tools.go`:

```go
var Registry = map[string]Tool{
    // ... existing tools
    "my_tool": MyTool,
}
```

## Related Examples

- [Python version](../agentic_loop_tool_call_openai_python/) - Same pattern in Python
- [Claude version](../agentic_loop_tool_call_claude_python/) - Using Anthropic's Claude
- [Human-in-the-loop](../human_in_the_loop_python/) - Adding approval workflows
