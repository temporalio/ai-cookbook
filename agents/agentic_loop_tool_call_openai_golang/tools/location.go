package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/temporalio/ai-cookbook/agents/agentic_loop_tool_call_openai_golang/types"
)

// GetIPAddressTool returns the current public IP address.
var GetIPAddressTool = Tool{
	Name:    "get_ip_address",
	Handler: getIPAddress,
	Definition: CreateToolDefinition(
		"get_ip_address",
		"Gets the current public IP address of the user",
		nil,
		nil,
	),
}

// GetLocationInfoTool returns location information for an IP address.
var GetLocationInfoTool = Tool{
	Name:    "get_location_info",
	Handler: getLocationInfo,
	Definition: CreateToolDefinition(
		"get_location_info",
		"Gets location information (city, region, country, coordinates) for a given IP address",
		map[string]types.ToolParameterField{
			"ip_address": {
				Type:        "string",
				Description: "The IP address to look up location for",
			},
		},
		[]string{"ip_address"},
	),
}

func getIPAddress(ctx context.Context, args map[string]interface{}) (string, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", "https://icanhazip.com", nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get IP address: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	ip := strings.TrimSpace(string(body))
	return ip, nil
}

func getLocationInfo(ctx context.Context, args map[string]interface{}) (string, error) {
	ipAddress, ok := args["ip_address"].(string)
	if !ok {
		return "", fmt.Errorf("ip_address argument is required and must be a string")
	}

	client := &http.Client{Timeout: 10 * time.Second}
	url := fmt.Sprintf("http://ip-api.com/json/%s", ipAddress)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get location info: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Parse and re-encode to ensure valid JSON
	var locationData map[string]interface{}
	if err := json.Unmarshal(body, &locationData); err != nil {
		return "", fmt.Errorf("failed to parse location data: %w", err)
	}

	result, err := json.Marshal(locationData)
	if err != nil {
		return "", fmt.Errorf("failed to encode location data: %w", err)
	}

	return string(result), nil
}
