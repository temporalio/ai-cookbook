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

// GetWeatherAlertsTool returns active weather alerts for a US state.
var GetWeatherAlertsTool = Tool{
	Name:    "get_weather_alerts",
	Handler: getWeatherAlerts,
	Definition: CreateToolDefinition(
		"get_weather_alerts",
		"Gets active weather alerts from the National Weather Service for a given US state",
		map[string]types.ToolParameterField{
			"state": {
				Type:        "string",
				Description: "Two-letter US state code (e.g., 'CA', 'NY', 'TX')",
			},
		},
		[]string{"state"},
	),
}

// WeatherAlert represents a simplified weather alert from the NWS API.
type WeatherAlert struct {
	Headline    string `json:"headline"`
	Description string `json:"description"`
	Severity    string `json:"severity"`
	Event       string `json:"event"`
	AreaDesc    string `json:"areaDesc"`
}

func getWeatherAlerts(ctx context.Context, args map[string]interface{}) (string, error) {
	state, ok := args["state"].(string)
	if !ok {
		return "", fmt.Errorf("state argument is required and must be a string")
	}

	state = strings.ToUpper(state)
	url := fmt.Sprintf("https://api.weather.gov/alerts/active?area=%s", state)

	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", "temporal-ai-cookbook/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get weather alerts: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Parse the NWS API response
	var apiResponse struct {
		Features []struct {
			Properties struct {
				Headline    string `json:"headline"`
				Description string `json:"description"`
				Severity    string `json:"severity"`
				Event       string `json:"event"`
				AreaDesc    string `json:"areaDesc"`
			} `json:"properties"`
		} `json:"features"`
	}

	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return "", fmt.Errorf("failed to parse weather alerts: %w", err)
	}

	// Extract relevant alert information
	alerts := make([]WeatherAlert, 0, len(apiResponse.Features))
	for _, feature := range apiResponse.Features {
		alerts = append(alerts, WeatherAlert{
			Headline:    feature.Properties.Headline,
			Description: feature.Properties.Description,
			Severity:    feature.Properties.Severity,
			Event:       feature.Properties.Event,
			AreaDesc:    feature.Properties.AreaDesc,
		})
	}

	if len(alerts) == 0 {
		return fmt.Sprintf("No active weather alerts for %s", state), nil
	}

	result, err := json.Marshal(alerts)
	if err != nil {
		return "", fmt.Errorf("failed to encode alerts: %w", err)
	}

	return string(result), nil
}
