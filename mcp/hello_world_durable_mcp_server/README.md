# Hello World Durable MCP Server [PLACEHOLDER]

A durable MCP (Model Context Protocol) server that provides weather information using Temporal workflows for reliable execution.

## Overview

This project demonstrates how to build an MCP server with durable execution using Temporal. The server exposes weather tools that fetch alerts and forecasts from the National Weather Service API.

## Features

- **MCP Tools**: `get_alerts` and `get_forecast` exposed via FastMCP
- **Durable Workflows**: Weather operations run as Temporal workflows for reliability
- **NWS Integration**: Fetches data from the National Weather Service API

