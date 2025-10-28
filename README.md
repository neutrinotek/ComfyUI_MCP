# ComfyUI_MCP

This repository provides a Model Context Protocol (MCP) server that exposes local ComfyUI workflow files and the remote ComfyUI model catalog to compatible LLM clients.

## Features

- Discovers workflow `.json` files stored under the `workflows/` directory and exposes them as MCP resources.
- Tools for listing workflow metadata and reading workflow contents.
- Tool for recursively querying the ComfyUI `/api/models` endpoints so an LLM can inspect the available checkpoints, LoRAs, and other assets.
- Configuration via CLI arguments or environment variables so the server can be launched from JSON descriptors (e.g., Cursor MCP definitions).

## Installation

Create a virtual environment (recommended) and install the package in editable mode:

```bash
pip install -e .
```

## Running the server

The installed `comfyui-mcp` entry point launches the MCP server over stdio. Common configuration options can be supplied either as CLI flags or environment variables:

| Purpose | CLI flag | Environment variable | Default |
|---------|----------|----------------------|---------|
| Workflow directory | `--workflow-dir` | `COMFYUI_WORKFLOW_DIR` | `<repo>/workflows` |
| ComfyUI API base URL | `--api-base-url` | `COMFYUI_API_BASE_URL` | `http://10.27.27.5:8000/` |
| Models endpoint | `--models-base-url` | `COMFYUI_MODELS_BASE_URL` | `<api-base-url>/api/models` |
| HTTP timeout (seconds) | `--http-timeout` | `COMFYUI_HTTP_TIMEOUT` | `30` |
| Log level | `--log-level` | `COMFYUI_MCP_LOG_LEVEL` | `INFO` |

Example stdio launch configuration for a Cursor MCP JSON definition:

```json
{
  "comfyui": {
    "command": "python",
    "args": [
      "-m",
      "comfyui_mcp.server",
      "--workflow-dir",
      "./workflows",
      "--models-base-url",
      "http://10.27.27.5:8000/api/models"
    ]
  }
}
```

Place your ComfyUI workflow files in the `workflows/` directory (or whatever directory you configure) so they are available to the LLM.
