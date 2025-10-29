# ComfyUI_MCP

This repository provides a Model Context Protocol (MCP) server that exposes local ComfyUI workflow files and the remote ComfyUI model catalog to compatible LLM clients.

## Features

- Discovers workflow `.json` files stored under the `workflows/` directory and exposes them as MCP resources.
- Tools for listing workflow metadata and reading workflow contents.
- Tool for recursively querying the ComfyUI `/api/models` endpoints so an LLM can inspect the available checkpoints, LoRAs, and other assets.
- Configuration via CLI arguments or environment variables so the server can be launched from JSON descriptors (e.g., Cursor MCP definitions).

## Installation

Create a virtual environment (recommended) and install the package in editable mode with your preferred installer:

```bash
pip install -e .
```

The project also works with [uv](https://github.com/astral-sh/uv) so you can install or run it without using `pip` directly:

```bash
uv pip install -e .

# or run without installing into the current environment
uvx --from . comfyui-mcp --help
```

## Running the server

The installed `comfyui-mcp` entry point launches the MCP server over stdio. Common configuration options can be supplied either as CLI flags or environment variables:

| Purpose | CLI flag | Environment variable | Default |
|---------|----------|----------------------|---------|
| Workflow directory | `--workflow-dir` | `COMFYUI_WORKFLOW_DIR` | `<repo>/workflows` |
| ComfyUI API base URL | `--api-base-url` | `COMFYUI_API_BASE_URL` | `http://127.0.0.1:8188/` |
| HTTP timeout (seconds) | `--http-timeout` | `COMFYUI_HTTP_TIMEOUT` | `30` |
| Log level | `--log-level` | `COMFYUI_MCP_LOG_LEVEL` | `INFO` |

Example stdio launch configuration for a Cursor MCP JSON definition:

If you prefer `uvx`, the same configuration can be expressed as:

```json
{
  "comfyui": {
    "command": "uvx",
    "args": [
      "--from",
      ".",
      "comfyui-mcp",
      "--workflow-dir",
      "./workflows",
      "--api-base-url",
      "http://127.0.0.1:8188/"
    ]
  }
}
```

When referencing the project locally with `uvx`, ensure the working directory is set to the
repository root (or adjust the `--from` path accordingly) so the package can be resolved without
requiring it to be published to an external index.

Place your ComfyUI workflow files in the `workflows/` directory (or whatever directory you configure) so they are available to the LLM.
