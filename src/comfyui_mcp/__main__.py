"""Module entry-point for running the ComfyUI MCP server."""

from __future__ import annotations

from .server import main


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    main()
