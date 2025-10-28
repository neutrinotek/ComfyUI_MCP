"""ComfyUI Model Context Protocol server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin

import anyio
import httpx
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.resources import FileResource
from mcp.server.fastmcp.server import FastMCP


@dataclass(slots=True)
class ServerConfig:
    """Configuration container for the MCP server."""

    workflow_dir: Path
    models_base_url: str
    http_timeout: float = 30.0

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "ServerConfig":
        """Create a configuration object by merging CLI arguments and environment variables."""

        workflow_dir = Path(
            args.workflow_dir
            or os.environ.get("COMFYUI_WORKFLOW_DIR")
            or Path.cwd() / "workflows"
        )

        models_base_url = (
            args.models_base_url
            or os.environ.get("COMFYUI_MODELS_BASE_URL")
            or _derive_models_url(
                args.api_base_url or os.environ.get("COMFYUI_API_BASE_URL")
            )
        )

        timeout = float(
            args.http_timeout
            or os.environ.get("COMFYUI_HTTP_TIMEOUT")
            or 30.0
        )

        return cls(
            workflow_dir=workflow_dir,
            models_base_url=models_base_url,
            http_timeout=timeout,
        )


def _derive_models_url(api_base_url: Optional[str]) -> str:
    """Infer the models endpoint from the broader ComfyUI API base URL."""

    default_base = "http://10.27.27.5:8000/"
    base = (api_base_url or default_base).rstrip("/") + "/"
    return urljoin(base, "api/models")


class WorkflowRepository:
    """Helper around the workflows directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def iter_workflows(self) -> Iterable[Path]:
        """Yield all workflow JSON files sorted by path."""

        return sorted(self.root.rglob("*.json"))

    def describe(self, workflow: Path) -> dict[str, Any]:
        """Return metadata about a workflow file."""

        stat = workflow.stat()
        rel_path = workflow.relative_to(self.root).as_posix()
        return {
            "name": workflow.stem,
            "relative_path": rel_path,
            "absolute_path": str(workflow),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

    def read(self, relative_path: str) -> str:
        """Read a workflow file by its relative path."""

        candidate = (self.root / relative_path).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise ValueError("Attempted to read workflow outside of the configured directory")
        if not candidate.exists():
            raise FileNotFoundError(f"Workflow '{relative_path}' not found")
        return candidate.read_text(encoding="utf-8")


class ComfyUIModelClient:
    """Client responsible for retrieving model catalog information."""

    def __init__(self, models_base_url: str, timeout: float = 30.0) -> None:
        self.models_base_url = models_base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(self) -> Iterable[None]:
        """Ensure the underlying HTTP client is properly managed."""

        async with httpx.AsyncClient(timeout=self.timeout) as session:
            self._client = session
            try:
                yield
            finally:
                self._client = None

    async def list_models(
        self,
        *,
        path: str | None = None,
        recursive: bool = True,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Fetch models from the ComfyUI API."""

        if self._client is None:
            raise RuntimeError("Model client not initialised")

        effective_path = path.strip("/") if path else ""
        url = self._compose_url(effective_path)

        response = await self._client.get(url, params={"recursive": str(recursive).lower()})
        response.raise_for_status()

        parsed = await self._parse_response(response)

        if isinstance(parsed, dict) and set(parsed.keys()) == {"raw"}:
            return {
                "requested_path": effective_path,
                "base_url": self.models_base_url,
                "recursive": recursive,
                "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
                "entries": [],
                "raw_response": parsed["raw"],
            }

        flattened = self._flatten(parsed, prefix=effective_path)
        if search:
            needle = search.lower()
            flattened = [
                entry
                for entry in flattened
                if needle in entry["name"].lower()
                or needle in entry.get("path", "").lower()
                or any(
                    needle in str(value).lower()
                    for value in entry.get("metadata", {}).values()
                )
            ]

        return {
            "requested_path": effective_path,
            "base_url": self.models_base_url,
            "recursive": recursive,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "entries": flattened,
        }

    async def _parse_response(self, response: httpx.Response) -> Any:
        """Parse the API response as JSON, falling back to text."""

        content_type = response.headers.get("content-type", "")
        text = response.text
        if "application/json" in content_type:
            return response.json()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def _compose_url(self, path: str) -> str:
        if path:
            return f"{self.models_base_url}/{path}"
        return self.models_base_url

    def _flatten(self, payload: Any, *, prefix: str = "") -> list[dict[str, Any]]:
        """Flatten nested model information into searchable entries."""

        entries: list[dict[str, Any]] = []

        def handle(value: Any, segments: list[str]) -> None:
            if isinstance(value, dict):
                if _looks_like_file_entry(value):
                    entries.append(_normalise_file_entry(value, segments))
                    return

                files_handled = False
                if "files" in value:
                    files_handled = True
                    handle(value["files"], segments)
                if "items" in value:
                    files_handled = True
                    handle(value["items"], segments)
                if "models" in value and isinstance(value["models"], (list, dict)):
                    files_handled = True
                    handle(value["models"], segments)

                directories = value.get("directories") or value.get("children") or value.get("folders")
                if directories is not None:
                    files_handled = True
                    if isinstance(directories, dict):
                        for name, child in directories.items():
                            handle(child, segments + [str(name)])
                    elif isinstance(directories, list):
                        for child in directories:
                            handle(child, segments)

                if files_handled:
                    remaining = {
                        key: val
                        for key, val in value.items()
                        if key
                        not in {"files", "items", "models", "directories", "children", "folders"}
                    }
                    if remaining:
                        handle(remaining, segments)
                else:
                    for key, child in value.items():
                        handle(child, segments + [str(key)])
            elif isinstance(value, list):
                for item in value:
                    handle(item, segments)
            else:
                entries.append(
                    {
                        "kind": "file",
                        "name": str(value),
                        "path": "/".join([s for s in segments if s]) or prefix,
                        "metadata": {},
                    }
                )

        base_segments = [s for s in prefix.split("/") if s] if prefix else []
        handle(payload, base_segments)
        unique_entries: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in entries:
            key = (entry.get("path", ""), entry["name"])
            if key not in unique_entries:
                unique_entries[key] = entry
        return list(unique_entries.values())


def _looks_like_file_entry(data: dict[str, Any]) -> bool:
    keys = set(data)
    file_keys = {"name", "filename", "path", "file", "title"}
    return bool(keys & file_keys) and not keys & {"directories", "children", "folders"}


def _normalise_file_entry(data: dict[str, Any], segments: list[str]) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"name", "filename", "path", "file", "title"}
    }
    name = (
        str(data.get("name")
            or data.get("filename")
            or data.get("title")
            or data.get("file")
            or data.get("path")
            or "entry")
    )
    source_path = data.get("path") or "/".join(segments + [name])
    return {
        "kind": "file",
        "name": name,
        "path": str(source_path),
        "metadata": metadata,
    }


def create_server(config: ServerConfig) -> FastMCP:
    """Build the configured FastMCP server."""

    workflow_repo = WorkflowRepository(config.workflow_dir)
    model_client = ComfyUIModelClient(config.models_base_url, timeout=config.http_timeout)

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        async with model_client.lifespan():
            yield

    server = FastMCP(
        name="comfyui-mcp",
        instructions=(
            "Expose ComfyUI workflows and model catalog information as MCP resources"
            " and tools."
        ),
        lifespan=lifespan,
    )

    for workflow in workflow_repo.iter_workflows():
        relative = workflow.relative_to(workflow_repo.root).as_posix()
        server.add_resource(
            FileResource(
                uri=f"workflow://local/{relative}",
                name=workflow.stem,
                description="ComfyUI workflow definition",
                mime_type="application/json",
                path=workflow,
            )
        )

    @server.tool(
        name="list_workflows",
        description="List all ComfyUI workflow files that are available on disk.",
    )
    async def list_workflows(include_preview: bool = False, context: Context | None = None) -> dict[str, Any]:
        """Return metadata about workflows stored on disk."""

        entries = []
        for workflow in workflow_repo.iter_workflows():
            description = workflow_repo.describe(workflow)
            if include_preview:
                preview = await anyio.to_thread.run_sync(
                    workflow.read_text, "utf-8"
                )
                description["preview"] = preview
            entries.append(description)
        entries.sort(key=lambda item: item["relative_path"])
        if context is not None:
            context.info(f"Found {len(entries)} workflow files")
        return {
            "workflow_root": str(workflow_repo.root),
            "count": len(entries),
            "workflows": entries,
        }

    @server.tool(
        name="read_workflow",
        description=(
            "Read a specific workflow file by its relative path inside the"
            " configured workflow directory."
        ),
    )
    async def read_workflow(relative_path: str, context: Context | None = None) -> dict[str, Any]:
        """Return the contents of a workflow JSON file."""

        text = await anyio.to_thread.run_sync(workflow_repo.read, relative_path)
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if context is not None:
            context.info(f"Read workflow {relative_path}")
        return {
            "relative_path": relative_path,
            "text": text,
            "json": parsed,
        }

    @server.tool(
        name="list_models",
        description=(
            "Inspect the ComfyUI model folders via the configured API endpoint."
            " Supports recursive traversal and basic search."
        ),
    )
    async def list_models(
        path: str | None = None,
        recursive: bool = True,
        search: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """List models available in the remote ComfyUI instance."""

        result = await model_client.list_models(path=path, recursive=recursive, search=search)
        if context is not None:
            context.info(
                "Retrieved %s model entries from %s",
                len(result.get("entries", [])),
                result.get("base_url"),
            )
        return result

    return server


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(description="ComfyUI MCP server")
    parser.add_argument(
        "--workflow-dir",
        help="Directory containing ComfyUI workflow JSON files",
    )
    parser.add_argument(
        "--api-base-url",
        help="Base URL for the ComfyUI HTTP API (used to derive the models endpoint)",
    )
    parser.add_argument(
        "--models-base-url",
        help="Explicit URL for the ComfyUI models API (overrides the derived value)",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        help="HTTP timeout when querying the ComfyUI API",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("COMFYUI_MCP_LOG_LEVEL", "INFO"),
        help="Logging verbosity for the underlying FastMCP server",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MCP server CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = ServerConfig.from_env_and_args(args)

    server = create_server(config)

    # FastMCP respects the global logging level; expose CLI/environment setting.
    os.environ["MCP_LOG_LEVEL"] = args.log_level

    asyncio.run(server.run_stdio_async())


__all__ = ["create_server", "main", "ServerConfig"]


if __name__ == "__main__":  # pragma: no cover
    main()
