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

import anyio
import httpx
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.resources import FileResource
from mcp.server.fastmcp.server import FastMCP


# Mapping from user-friendly model kinds to ComfyUI loader node metadata
MODEL_KIND_MAP: dict[str, tuple[str, str]] = {
    "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
    "loras": ("LoraLoader", "lora_name"),
    "vae": ("VAELoader", "vae_name"),
    "clip": ("CLIPLoader", "clip_name"),
    "controlnet": ("ControlNetLoader", "control_net_name"),
}


@dataclass(slots=True)
class ServerConfig:
    """Configuration container for the MCP server."""

    workflow_dir: Path
    api_base_url: str
    http_timeout: float = 30.0

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "ServerConfig":
        """Create a configuration object by merging CLI arguments and environment variables."""

        workflow_dir = Path(
            args.workflow_dir
            or os.environ.get("COMFYUI_WORKFLOW_DIR")
            or Path.cwd() / "workflows"
        )

        legacy_models_url = getattr(args, "models_base_url", None) or os.environ.get(
            "COMFYUI_MODELS_BASE_URL"
        )

        api_base_url = _normalise_api_base_url(
            args.api_base_url
            or os.environ.get("COMFYUI_API_BASE_URL")
            or _fallback_models_url(legacy_models_url)
            or "http://127.0.0.1:8188"
        )

        timeout = float(
            args.http_timeout
            or os.environ.get("COMFYUI_HTTP_TIMEOUT")
            or 30.0
        )

        return cls(
            workflow_dir=workflow_dir,
            api_base_url=api_base_url,
            http_timeout=timeout,
        )


def _normalise_api_base_url(raw: str) -> str:
    """Ensure the configured base URL is well-formed."""

    candidate = raw.strip()
    if not candidate:
        raise ValueError("ComfyUI API base URL cannot be empty")
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _fallback_models_url(raw: Optional[str]) -> Optional[str]:
    """Allow legacy COMFYUI_MODELS_BASE_URL to specify the API base."""

    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if "://" not in cleaned:
        cleaned = f"http://{cleaned}"
    cleaned = cleaned.rstrip("/")
    suffix = "/api/models"
    if cleaned.endswith(suffix):
        return cleaned[: -len(suffix)]
    return cleaned


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

    def __init__(self, api_base_url: str, timeout: float = 30.0) -> None:
        self.api_base_url = api_base_url.rstrip("/")
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
        kind: str | None = None,
        recursive: bool = False,
        search: str | None = None,
    ) -> dict[str, Any] | list[str]:
        """Fetch model information from the ComfyUI object info endpoints."""

        if self._client is None:
            raise RuntimeError("Model client not initialised")

        kinds = [kind] if kind else list(MODEL_KIND_MAP.keys())
        if recursive:
            kinds = list(dict.fromkeys(kinds))

        search_lower = search.lower() if search else None
        inventory: dict[str, list[str]] = {}
        errors: dict[str, str] = {}

        for model_kind in kinds:
            if model_kind not in MODEL_KIND_MAP:
                raise ValueError(f"Unsupported model kind: {model_kind}")
            try:
                choices = await self._choices_for_kind(model_kind)
            except Exception as exc:  # pragma: no cover - defensive logging path
                inventory[model_kind] = []
                errors[model_kind] = str(exc)
                continue

            if search_lower:
                choices = [item for item in choices if search_lower in item.lower()]
            inventory[model_kind] = sorted(dict.fromkeys(choices))

        summary: dict[str, Any] = {
            "base_url": self.api_base_url,
            "kinds": kinds,
            "counts": {kind: len(items) for kind, items in inventory.items()},
            "retrieved_at": datetime.now(tz=timezone.utc).isoformat(),
            "models": inventory,
        }
        if errors:
            summary["errors"] = errors

        if kind and not recursive:
            return inventory.get(kind, [])
        return summary

    async def _choices_for_kind(self, kind: str) -> list[str]:
        client = self._client
        if client is None:
            raise RuntimeError("Model client not initialised")

        node_class, input_name = MODEL_KIND_MAP[kind]
        url = f"{self.api_base_url}/object_info/{node_class}"
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        input_block = payload.get("input", {}) or {}
        field = (
            input_block.get("required", {}).get(input_name)
            or input_block.get("properties", {}).get(input_name)
            or {}
        )
        choices = field.get("choices") or field.get("items") or []
        if isinstance(choices, dict) and "enum" in choices:
            choices = choices["enum"]
        return [item for item in choices if isinstance(item, str)]

def create_server(config: ServerConfig) -> FastMCP:
    """Build the configured FastMCP server."""

    workflow_repo = WorkflowRepository(config.workflow_dir)
    model_client = ComfyUIModelClient(config.api_base_url, timeout=config.http_timeout)

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
            await context.info(f"Found {len(entries)} workflow files")
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
            await context.info(f"Read workflow {relative_path}")
        return {
            "relative_path": relative_path,
            "text": text,
            "json": parsed,
        }

    @server.tool(
        name="list_models",
        description=(
            "List models exposed by the configured ComfyUI server using the object"
            " info endpoints. Specify a kind to get a flat list or use recursive"
            " mode to aggregate multiple kinds."
        ),
    )
    async def list_models(
        kind: str | None = None,
        recursive: bool = False,
        search: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any] | list[str]:
        """List models available on the remote ComfyUI instance."""

        result = await model_client.list_models(kind=kind, recursive=recursive, search=search)
        if context is not None:
            if isinstance(result, dict):
                counts = result.get("counts", {})
                await context.info(
                    "Model inventory retrieved",
                    data={"kinds": list(counts.keys()), "counts": counts},
                )
            else:
                await context.info(
                    "Model inventory retrieved",
                    data={"kind": kind, "count": len(result)},
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
        help=(
            "[Deprecated] Explicit URL for the legacy ComfyUI models endpoint. "
            "Prefer configuring --api-base-url instead."
        ),
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
