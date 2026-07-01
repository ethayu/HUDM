from __future__ import annotations

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mwm.benchmark.review_media import (
    render_rollout_media,
    rollout_by_index,
    rollout_key,
)
from mwm.io import load_json


def resolve_eval_path(root: str | Path, path_text: str) -> Path:
    base = Path(root).resolve()
    raw = Path(str(path_text))
    candidate = raw if raw.is_absolute() else base / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"eval path is outside review root: {path_text}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing eval json: {resolved}")
    if resolved.name != "eval.json":
        raise ValueError(f"expected eval.json path, got {resolved.name!r}")
    return resolved


def _href(path_text: Any, base_dir: Path) -> str:
    path = Path(str(path_text or ""))
    try:
        if path.is_absolute():
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
        return (Path.cwd() / path).resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _media_html(payload: dict[str, Any], episode_index: int, base_dir: Path) -> str:
    entries = payload.get("review_media", {}).get("rollouts", {}).get(rollout_key(episode_index), {})
    if not isinstance(entries, dict) or not entries:
        return "<p class='muted'>No media rendered yet.</p>"
    cards: list[str] = []
    for kind, entry in sorted(entries.items()):
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        href = _href(entry["path"], base_dir)
        label = str(kind).replace("_", " ")
        cards.append(
            "<figure>"
            f"<video controls preload='metadata' src='/{html.escape(href)}'></video>"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    return "".join(cards) or "<p class='muted'>No media rendered yet.</p>"


def rollout_page_html(root: Path, eval_path: Path, episode_index: int) -> str:
    payload = load_json(eval_path)
    rollout = rollout_by_index(payload, episode_index)
    rel_eval = eval_path.resolve().relative_to(root.resolve()).as_posix()
    status = "success" if rollout.get("success") else "failure" if rollout.get("success") is False else "unknown"
    media = _media_html(payload, episode_index, root)
    title = f"{eval_path.parent.name} {rollout_key(episode_index)}"
    env_disabled = "" if rollout.get("action_trace") else " disabled title='requires action_trace'"
    latent_disabled = "" if rollout.get("fidelity_trace") else " disabled title='requires fidelity_trace'"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; color: #172026; background: #f6f8fa; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px; }}
    a {{ color: #0b5cad; text-decoration: none; }}
    button {{ border: 1px solid #9fb3c8; background: #fff; border-radius: 6px; padding: 8px 12px; cursor: pointer; margin-right: 8px; }}
    button:hover {{ background: #eef3f7; }}
    .panel {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin-top: 16px; }}
    .muted {{ color: #627282; }}
    .media {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; }}
    video {{ width: 100%; max-height: 480px; background: #000; }}
    code {{ background: #eef3f7; padding: 2px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f8fafc; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
<main>
  <p><a href="/review.html">Back to benchmark review</a></p>
  <h1>{html.escape(title)}</h1>
  <section class="panel">
    <p><strong>Status:</strong> {html.escape(status)}</p>
    <p><strong>Dataset episode:</strong> {html.escape(str(rollout.get("dataset_episode", "")))}</p>
    <p><strong>Start/goal:</strong> {html.escape(str(rollout.get("start_step", "")))} -> {html.escape(str(rollout.get("goal_step", "")))}</p>
    <p><strong>Eval:</strong> <code>{html.escape(rel_eval)}</code></p>
  </section>
  <section class="panel">
    <button data-source="env"{env_disabled}>Render Env</button>
    <button data-source="latent"{latent_disabled}>Render Latent</button>
    <button data-source="both"{' disabled' if env_disabled and latent_disabled else ''}>Render Both</button>
    <label><input id="force" type="checkbox"> force</label>
    <pre id="result" aria-live="polite"></pre>
  </section>
  <section class="panel media">{media}</section>
</main>
<script>
async function render(source) {{
  const result = document.getElementById('result');
  result.textContent = 'Rendering ' + source + '...';
  const response = await fetch('/api/render-rollout', {{
    method: 'POST',
    headers: {{'content-type': 'application/json'}},
    body: JSON.stringify({{
      eval_path: {json.dumps(rel_eval)},
      episode_index: {int(episode_index)},
      sources: [source],
      force: document.getElementById('force').checked
    }})
  }});
  const payload = await response.json();
  result.textContent = JSON.stringify(payload, null, 2);
  if (response.ok) window.location.reload();
}}
document.querySelectorAll('button[data-source]').forEach((button) => {{
  button.addEventListener('click', () => render(button.dataset.source));
}});
</script>
</body>
</html>
"""


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, root: Path, **kwargs: Any) -> None:
        self.review_root = root.resolve()
        super().__init__(*args, directory=str(self.review_root), **kwargs)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/rollouts/") and path.endswith(".html"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3:
                run_name = parts[1]
                page_name = parts[2]
                try:
                    if not page_name.startswith("episode_"):
                        raise ValueError("not an episode page")
                    episode_index = int(page_name.removeprefix("episode_").removesuffix(".html"))
                    eval_path = resolve_eval_path(self.review_root, f"{run_name}/eval.json")
                    body = rollout_page_html(self.review_root, eval_path, episode_index).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                    return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        if parsed.path != "/api/render-rollout":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            eval_path = resolve_eval_path(self.review_root, str(data.get("eval_path", "")))
            sources = data.get("sources", ["both"])
            if isinstance(sources, str):
                sources = [sources]
            result = render_rollout_media(
                eval_path,
                episode_index=int(data.get("episode_index", 0)),
                sources=sources,
                force=bool(data.get("force", False)),
            )
            self._send_json(HTTPStatus.OK, result)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def serve_review(output_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = Path(output_dir).resolve()
    if str(host) not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--serve is restricted to localhost hosts")

    def handler(*args: Any, **kwargs: Any) -> ReviewRequestHandler:
        return ReviewRequestHandler(*args, root=root, **kwargs)

    server = ThreadingHTTPServer((str(host), int(port)), handler)
    print(f"Serving benchmark review at http://{host}:{int(port)}/review.html")
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["ReviewRequestHandler", "resolve_eval_path", "rollout_page_html", "serve_review"]
