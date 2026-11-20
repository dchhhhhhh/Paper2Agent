"""Call the authenticated Codex CLI and retain a per-attempt audit record."""

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def invoke_model(*, stage: str, prompt: str, log_dir: Path, model: str | None = None,
                 schema: dict | None = None, timeout: int = 600) -> str:
    """Return the model's last answer. Preserve the request, response and observed usage."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("stage must be a short filesystem-safe name")
    attempt = Path(log_dir).resolve() / f"{stage}-{uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    instruction = (
        "Return your answer directly. Use only the material in this request. "
        "Do not use tools, read local files, browse, or execute commands.\n\n"
    )
    full_prompt = instruction + prompt
    (attempt / "prompt.txt").write_text(full_prompt, encoding="utf-8")
    record = {
        "stage": stage, "attempt_id": attempt.name, "provider": "codex-cli",
        "requested_model": model or "CLI default", "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None, "status": "started", "exit_code": None,
        "usage": None, "cost": None, "billing_note": "The CLI reports tokens, not monetary charges.",
    }
    status_file = attempt / "call.json"
    status_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        exe = shutil.which("codex")
        if not exe:
            raise FileNotFoundError("codex executable was not found on PATH")
        args = [exe, "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                "--json", "--color", "never", "--disable", "shell_tool", "--disable", "unified_exec",
                "--disable", "apps", "--disable", "plugins", "--disable", "multi_agent"]
        if model:
            args.extend(["--model", model])
        if schema is not None:
            schema_path = attempt / "output-schema.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            args.extend(["--output-schema", str(schema_path)])
        # Keep an empty working directory; the transmitted material arrives through stdin.
        with tempfile.TemporaryDirectory(prefix="paper2agent-model-") as isolated:
            result = subprocess.run(args + ["-"], input=full_prompt.encode("utf-8"),
                                    capture_output=True, cwd=isolated, timeout=timeout, check=False)
        record["exit_code"] = result.returncode
        replies, errors, unexpected_items = [], [], set()
        stderr = result.stderr.decode("utf-8", "replace")
        (attempt / "stderr.log").write_text(stderr, encoding="utf-8")
        for line in result.stdout.decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "thread.started":
                record["thread_id"] = event.get("thread_id")
            elif kind == "turn.completed":
                record["usage"] = event.get("usage")
            elif kind in {"error", "turn.failed"}:
                errors.append(str(event.get("message") or event.get("error")))
            elif kind in {"item.started", "item.completed"}:
                item = event.get("item", {})
                if item.get("type") not in {"agent_message", "reasoning"}:
                    unexpected_items.add(str(item.get("type")))
                elif item["type"] == "agent_message" and kind == "item.completed":
                    replies.append(item.get("text", ""))
        answer = replies[-1] if replies else ""
        (attempt / "response.txt").write_text(answer, encoding="utf-8")
        if unexpected_items:
            raise RuntimeError(f"Model attempted a tool: {sorted(unexpected_items)}")
        if result.returncode or not answer:
            detail = "; ".join(errors[-2:]) or stderr[-500:] or "No final answer"
            raise RuntimeError(f"Codex call failed: {detail}")
        if schema is not None:
            json.loads(answer)
        record["status"] = "completed"
        return answer
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
