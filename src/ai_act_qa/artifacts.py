"""Reports and their provenance: which code, configuration, document and questions made them."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime

from ai_act_qa.config import load_config, project_path

TRACKED = ["src", "configs", "eval", "pyproject.toml", "uv.lock"]


def file_sha256(relative: str) -> str:
    return hashlib.sha256(project_path(relative).read_bytes()).hexdigest()


def provenance() -> dict:
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=project_path("."), capture_output=True,
                              text=True)  # fmt: skip
        return done.stdout.strip()

    return {
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git("rev-parse", "--short", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain", "--", *TRACKED)),
        "document_sha256": load_config()["source"]["sha256"],
        "questions_sha256": file_sha256("eval/questions.yaml"),
    }


def write_json(relative: str, data: dict) -> None:
    path = project_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
