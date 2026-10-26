"""Open-weight models with llama.cpp: download a pinned GGUF file, serve it, and talk to it.

The server is llama.cpp's `llama-server` (an OpenAI-compatible chat API), started either as a
local binary (Kaggle, built for its GPU by scripts/build_llama.sh) or as llama.cpp's official
Docker image (the Mac). Both read the same GGUF file, checked against its SHA-256.
"""

import hashlib
import subprocess
import time
from pathlib import Path

import httpx

DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:server"
HF = "https://huggingface.co"


def download(spec: dict, folder: Path) -> Path:
    """The model file from Hugging Face, verified against its pinned SHA-256 (kept if present)."""
    path = folder / spec["file"]
    if path.exists() and sha256_file(path) == spec["sha256"]:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/{spec['repo']}/resolve/main/{spec['file']}"
    partial = path.with_suffix(".part")
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for block in response.iter_bytes(1 << 20):
                        out.write(block)
                        digest.update(block)
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"{spec['file']}: SHA-256 {digest.hexdigest()} is not the pin")
            partial.rename(path)
            return path
        except httpx.HTTPError:
            if attempt == 2:
                raise
            time.sleep(10)
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LlamaServer:
    """`with LlamaServer(model_path, ...) as url:` serves the model until the block ends.

    mode "binary": `binary` is a llama-server built for this machine; all layers on the GPU(s).
    mode "docker": llama.cpp's image on the CPU, `threads` threads (the Mac has 2 cores).
    """

    def __init__(self, model: Path, mode: str, context: int, port: int = 8090,
                 binary: str | None = None, threads: int = 2) -> None:  # fmt: skip
        self.model, self.mode, self.context, self.port = model, mode, context, port
        self.binary, self.threads = binary, threads
        self.process: subprocess.Popen | None = None
        self.container = f"ai-act-qa-llm-{port}"

    def command(self) -> list[str]:
        common = ["-c", str(self.context), "--port", str(self.port), "--jinja"]
        if self.mode == "binary":
            return [self.binary, "-m", str(self.model), "-ngl", "999", "--host", "127.0.0.1",
                    *common]  # fmt: skip
        return ["docker", "run", "--rm", "--name", self.container,
                "-p", f"127.0.0.1:{self.port}:{self.port}",
                "-v", f"{self.model.parent}:/models:ro", DOCKER_IMAGE,
                "-m", f"/models/{self.model.name}", "-t", str(self.threads), "--host", "0.0.0.0",
                *common]  # fmt: skip

    def __enter__(self) -> str:
        self.log = (self.model.parent / f"server-{self.port}.log").open("w")
        self.process = subprocess.Popen(self.command(), stdout=self.log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 900  # loading a 18 GB model from disk can take minutes
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited: see {self.log.name}")
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    return url
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise TimeoutError("llama-server did not become ready")

    def __exit__(self, *exc) -> None:
        if self.mode == "docker":
            subprocess.run(["docker", "stop", self.container], capture_output=True)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(60)
        self.log.close()


def chat(url: str, messages: list[dict], max_tokens: int, temperature: float = 0.0,
         seed: int = 42, thinking: bool = False, schema: dict | None = None,
         timeout: float = 900) -> dict:  # fmt: skip
    """One chat completion -> {"text", "prompt_tokens", "completion_tokens", "seconds"}."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "grade", "schema": schema}}  # fmt: skip
    started = time.time()
    response = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data["choices"][0]["message"]["content"] or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "seconds": round(time.time() - started, 2),
    }
