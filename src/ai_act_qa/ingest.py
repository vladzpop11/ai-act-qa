"""Download the AI Act and check it is the pinned version.

python -m ai_act_qa.ingest [--check]
  Downloads the consolidated text (configs/config.yaml, `source`) unless the file is already
  there, then checks its SHA-256. A different file is refused: a new consolidated version means
  new answers and a new evaluation, so it is adopted by changing the pin, not silently.
  --check: only verify the file already present (no network).
"""

import hashlib
import sys

import httpx

from ai_act_qa.config import load_config, project_path


class SourceChanged(Exception):
    """The downloaded document is not the pinned version."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(source: dict) -> bytes:
    for attempt in range(3):
        try:
            response = httpx.get(source["url"], headers=source["headers"], timeout=120,
                                 follow_redirects=True)  # fmt: skip
            response.raise_for_status()
            return response.content
        except httpx.HTTPError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def main(check_only: bool = False) -> None:
    source = load_config()["source"]
    path = project_path(source["path"])
    if not path.exists():
        if check_only:
            raise FileNotFoundError(f"{path} is missing: run `make data`")
        data = download(source)
        if sha256(data) != source["sha256"]:
            raise SourceChanged(
                f"{source['url']} returned a file with SHA-256 {sha256(data)}, not the pinned "
                f"{source['sha256']}: the publisher changed it. Review the new version, then "
                "update source.sha256 (and re-run the evaluation)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"downloaded {path.name} ({len(data):,} bytes)")
    if sha256(path.read_bytes()) != source["sha256"]:
        raise SourceChanged(f"{path} does not match the pinned SHA-256")
    print(f"{path.name}: SHA-256 matches the pin")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv[1:])
