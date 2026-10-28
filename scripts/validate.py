"""Record reproducible checks, exact source state and JUnit-derived counts.

Run: python scripts/validate.py --output .state/validation -- pytest -q
Secrets and environment values are deliberately excluded from the report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state(root: Path, excluded: Path | None = None) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    )
    hashes = {
        name: digest(root / name)
        for name in sorted(set(files))
        if name
        and (root / name).is_file()
        and (excluded is None or not (root / name).resolve().is_relative_to(excluded))
    }
    return {
        "commit": commit,
        "files": hashes,
        "tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git checkout or subdirectory to record and run from",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("--root must be an existing directory")
    try:
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError):
        parser.error("--root must be inside a Git checkout")
    output = args.output.resolve()
    if output == root or root.is_relative_to(output):
        parser.error("output must not contain the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("use a fresh empty output directory; refusing stale artifacts")
    junit = output / "junit.xml"
    is_pytest = any(Path(part).name == "pytest" for part in command)
    if is_pytest:
        command.append(f"--junitxml={junit}")
    before = source_state(root, output)
    started = time.time()
    with (output / "command.log").open("w") as log:
        try:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            log.write(f"Could not start command: {exc}\n")
            result = subprocess.CompletedProcess(command, 127)
    counts = None
    junit_error = None
    cases = None
    if junit.exists():
        try:
            cases = ET.parse(junit).getroot().findall(".//testcase")
        except ET.ParseError:
            junit_error = "JUnit artifact is malformed"
    elif is_pytest:
        junit_error = "pytest did not produce its required JUnit artifact"
    if cases is not None:
        counts = {
            "total": len(cases),
            "failed": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases),
        }
        counts["passed"] = counts["total"] - counts["failed"] - counts["errors"] - counts["skipped"]
    after = source_state(root, output)
    exit_code = result.returncode or int(
        bool(junit_error) or before != after or bool(counts and (counts["failed"] or counts["errors"]))
    )
    report = {
        "command": command,
        "cwd": str(root),
        "started_unix": started,
        "duration_seconds": time.time() - started,
        "exit_code": exit_code,
        "command_exit_code": result.returncode,
        "junit_error": junit_error,
        "source_before": before,
        "source_after": after,
        "source_changed_during_check": before != after,
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "counts": counts,
        "artifacts": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("exit_code", "duration_seconds", "counts", "source_changed_during_check")})
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
