"""Regression checks for trustworthy execution evidence."""

import json
import subprocess
import sys

import pytest

from scripts import validate


def test_source_hash_excludes_output_artifacts(tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("source")
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "command.log").write_text("one")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kw: "commit" if "rev-parse" in command else b"source.py\0evidence/command.log\0",
    )
    first = validate.source_state(tmp_path, output)
    (output / "command.log").write_text("two")
    assert validate.source_state(tmp_path, output) == first
    assert list(first["files"]) == ["source.py"]


@pytest.mark.parametrize("failure", ["launch", "missing_junit", "malformed_junit", "source_changed"])
def test_validator_always_records_failed_execution(tmp_path, monkeypatch, failure):
    output = tmp_path / "evidence"
    calls = []

    def source_state(*args):
        calls.append(None)
        return {
            "commit": "base",
            "files": {},
            "tree_sha256": str(len(calls)) if failure == "source_changed" else "same",
        }

    monkeypatch.setattr(validate, "source_state", source_state)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"/checkout")

    def run(command, **kwargs):
        if failure == "launch":
            raise FileNotFoundError("missing executable")
        if failure == "malformed_junit":
            (output / "junit.xml").write_text("<bad")
        elif failure == "source_changed":
            (output / "junit.xml").write_text('<testsuites><testsuite><testcase name="ok"/></testsuite></testsuites>')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["validate", "--output", str(output), "--", "pytest", "-q"])
    with pytest.raises(SystemExit) as caught:
        validate.main()
    assert caught.value.code != 0
    report = json.loads((output / "report.json").read_text())
    assert report["exit_code"] == caught.value.code
    assert (output / "command.log").exists()
    if failure in {"missing_junit", "malformed_junit"}:
        assert report["junit_error"]


def test_comparison_cleanup_needs_no_models(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from examples import evaluate_retrieval

    monkeypatch.delenv("ANTFLY_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("ANTFLY_EMBEDDING_MODEL", raising=False)
    dropped = []

    class Client:
        def list_tables(self):
            return [{"name": "scratch_native"}]

        def drop_table(self, table):
            dropped.append(table)

    @contextmanager
    def client():
        yield Client()

    monkeypatch.setattr(evaluate_retrieval, "client_from_env", client)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--native-table", "scratch_native", "--output", str(tmp_path / "unused.json"), "cleanup"],
    )
    evaluate_retrieval.main()
    assert dropped == ["scratch_native"]


def test_comparison_rejects_native_proxy_alias_before_connecting(monkeypatch, tmp_path):
    from examples import evaluate_retrieval

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--prefix",
            "scope",
            "--native-table",
            "scope_vectors",
            "--output",
            str(tmp_path / "unused.json"),
            "cleanup",
        ],
    )
    monkeypatch.setattr(evaluate_retrieval, "client_from_env", lambda: pytest.fail("must not connect"))
    with pytest.raises(SystemExit) as caught:
        evaluate_retrieval.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("root_kind", ["missing", "file", "nongit"])
def test_validator_rejects_invalid_source_root(tmp_path, monkeypatch, root_kind):
    root = tmp_path / root_kind
    if root_kind == "file":
        root.write_text("not a directory")
    elif root_kind == "nongit":
        root.mkdir()
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["validate", "--root", str(root), "--output", str(output), "--", "pytest"])
    with pytest.raises(SystemExit) as caught:
        validate.main()
    assert caught.value.code == 2
    assert not output.exists()


def test_validator_runs_and_hashes_explicit_root(tmp_path, monkeypatch):
    root = tmp_path / "sdk"
    root.mkdir()
    output = tmp_path / "evidence"
    roots = []
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(root).encode())

    def state(selected, excluded):
        roots.append(selected)
        return {"commit": "sdk", "files": {}, "tree_sha256": "stable"}

    monkeypatch.setattr(validate, "source_state", state)

    def run(command, **kwargs):
        assert kwargs["cwd"] == root
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        sys, "argv", ["validate", "--root", str(root), "--output", str(output), "--", "python", "-c", "pass"]
    )
    with pytest.raises(SystemExit) as caught:
        validate.main()
    assert caught.value.code == 0
    assert roots == [root, root]
    assert json.loads((output / "report.json").read_text())["cwd"] == str(root)
