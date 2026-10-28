import hashlib
from zipfile import ZipFile

import pytest

from scripts.verify_registry import verify


def candidate(tmp_path):
    wheel = tmp_path / "example-0.1.0rc1-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("example-0.1.0rc1.dist-info/METADATA", "Name: example\nVersion: 0.1.0rc1\n")
    source = tmp_path / "example-0.1.0rc1.tar.gz"
    source.write_bytes(b"reviewed source distribution")
    published = {
        "info": {"version": "0.1.0rc1"},
        "urls": [
            {"filename": path.name, "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            for path in (wheel, source)
        ],
    }
    return published


def test_verifies_both_exact_reviewed_distributions(tmp_path):
    published = candidate(tmp_path)
    urls = []

    def fetch(url):
        urls.append(url)
        return published

    assert verify(tmp_path, "testpypi", fetch)[0]["version"] == "0.1.0rc1"
    assert urls == ["https://test.pypi.org/pypi/example/0.1.0rc1/json"]


@pytest.mark.parametrize("failure", ["hash", "missing", "duplicate", "version"])
def test_rejects_different_incomplete_or_ambiguous_registry_release(tmp_path, failure):
    published = candidate(tmp_path)
    if failure == "hash":
        published["urls"][0]["digests"]["sha256"] = "0" * 64
    elif failure == "missing":
        published["urls"].pop()
    elif failure == "duplicate":
        published["urls"].append(published["urls"][0])
    else:
        published["info"]["version"] = "0.1.0"
    with pytest.raises(ValueError):
        verify(tmp_path, "pypi", lambda _: published)


def test_requires_local_source_archive(tmp_path):
    published = candidate(tmp_path)
    (tmp_path / "example-0.1.0rc1.tar.gz").unlink()
    with pytest.raises(FileNotFoundError):
        verify(tmp_path, "pypi", lambda _: published)


def test_requires_reviewed_artifacts(tmp_path):
    with pytest.raises(ValueError, match="No reviewed wheels"):
        verify(tmp_path, "pypi")
