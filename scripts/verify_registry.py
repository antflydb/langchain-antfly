"""Verify that a registry serves the exact reviewed wheel and source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile

REGISTRIES = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def verify(directory: Path, registry: str, fetch=fetch_json) -> list[dict]:
    wheels = sorted(directory.rglob("*.whl"))
    if not wheels:
        raise ValueError("No reviewed wheels found")
    results = []
    for wheel in wheels:
        with ZipFile(wheel) as archive:
            metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                raise ValueError(f"Expected one metadata file in {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        name, version = metadata.get("Name"), metadata.get("Version")
        if not name or not version:
            raise ValueError(f"Missing package identity in {wheel.name}")
        source = wheel.with_name(f"{name.replace('-', '_')}-{version}.tar.gz")
        expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, source)}
        url = f"{REGISTRIES[registry]}/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
        published = fetch(url)
        if published.get("info", {}).get("version") != version:
            raise ValueError(f"Registry version mismatch for {name}")
        files = published["urls"]
        actual = {item["filename"]: item["digests"]["sha256"] for item in files}
        if len(actual) != len(files) or actual != expected:
            raise ValueError(f"Registry artifacts differ from reviewed files for {name}=={version}")
        results.append({"package": name, "version": version, "registry": registry, "sha256": expected})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--registry", choices=REGISTRIES, required=True)
    args = parser.parse_args()
    print(json.dumps({"verified": verify(args.directory, args.registry)}, indent=2))


if __name__ == "__main__":
    main()
