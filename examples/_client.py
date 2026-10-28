"""Shared example connection settings, without optional application dependencies."""

import os

from antfly import AntflyClient


def client_from_env() -> AntflyClient:
    return AntflyClient(
        base_url=os.getenv("ANTFLY_URL", "http://localhost:8080"), token=os.getenv("ANTFLY_TOKEN"), timeout=60
    )
