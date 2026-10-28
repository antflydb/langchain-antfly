"""Upstream embedding contract suites; live execution is explicit."""

import os

import pytest
from antfly import AntflyClient
from langchain_antfly import AntflyEmbeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests


class TestEmbeddingInitialization(EmbeddingsUnitTests):
    @property
    def embeddings_class(self):
        return AntflyEmbeddings

    @property
    def embedding_model_params(self):
        return {"client": object(), "model": "explicit-model"}


@pytest.mark.skipif(
    not (os.getenv("ANTFLY_TEST_URL") and os.getenv("ANTFLY_TEST_EMBEDDING_MODEL")),
    reason="Requires real Antfly and an explicit installed embedding model",
)
class TestLiveEmbeddingContract(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self):
        return AntflyEmbeddings

    @property
    def embedding_model_params(self):
        return {
            "client": AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN")),
            "model": os.environ["ANTFLY_TEST_EMBEDDING_MODEL"],
        }

    @pytest.fixture
    def model(self):
        params = self.embedding_model_params
        try:
            yield AntflyEmbeddings(**params)
        finally:
            params["client"].close()
