# Contributing

Bug reports, documentation improvements and focused pull requests are welcome.
Use [GitHub issues](https://github.com/antflydb/langchain-antfly/issues) for
reproducible bugs and feature requests. Include package/server versions,
a minimal example and expected versus observed behavior; omit credentials.

## Development

Fork the repository, create a branch and install the reproducible environment:

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run pytest -q
```

The default contracts require no running server or cloud credentials. See
[Testing and compatibility](docs/validation.md) for dedicated live-model runs,
clean-wheel checks and the supported dependency matrix.

For documentation changes, verify local links and run the relevant example.
For package-description changes, build both distributions and run strict Twine
checks. Keep the README, package pages and examples consistent.

## Pull requests

Describe the problem and resulting behavior, include relevant validation, and
keep changes focused. New behavior should have meaningful contract coverage.
Preserve client ownership, explicit table provisioning, namespace/source scope,
safe serialization and honest write outcomes. Do not add implicit paid calls,
model downloads or automatic retries of ambiguous writes.

Report security-sensitive issues through the [security policy](SECURITY.md).
Maintainer release steps are documented in the [release guide](docs/publishing.md).
