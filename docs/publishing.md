# Release guide

This repository publishes `langchain-antfly` and `langgraph-store-antfly` as
separate distributions on PyPI. The same packages are installable with pip, uv
or Poetry.

## Prepare a release

1. Update package versions and the memory package's dependency on the LangChain
   package. For prereleases, use PEP 440 versions such as `0.1.0rc1`; a requirement
   of `>=0.1.0` does not accept that candidate.
2. Update the changelog and verify the documented server compatibility.
3. Run the checks in [Testing and compatibility](validation.md), including a full
   live profile against the target server. Verify the multimodal and agent recipes.
4. Review changes, merge to `main` and confirm the hosted contract/wheel matrix.

## Build without uploading

The manual **Build release candidate (no publishing)** workflow builds both
wheels and source distributions, checks their descriptions with Twine, and
uploads the files with checksums and their source commit. It has no registry
upload step.

The equivalent local build uses a fresh output directory:

```bash
uv build --no-sources --package langchain-antfly --out-dir .state/candidate/langchain-antfly
uv build --no-sources --package langgraph-store-antfly --out-dir .state/candidate/langgraph-store-antfly
uvx 'twine>=7,<8' check --strict .state/candidate/*/*.whl .state/candidate/*/*.tar.gz
```

## Publish reviewed artifacts

Run **Publish Python packages** from `main`, selecting `testpypi` first and
`pypi` after validation.

The workflow checks source contracts and clean-wheel installations, builds the
release files once, and publishes those exact artifacts: LangChain first, then
memory. Only the upload jobs receive an OIDC identity.

Publishing requires matching [Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
and GitHub environments named `testpypi` and `pypi`. Each environment requires
reviewers and restricts deployment to `main`. The person starting a run may also
approve it. The workflow
fails before building if these protections are absent; it never creates an
unprotected substitute. Publisher configuration must identify this repository,
`publish.yml` and the matching environment. Production and test registries have
separate configuration.

Before approving an upload, review the source commit, validation artifacts,
distribution metadata and checksums. The workflow verifies registry digests and
a fresh registry installation after publishing. TestPyPI installations resolve
ordinary dependencies from PyPI and fetch only the two exact candidates from
TestPyPI, avoiding mixed-index dependency resolution.

Uploaded versions cannot be overwritten. Inspect a partial upload before retrying;
rerun failed jobs to reuse the successful build rather than rebuilding a version
already partly published. The workflow does not silently skip existing files.

## Verify and announce

Compare the registry files against the reviewed artifacts:

```bash
uv run python scripts/verify_registry.py .state/candidate --registry testpypi
uv run python scripts/verify_registry.py .state/candidate --registry pypi
```

After verification, attach the exact distributions and checksums to
package-specific GitHub releases, using tags such as
`langchain-antfly/v0.1.0rc1` and `langgraph-store-antfly/v0.1.0rc1`.
Update release availability and documentation links.

Keep the supported dependency matrix current, run the weekly upstream canary,
and investigate compatibility regressions before expanding supported ranges.
