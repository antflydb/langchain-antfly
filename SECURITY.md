# Security policy

Security fixes target the latest supported release candidate or stable release.
Report ordinary bugs through GitHub issues.

For a suspected vulnerability, use
[private vulnerability reporting](https://github.com/antflydb/langchain-antfly/security/advisories/new).
Include affected versions, a minimal reproduction and the impact. Do not post
credentials, production data or exploit details in a public issue.

Applications remain responsible for server authentication/authorization and for
deriving user namespaces from trusted identity. Namespace filters are not a
substitute for access control. Keep credentials in environment variables or
your secret manager, not committed code or example configuration.
