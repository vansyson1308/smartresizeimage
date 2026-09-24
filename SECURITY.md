# Security Policy

## Reporting a vulnerability
Please do **not** open public issues for security vulnerabilities.

Instead, report privately to project maintainers (or security contact once published)
with:
- affected component/file
- reproduction steps
- impact assessment
- optional patch suggestion

## Secrets policy
- Do not commit credentials, tokens, API keys, or private data.
- Use environment variables and local `.env` files (ignored by git).

## Deployment hardening
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#security-checklist). In short: always set
`AUTOBANNER_API_KEYS` when the service is reachable from outside a trusted network,
terminate TLS in front of it, and run the container read-only and non-root.

## Built-in protections
- Upload content must match its extension (magic bytes); size, dimension and pixel-count
  limits guard against decompression bombs and memory exhaustion.
- Output filenames are sanitised; manual anchors are validated and clipped.
- API keys are compared as SHA-256 digests in constant time; jobs are visible only to
  the key that created them.
- The studio is served with a strict Content-Security-Policy and never renders
  user-supplied strings as HTML.
