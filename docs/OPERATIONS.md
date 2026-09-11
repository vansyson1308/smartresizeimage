# Operating AutoBanner

This document covers the API server (`backend/app/api/server.py`) and its web UI. The legacy
Gradio interface (`python -m backend.app.main`) is stateless per session and needs no storage.

## Run

```bash
# local
AUTOBANNER_DATA_DIR=./data uvicorn backend.app.api.server:app --host 0.0.0.0 --port 8000
# docker
docker compose up --build            # serves http://localhost:8000, data in ./data
```

Health: `GET /api/health` (no auth) reports version, environment (Python/Pillow/NumPy/OCR),
auth mode and limits. API docs: `/api/docs`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOBANNER_DATA_DIR` | `data` | Root for projects, jobs, usage, fonts, users. One directory = one deployment. |
| `AUTOBANNER_AUTH` | auto | `local` (users + sessions + personal tokens, the documented deployment mode; chosen automatically once a user exists), `api_key` (static keys only; chosen automatically when keys are set and no user exists) or `open` (no credentials, owner `local`, admin role — development only; the health endpoint and the UI say so). |
| `AUTOBANNER_SETUP_TOKEN` | generated | Operator secret that creates a workspace with its first administrator (`POST /api/auth/setup`, or the UI's first-run screen). When unset, a random token is generated at startup and printed in the log while no user exists. |
| `AUTOBANNER_COOKIE_SECURE` | `false` | Mark the session cookie `Secure`. Set to `true` behind HTTPS. |
| `AUTOBANNER_PBKDF2_ITERATIONS` | `600000` | PBKDF2-HMAC-SHA256 rounds for password hashes (lower only for tests). |
| `AUTOBANNER_API_KEYS` | unset | `key1:owner-a[:role],key2:owner-b[:role]`. Static keys for automation, sent as `X-API-Key`; each owner (workspace) sees only its own projects/jobs. Roles: `viewer` (read), `editor` (edit, generate, export; no approvals), `approver` (approve/reject only), `admin` (all, default). An unknown role fails startup. Also accepted in `local` mode. |
| `AUTOBANNER_RATE_LIMIT` | unset | Per-owner token bucket for mutating requests, e.g. `60/minute`, `600/hour`, `5/10s` → HTTP 429 with `Retry-After`; reads are never limited; hits are logged as `rate_limited` events. In-memory, per process. |
| `AUTOBANNER_RETENTION_DAYS` | unset | Delete projects (variants, history, corrections) untouched for longer than this, and trim event-log lines older than this, at startup and once a day. Logged as `project_purged`. No undo: export a project zip first if you want to keep it. |
| `AUTOBANNER_API_KEY` | unset | Single-key form of `AUTOBANNER_API_KEYS` (owner `default`). |
| `AUTOBANNER_JOB_WORKERS` | `2` | Concurrent variant jobs. Each variant is CPU-bound (1–4 s on 4 cores). |
| `AUTOBANNER_QUOTA_VARIANTS_PER_DAY` | unlimited | Per-owner daily variant cap → HTTP 429. |
| `AUTOBANNER_QUOTA_PROJECTS` | unlimited | Per-owner project cap → HTTP 429. |
| `AUTOBANNER_QUOTA_STORAGE_BYTES` | unlimited | Per-owner storage cap (checked on project creation). |
| `AUTOBANNER_USE_AI` | `false` | Load CLIP for role classification (needs torch/transformers). |
| `AUTOBANNER_LOG_LEVEL` | `INFO` | Log level (structured lines on stdout). |
| `OMP_THREAD_LIMIT` | set to `1` in Docker | Keeps tesseract from oversubscribing CPUs. |

Fonts: drop `.ttf/.otf` files into `$AUTOBANNER_DATA_DIR/fonts/` (scanned before the bundled
DejaVu Sans faces in `backend/app/fonts/` and before system fonts). Missing fonts are
substituted and disclosed in every export manifest.

## Authentication (local mode)

1. First run: the log prints `use this one-time setup token ...` (or set
   `AUTOBANNER_SETUP_TOKEN`). Open the UI, or call `POST /api/auth/setup` with
   `{"workspace": "acme", "username": "ada", "password": "...", "setup_token": "..."}`.
   This creates the workspace's first **admin** and signs the browser in. Repeat with
   another workspace name to create an isolated second workspace (same token).
2. Members: an admin adds users with a role from the UI's Workspace card or
   `POST /api/auth/users {"username","password","role"}`; `PATCH /api/auth/users/{name}`
   changes role/password; `DELETE` removes (never the last admin). Users belong to the
   admin's workspace only.
3. Sessions: `POST /api/auth/login` sets an HttpOnly, SameSite=Lax cookie (12 h idle,
   7 days max); `POST /api/auth/logout` ends it. Cookie-authenticated writes must send an
   `X-Requested-With` header (the UI does; this blocks cross-site form posts). Images and
   downloads use the same cookie, so `<img>` and links just work.
4. Automation: `POST /api/auth/tokens {"name"}` returns a personal token once
   (`abt_...`), used as `X-API-Key` with the user's role; revoke with
   `DELETE /api/auth/tokens/{id}`. Static `AUTOBANNER_API_KEYS` keep working.
5. Throttling: 8 failed logins lock a username for 60 s (in memory, per process).

`auth/users.json` holds password hashes (PBKDF2), and only SHA-256 digests of sessions
and tokens, written with mode 0600. Back it up with the data directory; a copied file
grants no access by itself.

OCR: install `tesseract-ocr` (Docker image includes it). Without it, legibility checks
report `not_checked` and variants land in `needs_review`.

## Data layout

```
$AUTOBANNER_DATA_DIR/
  projects/<proj_id>/project.json      # document + meta (owner, versions)
  projects/<proj_id>/assets/<hash>.png # originals and recovered crops (content-addressed)
  projects/<proj_id>/variants/*.png|json + index.json
  projects/<proj_id>/history/NNNNN.json # document snapshots (undo/restore)
  projects/<proj_id>/source/original.*  # uploaded master
  jobs/job_*.json                        # durable job records
  usage/<owner>.json                     # metering counters
  auth/users.json                        # users (hashed passwords), session/token digests
  fonts/                                 # deployment fonts
```

Everything is plain files. Backup = copy the directory (or export projects as `.zip` via
`GET /api/projects/{id}/export/project`). Restore = copy back, or `POST /api/projects/import`.

## Limits and safety

- Uploads: 64 MB (streamed with an early 413), 40 MP pixel cap, PSD/PNG/JPG/WEBP only,
  archive import checks paths and expanded size.
- Targets: ≤ 48 per job, side ≤ 4096 px.
- Jobs are in-process threads; a restart marks running jobs and variants as `interrupted`/
  `failed` (never silently pending). Re-run them from the UI ("Regenerate").
- Idempotency: send `Idempotency-Key` on `POST /variants` to make retries safe.
- Roles limit what a key may do within an owner (403), rate limits bound write traffic per
  owner (429), retention removes untouched projects; all three are off unless configured.

## Deploy / rollback

- Build: `docker build -t autobanner:<git-sha> backend/`. Tag releases with the commit SHA.
- Deploy: run the new image against the same `data` volume. Project schema is versioned
  (`schema_version` in `project.json`); newer servers read older projects, older servers
  refuse newer schemas with a clear error.
- Rollback: start the previous image tag on the same volume. Nothing in this release
  changes the on-disk format in a way older servers cannot read.

## Monitoring

- `GET /api/health` for liveness; job records under `jobs/` show throughput and failures;
  `GET /api/usage` per owner. Logs are single-line with logger names
  (`autobanner.api.*`, `autobanner.design.*`).
- Error tracking: not wired to a provider. Hook `logging` handlers in
  `backend/app/logging_config.py` if you use Sentry or similar.

## Not yet provided

Multi-node job queue (rate limits and job records are per process), billing, SSO, a full audit
trail beyond the event log, rate limiting per IP (limits are per owner/key). See
`docs/mission/CAPABILITIES.md` for status.
