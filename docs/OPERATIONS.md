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
| `AUTOBANNER_DATA_DIR` | `data` | Root for projects, jobs, usage, fonts. One directory = one deployment. |
| `AUTOBANNER_API_KEYS` | unset | `key1:owner-a,key2:owner-b`. Requests must send `X-API-Key`; each owner sees only its own projects/jobs. |
| `AUTOBANNER_API_KEY` | unset | Single-key form (owner `default`). Without any key the server is **open** (owner `local`) — development only. |
| `AUTOBANNER_JOB_WORKERS` | `2` | Concurrent variant jobs. Each variant is CPU-bound (1–4 s on 4 cores). |
| `AUTOBANNER_QUOTA_VARIANTS_PER_DAY` | unlimited | Per-owner daily variant cap → HTTP 429. |
| `AUTOBANNER_QUOTA_PROJECTS` | unlimited | Per-owner project cap → HTTP 429. |
| `AUTOBANNER_QUOTA_STORAGE_BYTES` | unlimited | Per-owner storage cap (checked on project creation). |
| `AUTOBANNER_USE_AI` | `false` | Load CLIP for role classification (needs torch/transformers). |
| `AUTOBANNER_LOG_LEVEL` | `INFO` | Log level (structured lines on stdout). |
| `OMP_THREAD_LIMIT` | set to `1` in Docker | Keeps tesseract from oversubscribing CPUs. |

Fonts: drop `.ttf/.otf` files into `$AUTOBANNER_DATA_DIR/fonts/` (scanned before system
fonts). Missing fonts are substituted and disclosed in every export manifest.

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

Multi-node job queue, billing, SSO, audit trail, retention policies beyond manual delete,
rate limiting per IP. See `docs/mission/CAPABILITIES.md` for status.
