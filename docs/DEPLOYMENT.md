# Deploying AutoBanner

## Images

```bash
docker build -t autobanner backend                          # lite, ~740 MB, CPU only
docker build -t autobanner:ai --build-arg WITH_AI=true backend  # + CLIP/LaMa, several GB
```

The image runs as an unprivileged user (`uid 10001`), works with a read-only root
filesystem (give it a writable `/tmp`), exposes port `7860` and has a `HEALTHCHECK`
on `/healthz`.

```bash
docker run -d -p 7860:7860 --read-only --tmpfs /tmp:size=2g \
  -e AUTOBANNER_API_KEYS="$(openssl rand -hex 24)" autobanner
```

`docker compose up --build` does the same with settings from `.env`
(see `.env.example`).

## Sizing

- Rendering is CPU-bound. `AUTOBANNER_WORKERS` ≈ number of vCPUs (default 2).
- Memory: a 1080×1920 render from a 1200×628 source peaks at ~180 MB (Relayout) /
  ~300 MB (Redesign) above a ~50 MB baseline; large layered PSDs need more. The
  compose file caps the container at 4 GB.
- Typical latency on one vCPU: Relayout 0.1–1 s per size; Redesign (phase3) 1–14 s per
  size depending on output area.
- Job results (images + ZIP) are written to `AUTOBANNER_DATA_DIR` (`/data` in the image,
  a named volume in compose) and deleted after `AUTOBANNER_JOB_TTL_SECONDS` (default
  1 h); only job metadata stays in memory. Job ids are local to a replica: run one
  replica, or use a sticky load balancer when scaling horizontally with async jobs. `/v1/render` (synchronous) is stateless and scales
  behind any load balancer.

## Behind a reverse proxy

- Terminate TLS at the proxy and forward to `:7860`.
- Set `AUTOBANNER_TRUST_PROXY_HEADERS=true` so rate limiting uses the right-most
  `X-Forwarded-For` entry (the address your proxy appended). Only enable it when a
  proxy is always in front of the service.
- On PaaS platforms that inject `PORT` (Cloud Run, Railway, Heroku) the server binds to
  it automatically.
- Raise the proxy body limit to match `AUTOBANNER_MAX_UPLOAD_MB` (nginx:
  `client_max_body_size 150m;`) and the read timeout for large synchronous renders
  (e.g. `proxy_read_timeout 300s;`), or use `/v1/jobs`.

## Security checklist

- [ ] `AUTOBANNER_API_KEYS` set (long random values; rotate by listing old+new, then drop old).
- [ ] TLS in front of the service.
- [ ] `AUTOBANNER_CORS_ORIGINS` limited to your own front-ends (empty = no CORS).
- [ ] `AUTOBANNER_ENABLE_DOCS=false` if you do not want the schema public.
- [ ] Container run read-only, non-root, with memory limits.

Built in: API key, rate-limit and size checks run before the upload body is read,
magic-byte validation of uploads, PIL decompression-bomb guard, dimension and
byte limits, sanitised output filenames, constant-time key comparison, per-key job
isolation, strict CSP on the studio, `nosniff`/`DENY` frame headers.

## Observability

- `GET /healthz` — liveness; `GET /readyz` — `503` when the queue is saturated.
- `GET /metrics` (Prometheus): `autobanner_http_requests_total{method,route,status}`,
  `autobanner_http_request_duration_seconds`, `autobanner_job_duration_seconds`,
  `autobanner_jobs_submitted_total`, `autobanner_jobs_finished_total{status}`,
  `autobanner_jobs_pending`. Requires an API key when auth is enabled.
- `AUTOBANNER_LOG_FORMAT=json` emits one JSON object per line; every request is logged
  with its `X-Request-ID`.
