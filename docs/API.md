# AutoBanner REST API

Base URL: wherever the server runs (default `http://localhost:7860`). Interactive
OpenAPI reference: `/docs` (Swagger) and `/redoc`.

## Authentication

When `AUTOBANNER_API_KEYS` is set, every `/v1/analyze`, `/v1/render`, `/v1/jobs*` and
`/metrics` call needs a key, sent as either header:

```
X-API-Key: <key>
Authorization: Bearer <key>
```

`/healthz`, `/readyz`, `/v1/config` and `/v1/presets` are always public. Jobs are
private to the key that created them; other keys get `404`.

## Rate limits and back-pressure

- Token bucket of `AUTOBANNER_RATE_LIMIT_PER_MINUTE` requests per key (or client IP when
  auth is off) on analyze/render/job creation → `429` with `Retry-After`.
- At most `AUTOBANNER_WORKERS` renders run at once and `AUTOBANNER_MAX_QUEUE` wait;
  beyond that → `503 overloaded` with `Retry-After`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness (`{"status":"ok","version":…}`) |
| GET | `/readyz` | Readiness; `503` when the render queue is full |
| GET | `/metrics` | Prometheus text metrics |
| GET | `/v1/config` | Version, auth requirement, modes, formats, limits |
| GET | `/v1/presets[?platform=iab]` | Size catalog and packs |
| POST | `/v1/analyze[?auto_layers=true]` | Multipart `file` → detected layers/roles |
| POST | `/v1/render` | Multipart `file` + `options` → `application/zip` |
| POST | `/v1/jobs` | Same inputs → `202` job (async) |
| GET | `/v1/jobs/{id}` | Status, progress, manifest and links when done |
| GET | `/v1/jobs/{id}/download` | ZIP of all outputs + `manifest.json` |
| GET | `/v1/jobs/{id}/assets/{file}` | One output image |
| DELETE | `/v1/jobs/{id}` | Delete a finished job and its files |

Uploads accept PSD, PNG, JPG/JPEG and WEBP. Content must match the extension
(magic-byte check), up to `AUTOBANNER_MAX_UPLOAD_MB` and 12,000 px / 80 MP.

## Render options

Sent as a JSON string in the multipart field `options`. Unknown keys are rejected.

| Field | Type | Default | Notes |
|---|---|---|---|
| `packs` | string[] | `[]` | e.g. `google-display`, `meta-ads`, `all` |
| `presets` | string[] | `[]` | Preset ids from `/v1/presets` |
| `sizes` | string[] | `[]` | Custom `"WIDTHxHEIGHT"`, 10–4096 px |
| `mode` | `phase21` \| `phase3` | `phase21` | Relayout vs. target-first redesign |
| `format` | `png` \| `jpeg` \| `webp` | `png` | |
| `quality` | 1–100 | `90` | Starting quality for JPEG/WebP |
| `max_kb` | int | – | Budget for every file; overrides platform caps |
| `respect_platform_limits` | bool | `true` | Apply each preset's network cap when `max_kb` is unset |
| `enforce_safe_zones` | bool | `true` | Move key elements out of Story/Reels/TikTok UI zones |
| `anchor_preset` | `none` \| `flat_banner_3anchors` | `none` | Default brand anchors for flat images (phase3) |
| `anchors` | object[] | – | Manual anchors: `{id, role, x, y, width, height}` in source pixels |
| `role_overrides` | object | `{}` | `{element_id: role}` corrections from `/v1/analyze` |
| `auto_layers` | bool | `false` | Flat PNG/JPG/WEBP: detect headline, sub-copy, CTA, logo and hero as movable layers (falls back to whole-image fit when the design cannot be separated) |

At least one target is required and at most 60 per job. Packs, presets and sizes are
merged in order with duplicates removed.

## manifest.json

Every ZIP (and every finished job) includes a manifest:

```json
{
  "generator": "autobanner/2.0.0",
  "created_at": "2026-09-24T08:30:00+00:00",
  "mode": "phase21",
  "duration_ms": 1203,
  "source": {"file": "spring.psd", "width": 1200, "height": 628, "sha256": "…",
             "source_type": "layered", "layers": 7},
  "summary": {"total": 8, "succeeded": 8, "failed": 0, "with_warnings": 1},
  "assets": [
    {
      "id": "meta-story", "name": "Story / Reels", "platform": "meta",
      "width": 1080, "height": 1920, "file": "meta-story_1080x1920.jpg",
      "status": "ok", "error": null, "used_fallback": false, "duration_ms": 910,
      "warnings": [],
      "qa": {"evaluated": true, "critical_elements": 3,
             "safe_zone": {"rect": {"x1": 0, "y1": 269, "x2": 1080, "y2": 1536}, "violations": []},
             "small_text": [], "quality_gates_passed": true},
      "export": {"format": "jpeg", "bytes": 91234, "size_kb": 89.1, "quality": 90,
                 "max_kb": 30720, "within_budget": true, "palette_colors": null}
    }
  ]
}
```

`source.source_type` is `layered` (PSD), `flat_image` (PNG/JPG rendered as a whole) or
`auto_layers` (PNG/JPG split into detected elements).

A size that fails does not abort the job: it is reported with `status: "failed"` and an
`error`, and the other sizes are still delivered. `/v1/render` also exposes the counts
in `X-AutoBanner-Succeeded` / `X-AutoBanner-Failed` headers.

## Errors

All errors share one shape and carry the request id (also in the `X-Request-ID`
response header; send your own `X-Request-ID` to correlate logs):

```json
{"error": {"code": "invalid_options", "message": "sizes.0: …", "request_id": "…"}}
```

| Status | `code` | Meaning |
|---|---|---|
| 401 | `unauthorized` | Missing/invalid API key |
| 404 | `not_found` | Unknown or expired job/asset |
| 409 | `not_ready` / `job_active` | Job not finished / cannot delete a running job |
| 413 | `payload_too_large` | Upload over the size limit |
| 415 | `unsupported_format` | Not PSD/PNG/JPG/WEBP |
| 422 | `invalid_options` / `invalid_input` / `invalid_request` | Bad options, spoofed/corrupt file, bad sizes |
| 429 | `rate_limited` | Slow down; see `Retry-After` |
| 503 | `overloaded` | Queue full; see `Retry-After` |

## Examples

Python:

```python
import json, time, requests

BASE, KEY = "http://localhost:7860", "my-key"
H = {"X-API-Key": KEY}
with open("design.psd", "rb") as f:
    job = requests.post(f"{BASE}/v1/jobs", headers=H, files={"file": f},
                        data={"options": json.dumps({"packs": ["google-display"],
                                                      "format": "webp"})}).json()
while job["status"] in ("queued", "running"):
    time.sleep(1)
    job = requests.get(f"{BASE}/v1/jobs/{job['id']}", headers=H).json()
open("out.zip", "wb").write(requests.get(BASE + job["links"]["download"], headers=H).content)
```

JavaScript (browser, same origin or allowed via `AUTOBANNER_CORS_ORIGINS`):

```js
const body = new FormData();
body.append("file", fileInput.files[0]);
body.append("options", JSON.stringify({ packs: ["meta-ads"] }));
const res = await fetch("/v1/render", { method: "POST", body, headers: { "X-API-Key": key } });
const zip = await res.blob();
```
