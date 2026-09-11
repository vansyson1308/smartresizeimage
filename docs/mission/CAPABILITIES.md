# Capability status

Statuses: `PLANNED` (not implemented) · `IMPLEMENTED` (code exists, verification incomplete) ·
`VERIFIED_LOCAL` (passed stated local end-to-end checks) · `VERIFIED_CONNECTED` (passed against
a real external integration) · `VERIFIED_HUMAN` (passed stated human evaluation) ·
`BLOCKED_EXTERNAL` (specific external prerequisite missing) · `EXPERIMENTAL` (not part of the
supported promise).

Separate tracks: **Engineering**, **Quality**, **Operations**, **Competitive**, **Commercial**.

## Engineering

| Capability | Status | Evidence |
|---|---|---|
| PSD import (pixel/type/shape/group layers, opacity, blend mode, drop shadow subset) | IMPLEMENTED | `backend/app/parser/psd_parser.py`; unit tests with synthetic layers only, no real customer PSD verified. |
| Flat PNG/JPG/WEBP import | VERIFIED_LOCAL (decomposed with confidence) | `ImageParser` + `decompose_flat_image`: recovered text/mark/subject elements with `provenance.origin = recovered`; variants stay `needs_review` (`recovered_unconfirmed`) until roles are confirmed or text converted. |
| Semantic role classification (name rules -> optional CLIP -> heuristics) | IMPLEMENTED | CLIP path never exercised here (torch not installed). Rules/heuristics unit-tested. |
| Template + adaptive layout (Phase 2.1) | VERIFIED_LOCAL | Fixed 2026-09-09: background excluded from solver, zone members stacked, side-by-side rhythm, zone overflow, raster text uniform scale. `backend/tests/test_layout_fixes.py`. |
| Phase 3 target-first redesign (procedural background regeneration) | IMPLEMENTED | Deterministic procedural generator only. `GenerativeFillAdapter` is a visible mock (`is_mock: true`, `provider: none`). Selection status (`valid/degraded/last_resort`) now propagates to `used_fallback`. |
| Quality contract v2 (rendered-output + structural checks) | VERIFIED_LOCAL | `backend/app/quality`; 13 tests incl. preserved false positive. OCR via tesseract 5.3.4 when installed, else `NOT_CHECKED`. |
| Production/benchmark parity | VERIFIED_LOCAL | Bench runs `ReLayoutEngine.load_elements()` + production `relayout*`; config, seed, candidates, commit, environment recorded in `summary.json`. |
| Reproducible Phase 3 seeds across processes | VERIFIED_LOCAL | `test_phase3_generator_seed_is_stable_across_processes` (two PYTHONHASHSEED values). |
| Gradio UI with per-session state (legacy) | IMPLEMENTED | `gr.State` engine per session; verdict summary per variant. Not exercised in a browser here. |
| Typed design representation (native text, constraints, provenance, confidence) | VERIFIED_LOCAL | `backend/app/design/document.py`, schema 1.0 with migration guard; `test_design_document.py`. |
| Native text fitting/rendering with font disclosure | VERIFIED_LOCAL | FreeType + raqm; missing fonts reported `substituted`/`missing`; Vietnamese diacritics rendered in tests. |
| Mixed style runs (weight, size ratio, colour, tracking per run) through import, editing, wrapping, rendering, serialization, typography and export disclosure | VERIFIED_LOCAL | `text_render.py` per-run shaping on one baseline; PSD style runs extracted; `TextContent.edit_text`, `set_text` with `runs`; `test_text_runs.py` (12). Underline/strikethrough/baseline shift not rendered, disclosed in import notes. |
| Variant pipeline: plan → typeset → render → verify → bounded repair | VERIFIED_LOCAL | `design/variant.py`; bench `--mode design` 36/36 accepted (synthetic, constraint planner + contrast-aware plates). |
| Constraint-aware planner (layout families, reading order, hierarchy, clear space, content pressure) | VERIFIED_LOCAL | `design/planner.py`, `test_planner.py`; `Config.DESIGN_PLANNER`. |
| Approved translations per locale + glyph-coverage fallback with disclosure | VERIFIED_LOCAL | `TextContent.translations`, `FontRegistry.resolve_for_text`; tests incl. CJK fallback to IPAGothic in this environment. |
| Contrast-aware text plates (light/dark panel from text colour) | VERIFIED_LOCAL | busy-background fixtures read at OCR agreement 1.0 after plating. |
| Flat-image decomposition (OCR text cut-outs, GrabCut subject, inpainted background) with confidence, "convert to editable text" correction path | VERIFIED_LOCAL (synthetic banner) | `design/decompose.py`, `test_decompose.py`; recovered elements stay raster with `recovered_text` metadata and force `needs_review` until confirmed. Not validated on real photography. |
| Project persistence / reopen / round trip (history, undo, restore) | VERIFIED_LOCAL | `design/project.py`; API test exports project zip, re-imports, edits, regenerates. |
| Typed API + jobs (progress, cancel, idempotency, partial completion) | VERIFIED_LOCAL | `api/service.py`, `api/jobs.py`; `test_api.py` (8 tests). Jobs are in-process; restart marks running variants failed. |
| Web review/edit UI (select, move, resize, nudge, text/style edit, rules, approve/reject, undo, export) | VERIFIED_LOCAL | Committed Playwright journey (`backend/tests/e2e`, CI job) against the real server with local auth: 34 steps, no console/HTTP errors; 900px layout without horizontal scroll. |
| Exports: PNG/JPEG/WebP zip with manifest + quality reports + font disclosure; multi-page PDF (one variant per page, 150 dpi) in the same zip; editable project zip | VERIFIED_LOCAL | `service.export_deliverables` (`format=pdf`), `export_project`; API test counts PDF pages. Print-ready CMYK/bleed PLANNED. |
| Campaign table: content rows × formats in one job (12 × 6 = 72 measured), rows from a pasted CSV/TSV mapped onto text elements, per-row consistency checks, review filtered by row, one export folder per row | VERIFIED_LOCAL | `rows` on `POST /api/projects/{id}/variants`, `POST .../campaign/rows`; `test_campaign.py` (4), browser journey section 9b; `results/campaign_12x6_2026-09-11.md` (72/72 rendered in 88 s, 65 accepted, 5 needs_review, 2 failed for copy that does not fit MREC). |
| Brand profile (palette, headline/body fonts, logo clear space and minimum size, minimum text size, tone, never-list) per workspace; proposed as soft rules in the brand's projects; palette and font checks on every variant | VERIFIED_LOCAL | `design/brand.py` (`normalize_profile`, `propose_profile_rules`, `brand_profile_checks`), `/api/brands/{brand}`, UI brand card; `test_brand_profile.py` (3), browser journey step. Colour tolerance 24/255 per channel; fonts matched by family prefix. |
| Plan entitlements per workspace (variants/day, projects, members, campaign rows/job, storage) shown in usage and refused with 402; operator assignment with the setup token or env | VERIFIED_LOCAL | `Entitlements` in `api/service.py`, `/api/plans`, workspace card; `test_entitlement.py` (3). Local record only, no billing provider. |
| Offline after install: outbound connections refused at the socket level while the full browser journey passes | VERIFIED_LOCAL | `api/offline.py`, `AUTOBANNER_OFFLINE=1` in the e2e server; `test_offline_guard.py` (3). |
| Restart resume: variant jobs cut short by a restart continue from the stored briefs and layout families (`resumed_from`/`resumed_by`) | VERIFIED_LOCAL | `ProjectService._recover_interrupted`, `AUTOBANNER_RESUME_JOBS`; `test_resume_jobs.py` (3). Single process; no multi-node queue. |
| Copy overrides per run and per variant; locale tag; verbatim (protected) copy | VERIFIED_LOCAL | `VariantBrief.text_overrides`; protected text cannot be overridden from the UI. Locale-aware fitting rules PLANNED. |
| Channel presets with recorded provenance (`verified=false`) | IMPLEMENTED | `api/presets.py`; no platform policy claimed. |
| Compose master from separated assets (blank canvas + add text/image) | VERIFIED_LOCAL | `POST /api/projects/blank`, `POST /api/projects/{id}/elements`; exercised by the browser journey. |
| Learning from approved examples (H1): approved variants → per-orientation family, hierarchy scale, soft constraint proposals with confidence; used by the next generation | VERIFIED_LOCAL (product path) / EXPERIMENTAL (effect) | `design/examples.py`, `GET /api/projects/{id}/learned`, rules panel; `test_examples.py`, API test; ablation in `EXPERIMENTS.md` (agreement 0.24 → 0.85 on held-out sizes, synthetic examples only). |
| Joint family planning across a size set (H2) with cross-variant consistency checks | VERIFIED_LOCAL | `planner.choose_families`, `quality/family.py`; ablation in `EXPERIMENTS.md`. |
| Local edits on revision (H3): regenerate keeps the previous plan; copy/asset/style/hide changes stay inside the edited element's box; `layout_change` when copy no longer fits | VERIFIED_LOCAL | `variant.generate_variant(reference=...)`, `keep_layout` on `POST .../regenerate`, detail-view checkbox; `test_local_edits.py` (out-of-scope pixel diff 0 on 5 edit kinds), API test; corpus measurement in `EXPERIMENTS.md`. |
| Rules from correction history (H5): rejection + approved fix → reviewable proposal (`scale_range`, size-scaled `min_text_size`, `clear_space`) with evidence; one-click add; unparseable reasons reported | VERIFIED_LOCAL (mechanism) / EXPERIMENTAL (effect) | `design/corrections.py`, `project.record_rejection`, `GET .../learned` (`corrections`, `unresolved_corrections`), `POST .../learned/corrections/{i}/apply`; `test_corrections.py`, API test; rule-reviewer measurement in `EXPERIMENTS.md`. Confirmed rules (user or correction-derived) also carry across the owner's projects of the same brand as reviewable proposals (`design/brand.py`, `GET /api/brands/{brand}/rules`; unconfirmed proposals never propagate). |
| Real generative provider adapter | BLOCKED_EXTERNAL | No provider credentials/authorization in this environment; mock is labelled. |

## Quality

| Item | Status | Evidence |
|---|---|---|
| Evaluator detects known-bad output (occlusion, clipping, dropped element, size mismatch) | VERIFIED_LOCAL | `backend/tests/test_quality_contract.py` |
| Evaluator calibration vs human judgement | BLOCKED_EXTERNAL | No human reviewers available. |
| Synthetic 12-case benchmark (36 runs per mode) with contract v2 | VERIFIED_LOCAL | See `EVALUATION.md` for the numbers and configuration. Synthetic fixtures; not customer validation. |
| Ablations with frozen holdout (planner / repair / plates) | VERIFIED_LOCAL | `run_ablations.py`; `results/ablations_2026-09-09.md`. Wide intervals (n=36/12). |
| Pilot instruments (first-pass acceptance, time to decision, corrections) | VERIFIED_LOCAL | `api/events.py`, `GET /api/pilot/summary`; local log only, no blinded reviewers yet. |
| Real design corpus (~30 masters, >= 5 brands) | BLOCKED_EXTERNAL | No licensed/owner-authorized designs in the repository. |
| First-pass human acceptance >= 80% | PLANNED (target, untested) | — |
| Operator-time reduction >= 70% vs baseline | PLANNED (target, untested) | — |

## Operations

| Item | Status |
|---|---|
| Authentication (API keys → owners), per-owner project/job scoping, roles within an owner (viewer / editor / approver / admin) | VERIFIED_LOCAL | `AUTOBANNER_API_KEYS=key:owner:role`; foreign projects/jobs read as 404, disallowed actions 403; `test_ownership_and_jobs.py`. No SSO. |
| Local user journey for the UI: first-run setup token, PBKDF2 passwords, HttpOnly session cookie (also for images/downloads), CSRF header on cookie writes, login throttling, member management, personal API tokens, second isolated workspace, approver with a different account | VERIFIED_LOCAL | `api/auth.py`, `AUTOBANNER_AUTH=local` (default once a user exists); `test_auth_journey.py` (7), browser journey. Sessions/tokens stored as SHA-256 digests. |
| Import safety: archive-controlled paths/ids validated at import and use, member cap, symlink refusal, forged owner/approval metadata neutralised | VERIFIED_LOCAL | `design/project.py` `safe_relative_path`, `Project.load(strict=True)`; `test_archive_safety.py` (8). |
| Hard constraints evaluated after every transformation (planner, kept reference, repair); unevaluable hard rules block acceptance | VERIFIED_LOCAL | `variant.constraint_checks`; `test_hard_rules.py` (5). |
| Portable tests: committed browser journey, bundled licensed fonts (DejaVu), self-contained fixtures | VERIFIED_LOCAL | `backend/app/fonts/`, `tests/e2e`, CI `browser-journey` job. |
| Rate limiting per owner (token bucket on mutating requests, 429 + `Retry-After`) | VERIFIED_LOCAL | `api/ratelimit.py`, `AUTOBANNER_RATE_LIMIT`; in-memory per process (multi-node needs a shared store, PLANNED). |
| Retention policy (purge projects untouched for N days and trim event logs, at startup and daily) | VERIFIED_LOCAL | `ProjectService.purge_stale_projects`, `EventLog.trim`, `AUTOBANNER_RETENTION_DAYS`; `project_purged` events; no undo (operators export first). |
| Tenant isolation for state/assets/jobs | VERIFIED_LOCAL (single process) | Owner stored in project meta and job records; usage per owner. No per-owner encryption or separate storage. |
| Durable jobs, restart behaviour | VERIFIED_LOCAL | Job records under `data/jobs/`; restart reports `interrupted`; variants marked failed. Multi-node queue PLANNED. |
| Metering + quotas | VERIFIED_LOCAL | `GET /api/usage`; `AUTOBANNER_QUOTA_*` → HTTP 429. |
| Sandbox billing | BLOCKED_EXTERNAL | No billing provider keys; plans/entitlements are a local record (`/api/plans`) a billing integration would drive. |
| Backup/restore, deploy/rollback docs | IMPLEMENTED | `docs/OPERATIONS.md`; Docker image builds unverified in this environment (no Docker daemon). |
| Upload/import safety limits | VERIFIED_LOCAL | Streamed 64 MB cap (early 413), 40 MP pixel cap, archive path/size checks. |

## Commercial

| Item | Status |
|---|---|
| Customer conversations, pilots, willingness to pay | BLOCKED_EXTERNAL — none conducted; no claims. |
| Price hypothesis | PLANNED — scenario only: 200 customers × USD 499/month = USD 1,197,600 ARR before churn/discounts/costs. Not evidence. |

## Competitive reference matrix (documentation-level, 2026-09-09)

Evidence caveat: direct fetches of every vendor page were blocked by this environment's egress
proxy; entries rest on search-snippet text attributed to the official domains (tier
DOC-SNIPPET) except Qwen-Image-Layered (GitHub README fetched directly). All hands-on behaviour
is UNTESTED. No prices are asserted beyond what official snippets stated.

| Reference | Advertised capability | Input assumption | Manual setup noted | Pricing signal | Evidence |
|---|---|---|---|---|---|
| Canva Resize | Resize to multiple sizes; limits ≈ 40×40 to 8000×3125 px; ≤5 new sizes per design, ≤250 outputs per bulk action | Native Canva design | Copy-and-resize to keep original | Paid tiers; counts against AI usage limit | DOC-SNIPPET, UNTESTED |
| Canva Magic Layers (beta) | Flat JPEG/PNG -> editable layers incl. live text; "works best" with graphic/illustrated designs | Single-page JPEG/PNG | Convert other formats first | Not stated | DOC-SNIPPET + third-party, UNTESTED |
| Adobe Firefly bulk actions | Preset batch actions: background remove/replace, Resize (beta, focal point + Generative Expand), crop to marketing presets, colour grade | Flat JPEG/PNG (≤1000 images, ≤100 MB) | Choose preset, set focal point | Paid plans with premium generative features | DOC-SNIPPET, UNTESTED |
| Adobe Brand Intelligence | Brand ontology from guidelines/assets; "Validate" checks layout/typography/policy | Enterprise guidelines + DAM | Services-led onboarding | Contact sales | DOC-SNIPPET, UNTESTED |
| Cloudinary gravity + generative fill | URL-driven smart crop (`g_auto`, faces, custom coords, object priorities); `b_gen_fill` outpainting for ratio changes | Flat raster via URL | Optional custom coordinates | Transformation counts; add-ons | DOC-SNIPPET, UNTESTED |
| CHILI GraFx Smart Crop | Template frames with AI fill positioning, subject alignment, copy fitting, auto-grow, layout presets | Templates authored in GraFx Studio | Template authoring required | Not stated | DOC-SNIPPET, UNTESTED |
| Celtra Creative Automation | Master designs with locked/editable elements, auto-layout rules, content feeds, language/font pairing, preview/comment/approve | Templates built in Celtra | Designers define rules/lock states | Not stated | DOC-SNIPPET, UNTESTED |
| Bannerbear API | Every template layer addressable via API; auto-resize long text; template sets for multiple sizes | Template in Bannerbear editor | Build template, name layers | Credit-based plans (official $ not captured) | DOC-SNIPPET, UNTESTED |
| Qwen-Image-Layered | RGB -> variable RGBA layers, recursive decomposition; open weights (Apache 2.0); needs CUDA GPU | One RGB image (+ optional prompt) | Code-level only | Free weights, own compute | DOC-DIRECT (GitHub), UNTESTED |
| DesignAsCode (arXiv 2602.17690) | Design as HTML/CSS synthesis with plan-implement-reflect; editable text/layout/colour/font; layout retargeting | Generation brief | n/a | n/a | DOC-SNIPPET |
| PosterO (arXiv 2505.07843) | SVG-tree layouts predicted by LLM with intent-aligned examples | Background + intent + examples | n/a | n/a | DOC-SNIPPET |
| iPoster (arXiv 2603.29469) | Graph-enhanced diffusion layout with hard user constraints masked into denoising | Canvas + partial constraints | n/a | n/a | DOC-SNIPPET |

### Decisions per capability group

1. **Import / layer & text recovery / fonts**: implement native text + layered PSD import now
   (Phase B); flat-image decomposition later with honest confidence (`EXPERIMENTAL`); consider
   Qwen-Image-Layered integration only when a GPU and licence review exist.
2. **Multi-size / smart crop / safe areas / copy fitting / localization**: implement now — this is
   the core product; channel presets must carry a source and version.
3. **Product/logo preservation / background expansion**: implement protected-region contract now
   (exists as masks); generative expansion behind a real provider adapter — `BLOCKED_EXTERNAL`.
4. **Brand rules / templates / examples**: implement constraints in the IR now; learning from
   approved examples is research (H1).
5. **Editing / batch review / approvals / versions / undo**: implement now (Phase B) — no
   reference combines per-variant rendered diagnostics with review.
6. **Data-driven variants / exports / integrations / API**: typed API + PNG/JPEG exports in Phase
   B; CSV-driven variants and one integration after the journey works.
7. **Collaboration / isolation / economics**: session isolation done; tenant isolation, metering
   and billing in Phase D.

Gaps identified by the scan that AutoBanner targets: chaining flat/layered master -> constrained
editable program -> size family; explicit protected-region guarantees under generation;
per-variant rendered diagnostics in review; transparent per-family economics. These are
positioning hypotheses until measured.
