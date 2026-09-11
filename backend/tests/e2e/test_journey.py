"""Browser journey against the real server with local authentication (release gate).

Runs the shipped UI in Chromium through the whole product loop: first-run setup of a
workspace administrator, blank canvas, elements, direct manipulation, rules, brief,
generation, review with approve/reject/regenerate, learned rules, export, save and
reopen, member management, sign-in as an approver with a different account, a second
isolated workspace, and a 900 px layout without horizontal scroll. Console errors and
failed HTTP responses fail the test.

Requirements: ``playwright`` (Python package) and a Chromium build. The Chromium path
is taken from ``PLAYWRIGHT_CHROMIUM``, then ``/opt/pw-browsers/chromium``, then
Playwright's own install (``python -m playwright install chromium``). Without them the
test is skipped with a reason; CI runs it in the ``browser-journey`` job.

Set ``AUTOBANNER_E2E_SHOTS=/some/dir`` to keep the screenshots and the step report.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

REPO = Path(__file__).resolve().parents[3]
SETUP_TOKEN = "e2e-setup-token"
XRW = {"X-Requested-With": "e2e"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _chromium_path() -> str | None:
    for cand in (os.environ.get("PLAYWRIGHT_CHROMIUM"), "/opt/pw-browsers/chromium"):
        if cand and Path(cand).exists():
            return cand
    return None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    data = tmp_path_factory.mktemp("e2e-data")
    port = _free_port()
    env = {
        **os.environ,
        "AUTOBANNER_AUTH": "local",
        "AUTOBANNER_SETUP_TOKEN": SETUP_TOKEN,
        "AUTOBANNER_PBKDF2_ITERATIONS": "2000",
        "AUTOBANNER_DATA_DIR": str(data),
        "AUTOBANNER_JOB_WORKERS": "1",
        "AUTOBANNER_LOG_LEVEL": "WARNING",
        "AUTOBANNER_OFFLINE": "1",  # the whole journey runs with outbound network refused
        "OMP_THREAD_LIMIT": "1",
    }
    env.pop("AUTOBANNER_API_KEYS", None)
    env.pop("AUTOBANNER_API_KEY", None)
    env.pop("AUTOBANNER_RATE_LIMIT", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.api.server:app", "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.skip("uvicorn is not installed or failed to start")
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=1) as res:
                if res.status == 200:
                    ready = True
                    break
        except OSError:
            time.sleep(0.3)
    if not ready:
        proc.terminate()
        pytest.skip("server did not become ready")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _logo_png(path: Path) -> None:
    img = Image.new("RGBA", (300, 140), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 289, 129), outline=(30, 30, 30, 255), width=6)
    d.text((60, 55), "ACME LOGO", fill=(20, 20, 20, 255))
    img.save(path)


def _hero_png(path: Path) -> None:
    img = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 490, 490), fill=(236, 88, 88, 255))
    d.ellipse((150, 150, 350, 350), fill=(255, 220, 120, 255))
    img.save(path)


class Journey:
    def __init__(self, page, base: str, shots: Path):
        self.page, self.base, self.shots = page, base, shots
        self.steps: list[dict] = []
        self.errors: list[str] = []
        # Responses the journey provokes on purpose (a refused setup token).
        self.expected = {(403, "/api/auth/setup")}
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("console", self._console)
        page.on("response", self._response)

    def _response(self, r) -> None:
        if r.status >= 400 and not any(
            r.status == st and r.url.endswith(path) for st, path in self.expected
        ):
            self.errors.append(f"HTTP {r.status} {r.url}")

    def _console(self, m) -> None:
        if m.type != "error":
            return
        if any(f"status of {st}" in m.text for st, _ in self.expected):
            return  # the browser's own log line for an expected response
        self.errors.append(m.text)

    def step(self, name: str, ok: bool = True, **info) -> None:
        self.steps.append({"step": name, "ok": bool(ok), **info})
        assert ok, f"{name}: {info}"

    def shot(self, name: str, full: bool = False) -> None:
        self.page.screenshot(path=str(self.shots / f"{name}.png"), full_page=full)

    def get(self, path: str) -> dict:
        res = self.page.request.get(self.base + path, headers=XRW)
        assert res.ok, f"GET {path}: {res.status} {res.text()}"
        return res.json()

    def post(self, path: str, **kw):
        return self.page.request.post(self.base + path, headers=XRW, **kw)

    def wait_until(self, cond, timeout: float = 20) -> bool:
        """Poll ``cond`` (a callable) until it is truthy or ``timeout`` seconds pass."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return True
            time.sleep(0.4)
        return bool(cond())

    def wait_jobs(self, pid: str, timeout: float = 240) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            variants = self.get(f"/api/projects/{pid}/variants")["variants"]
            if variants and all(v["status"] in ("done", "failed", "cancelled") for v in variants):
                return variants
            time.sleep(1.0)
        raise AssertionError("variants did not finish in time")

    def sign_out(self) -> None:
        self.page.click("#btn-logout")
        self.page.wait_for_selector("#view-auth:not(.hidden)")
        self.page.wait_for_selector("#auth-login:not(.hidden)")

    def sign_in(self, username: str, password: str) -> None:
        self.page.fill("#login-username", username)
        self.page.fill("#login-password", password)
        self.page.click("#login-submit")
        # the UI returns to the project recorded in the URL hash, or to the list
        self.page.wait_for_selector("#view-auth.hidden", state="attached")
        self.page.wait_for_selector("#user-chip:not(.hidden)")
        self.page.click("header nav button[data-view='projects']")
        self.page.wait_for_selector("#view-projects:not(.hidden)")


def test_full_journey_with_local_auth(server: str, tmp_path: Path) -> None:
    shots = Path(os.environ.get("AUTOBANNER_E2E_SHOTS") or (tmp_path / "shots"))
    shots.mkdir(parents=True, exist_ok=True)
    logo, hero = tmp_path / "logo.png", tmp_path / "hero.png"
    _logo_png(logo)
    _hero_png(hero)

    with sync_playwright() as p:
        try:
            exe = _chromium_path()
            browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - environment without a browser
            pytest.skip(f"chromium not available: {exc}")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page = ctx.new_page()
        j = Journey(page, server, shots)
        try:
            _run(j, page, server, logo, hero)
        finally:
            (shots / "journey_report.json").write_text(
                json.dumps({"steps": j.steps, "errors": j.errors}, indent=2)
            )
            browser.close()


def _run(j: Journey, page, base: str, logo: Path, hero: Path) -> None:
    # 0) first run: create the workspace administrator through the UI
    page.goto(base + "/")
    page.wait_for_selector("#view-auth:not(.hidden)")
    page.wait_for_selector("#auth-setup:not(.hidden)")
    j.shot("00_setup")
    page.fill("#setup-workspace", "acme")
    page.fill("#setup-username", "ada")
    page.fill("#setup-password", "correct horse")
    page.fill("#setup-token", "wrong-token")
    page.click("#setup-submit")
    page.wait_for_selector("#auth-error:not(.hidden)")
    j.step("setup refuses a wrong setup token", "token" in page.inner_text("#auth-error"))
    page.fill("#setup-token", SETUP_TOKEN)
    page.click("#setup-submit")
    page.wait_for_selector("#view-projects:not(.hidden)")
    page.wait_for_selector("#user-chip:not(.hidden)")
    j.step("administrator created and signed in",
           "ada" in page.inner_text("#user-name") and page.inner_text("#user-role") == "admin")
    j.step("open-mode badge hidden", page.is_hidden("#open-badge"))
    j.shot("01_projects")

    # 1) blank canvas, text and logo through the UI
    page.fill("#new-name", "Playwright Campaign")
    page.fill("#new-brand", "Acme")
    page.fill("#blank-w", "1200")
    page.fill("#blank-h", "628")
    page.fill("#blank-bg", "#1f3b63")
    page.click("#blank-create")
    page.wait_for_selector("#view-design:not(.hidden)")
    page.wait_for_selector("#layer-list li")
    j.step("create blank canvas via UI")
    page.once("dialog", lambda d: d.accept("SUMMER SUPER SALE"))
    page.click("#add-text")
    page.wait_for_function("document.querySelectorAll('#layer-list li').length >= 2")
    j.step("add text via UI")
    page.set_input_files("#add-image-input", str(logo))
    page.wait_for_function("document.querySelectorAll('#layer-list li').length >= 3")
    j.step("add logo image via UI")

    projects = j.get("/api/projects")["projects"]
    pid = projects[0]["id"]

    def add(kind, name, role, x, y, w, h, text=None, style=None, file=None):
        data = {"kind": kind, "name": name, "role": role, "x": str(x), "y": str(y),
                "width": str(w), "height": str(h)}
        if text:
            data["text"] = text
        if style:
            data["style"] = json.dumps(style)
        multipart = dict(data)
        if file:
            multipart["file"] = {"name": Path(file).name, "mimeType": "image/png",
                                 "buffer": Path(file).read_bytes()}
        r = j.post(f"/api/projects/{pid}/elements", multipart=multipart)
        assert r.ok, r.text()

    add("text", "Subheadline", "subheadline", 72, 200, 560, 70,
        text="Up to 50% off selected items", style={"font_size": 34, "color": "#e8eef7"})
    add("text", "CTA", "cta", 72, 320, 300, 70, text="SHOP NOW",
        style={"font_size": 36, "weight": "bold", "color": "#ffd166"})
    add("text", "Price", "label", 72, 420, 300, 50, text="$19.99",
        style={"font_size": 32, "color": "#ffffff"})
    add("image", "Hero", "hero_image", 720, 80, 440, 440, file=str(hero))
    page.reload()
    page.wait_for_selector("#layer-list li")
    page.wait_for_function("document.querySelectorAll('#layer-list li').length >= 7")
    j.step("session survives a reload; remaining elements added",
           page.is_visible("#user-chip"), count=len(page.query_selector_all("#layer-list li")))

    # 2) element panel, drag, nudge, undo, rule
    page.click("#layer-list li:has-text('Price')")
    page.wait_for_selector("#element-panel:not(.hidden)")
    page.check("#el-protected")
    page.click("#el-apply")
    page.wait_for_timeout(600)
    doc = j.get(f"/api/projects/{pid}")["document"]
    price = next(e for e in doc["elements"] if e["name"] == "Price")
    j.step("mark price verbatim via panel", price["text"]["protected"] is True)

    # run-level editing: split the price into two styled runs from the panel
    original = page.input_value("#el-text")
    page.check("#el-runs-mode")
    page.wait_for_selector("#el-runs:not(.hidden) .run-row")
    cut = max(1, len(original) // 2)
    page.fill("#el-runs .run-row:nth-child(1) [data-rtext]", original[:cut])
    page.click("#el-run-add")
    page.fill("#el-runs .run-row:nth-child(2) [data-rtext]", original[cut:])
    page.select_option("#el-runs .run-row:nth-child(2) [data-rweight]", "bold")
    page.fill("#el-runs .run-row:nth-child(2) [data-rcolor]", "#d62828")
    page.click("#el-apply")
    page.wait_for_timeout(600)
    doc = j.get(f"/api/projects/{pid}")["document"]
    price = next(e for e in doc["elements"] if e["name"] == "Price")
    runs = price["text"]["runs"]
    page.click("#layer-list li:has-text('SUMMER')")
    page.click("#layer-list li:has-text('Price')")
    note = page.inner_text("#el-runs-note")
    j.step("split price into two styled runs via panel",
           len(runs) == 2 and "".join(r["text"] for r in runs) == original
           and runs[1]["style"]["weight"] == "bold" and runs[1]["style"]["color"] == "#d62828"
           and runs[0]["style"]["color"] != "#d62828" and price["text"]["protected"] is True
           and "2 styled runs" in note and page.is_checked("#el-runs-mode"),
           runs=[(r["text"], r["style"]["weight"], r["style"]["color"]) for r in runs], note=note)

    page.click("#layer-list li:has-text('SUMMER')")
    page.wait_for_selector("rect.el-box.selected")
    box = None
    for _ in range(10):  # the overlay is redrawn when the preview image (re)loads
        el = page.query_selector("rect.el-box.selected")
        box = el.bounding_box() if el else None
        if box:
            break
        page.wait_for_timeout(300)
    assert box is not None
    before = next(e for e in doc["elements"] if e["role"] == "headline")["geometry"]
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 60, box["y"] + box["height"] / 2 + 30, steps=8)
    page.mouse.up()
    page.wait_for_timeout(800)
    doc = j.get(f"/api/projects/{pid}")["document"]
    after = next(e for e in doc["elements"] if e["role"] == "headline")["geometry"]
    j.step("drag headline on canvas", after["x"] > before["x"] and after["y"] > before["y"])
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(600)
    doc = j.get(f"/api/projects/{pid}")["document"]
    nudged = next(e for e in doc["elements"] if e["role"] == "headline")["geometry"]
    j.step("nudge with arrow key", nudged["x"] == after["x"] + 1)
    page.click("#btn-undo")
    page.wait_for_timeout(800)
    doc = j.get(f"/api/projects/{pid}")["document"]
    undone = next(e for e in doc["elements"] if e["role"] == "headline")["geometry"]
    j.step("undo via button", undone["x"] == after["x"])
    page.select_option("#c-type", "allowed_overlap")
    opts = page.eval_on_selector_all("#c-a option", "els => els.map(e => e.value)")
    page.select_option("#c-a", opts[0])
    page.select_option("#c-b", opts[1])
    page.click("#c-add")
    page.wait_for_timeout(800)
    doc = j.get(f"/api/projects/{pid}")["document"]
    j.step("add rule via UI", any(c["type"] == "allowed_overlap" for c in doc["constraints"]))
    j.shot("02_design", full=True)

    # 3) brief and generation
    page.click("#btn-to-brief")
    page.wait_for_selector("#view-brief:not(.hidden)")
    page.wait_for_selector("#preset-grid input")
    page.fill("#custom-w", "300")
    page.fill("#custom-h", "250")
    page.fill("#custom-name", "MREC")
    page.click("#custom-add")
    over = page.query_selector("#copy-overrides input[data-override]")
    if over:
        over.fill("WINTER SUPER SALE")
    page.fill("#brief-locale", "en")
    j.shot("03_brief", full=True)
    page.click("#btn-generate")
    page.wait_for_selector("#view-review:not(.hidden)", timeout=240000)
    page.wait_for_selector(".variant-card", timeout=60000)
    j.step("generate variants and land on review",
           count=len(page.query_selector_all(".variant-card")))
    j.shot("04_review", full=True)

    # 4) review: approve, reject with reason, per-variant override
    page.click(".variant-card [data-open]")
    page.wait_for_selector("#variant-detail:not(.hidden)")
    page.wait_for_selector("#detail-issues li")
    page.check("#detail-compare")
    j.shot("05_detail", full=True)
    page.click("#detail-approve")
    j.step("approve via detail", j.wait_until(lambda: any(
        v["approval"] == "approved" for v in j.get(f"/api/projects/{pid}/variants")["variants"]
    )))
    page.click("#detail-close")
    page.click(".variant-card:nth-child(2) [data-open]")
    page.wait_for_selector("#variant-detail:not(.hidden)")
    page.click("#detail-reject")
    page.wait_for_timeout(300)  # no reason -> toast only
    page.fill("#detail-reason", "hero too small")
    page.click("#detail-reject")
    j.step("reject with reason via detail", j.wait_until(lambda: any(
        v["approval"] == "rejected" and v["approval_reason"] == "hero too small"
        for v in j.get(f"/api/projects/{pid}/variants")["variants"]
    )))
    # the detail re-renders once the rejection lands; type only after that so the
    # re-render cannot wipe the override
    page.wait_for_selector("#detail-verdict:has-text('hero too small')")
    inp = page.query_selector("#detail-overrides input[data-dov]")
    assert inp is not None
    inp.fill("BUY TODAY")
    page.click("#detail-apply-overrides")
    dov = inp.get_attribute("data-dov")

    def _regenerated() -> bool:
        # the override job is queued asynchronously; wait for it to exist and finish
        variants = j.get(f"/api/projects/{pid}/variants")["variants"]
        regen = [v for v in variants
                 if v["brief"].get("text_overrides", {}).get(dov) == "BUY TODAY"]
        return bool(regen) and all(v["status"] == "done" for v in regen)

    j.step("per-variant override regenerate", j.wait_until(_regenerated, timeout=120))
    j.wait_jobs(pid)
    page.click("#detail-close")

    page.click('header nav button[data-view="design"]')
    page.wait_for_function(
        "document.querySelector('#learned-list').textContent.includes('Learned from')", timeout=8000
    )
    learned = j.get(f"/api/projects/{pid}/learned")
    j.step("learned rules panel after approval", len(learned["families"]) >= 1)
    j.step("rejection recorded as correction evidence", any(
        u["reason"] == "hero too small" for u in learned.get("unresolved_corrections", [])
    ) or any(c["reason"] == "hero too small" for c in learned.get("corrections", [])))

    # 5) export approved, save and reopen the project
    page.click('header nav button[data-view="review"]')
    page.wait_for_selector("#view-review:not(.hidden)")
    with page.expect_download(timeout=60000) as dl:
        page.click("#btn-export-approved")
    with zipfile.ZipFile(dl.value.path()) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    j.step("export approved zip",
           any(n.endswith(".png") for n in names) and len(manifest["variants"]) >= 1)
    page.click("header nav button[data-view='design']")
    page.wait_for_selector("#view-design:not(.hidden)")
    with page.expect_download(timeout=60000) as dl2:
        page.click("#btn-save-project")
    proj_zip = Path(dl2.value.path())
    j.step("save project zip", proj_zip.stat().st_size > 1000)
    n_before = len(j.get("/api/projects")["projects"])
    page.click("header nav button[data-view='projects']")
    page.wait_for_selector("#view-projects:not(.hidden)")
    page.set_input_files("#import-input", str(proj_zip))
    page.wait_for_selector("#view-design:not(.hidden)")
    page.wait_for_function("document.querySelectorAll('#layer-list li').length >= 7")
    j.step("reopen project from zip via UI",
           len(j.get("/api/projects")["projects"]) == n_before + 1)

    # 6) members: add an approver through the workspace card
    page.click("header nav button[data-view='projects']")
    page.wait_for_selector("#workspace-card:not(.hidden)")
    page.wait_for_selector("#members-wrap:not(.hidden)")
    page.fill("#member-username", "reviewer")
    page.fill("#member-password", "review-pass-1")
    page.select_option("#member-role", "approver")
    page.click("#member-add")
    page.wait_for_selector("#member-table td:has-text('reviewer')")
    j.step("add approver member via UI")
    j.step("plan and usage shown in the workspace card",
           page.inner_text("#plan-line").startswith("Plan unlimited"),
           text=page.inner_text("#plan-line"))
    # brand profile card: colours, fonts and logo rules stored per workspace
    page.fill("#brand-name", "Acme")
    page.click("#brand-load")
    page.wait_for_selector("#brand-form:not(.hidden)")
    page.fill("#brand-primary", "#1f3b63")
    page.fill("#brand-font-head", "DejaVu Sans")
    page.fill("#brand-logo-clear", "0.6")
    page.fill("#brand-text-min", "12")
    page.click("#brand-save")
    page.wait_for_selector("#brand-status:has-text('Saved')")
    saved = j.get("/api/brands/Acme")["profile"]
    j.step("brand profile saved via UI",
           saved["colors"]["primary"] == "#1f3b63"
           and saved["fonts"]["headline"]["family"] == "DejaVu Sans"
           and saved["logo"]["clear_space_ratio"] == 0.6 and saved["text"]["min_px"] == 12)
    page.fill("#token-name", "ci")
    page.click("#token-create")
    page.wait_for_selector("#token-new:not(.hidden)")
    token = page.inner_text("#token-new").split("\n")[-1].strip()
    j.step("create personal API token via UI", token.startswith("abt_"))
    j.shot("06_workspace", full=True)

    # 7) sign out, sign in as the approver: review only
    j.sign_out()
    j.step("signed out: UI asks for credentials", page.is_visible("#auth-login"))
    j.sign_in("reviewer", "review-pass-1")
    j.step("approver signed in", page.inner_text("#user-role") == "approver")
    j.step("approver cannot create designs", page.is_disabled("#blank-create"))
    # the reopened copy is listed too: open the original project explicitly
    page.click(f"#project-list .project-tile[data-id='{pid}']")
    page.wait_for_selector("#view-design:not(.hidden)")
    j.step("approver: design tools disabled",
           page.is_disabled("#add-text") and page.is_disabled("#el-apply"))
    page.click("header nav button[data-view='review']")
    page.wait_for_selector(".variant-card")
    j.step("approver: regenerate/delete disabled, approve enabled",
           page.is_disabled(".variant-card [data-regen]") and page.is_enabled("#detail-approve"))
    page.click(".variant-card:nth-child(3) [data-open]")
    page.wait_for_selector("#variant-detail:not(.hidden)")
    page.click("#detail-approve")
    j.step("approver approves with a different account", j.wait_until(lambda: sum(
        1 for v in j.get(f"/api/projects/{pid}/variants")["variants"]
        if v["approval"] == "approved"
    ) >= 2))
    j.shot("07_approver_review", full=True)
    j.sign_out()

    # 8) a second workspace does not see the first one (operator setup via the API)
    r = j.post("/api/auth/setup", data={"username": "grace", "password": "globex-pass-1",
                                        "workspace": "globex", "setup_token": SETUP_TOKEN})
    assert r.status == 201, r.text()
    j.step("second workspace sees no projects", j.get("/api/projects")["projects"] == [])
    r = page.request.get(f"{base}/api/projects/{pid}/preview.png", headers=XRW)
    j.step("second workspace cannot read the first workspace's design", r.status == 404)
    j.post("/api/auth/logout")

    # 9) the personal token authenticates automation without a cookie
    r = page.request.get(f"{base}/api/projects", headers={"X-API-Key": token})
    j.step("personal token lists the workspace's projects",
           r.ok and len(r.json()["projects"]) == n_before + 1)

    # 9b) campaign table: two rows generated in one size, reviewed per row
    page.reload()
    page.wait_for_selector("#auth-login:not(.hidden)")
    j.sign_in("ada", "correct horse")
    page.click(f"#project-list .project-tile[data-id='{pid}']")
    page.wait_for_selector("#view-design:not(.hidden)")
    page.click("header nav button[data-view='brief']")
    page.wait_for_selector("#view-brief:not(.hidden)")
    page.fill("#campaign-csv", "label,headline,CTA,locale,Price\n"
                               "Week 1,SPRING SALE,SHOP NOW,en,$1\n"
                               "Tuần 2,GIẢM GIÁ XUÂN,MUA NGAY,vi,\n")
    page.click("#campaign-parse")
    page.wait_for_selector("table.campaign")
    j.step("campaign table read from CSV",
           len(page.query_selector_all("table.campaign tr")) == 3
           and "Verbatim" in page.inner_text("#campaign-note"))
    for cb in page.query_selector_all("#preset-grid input:checked"):  # only the custom size
        cb.uncheck()
    page.fill("#custom-w", "300")
    page.fill("#custom-h", "250")
    page.fill("#custom-name", "MREC")
    page.click("#custom-add")
    count = page.inner_text("#campaign-count")
    j.step("campaign count shown", "2 rows × 1 size = 2 variants" in count, text=count)
    n_variants = len(j.get(f"/api/projects/{pid}/variants")["variants"])
    page.select_option("#brief-direction", "text-left")  # creative direction for the run
    page.click("#btn-generate")
    page.wait_for_selector("#view-review:not(.hidden)", timeout=240000)
    variants = j.wait_jobs(pid)
    rows = {v["brief"]["row"]["id"]: v for v in variants if v["brief"].get("row")}
    plans = {k: j.get(f"/api/projects/{pid}/variants/{v['id']}")["plan"] for k, v in rows.items()}
    j.step("creative direction applied to the campaign",
           all(v["brief"].get("direction") == {"text_side": "left"} for v in rows.values())
           and all(p["planner_meta"].get("traits", {}).get("text_side") == "left"
                   for p in plans.values()),
           traits=[p["planner_meta"].get("traits") for p in plans.values()])
    doc = j.get(f"/api/projects/{pid}")["document"]
    cta_id = next(e["id"] for e in doc["elements"] if e["name"] == "CTA")
    j.step("campaign rows generated in the chosen size",
           len(variants) == n_variants + 2 and len(rows) == 2
           and all(v["status"] == "done" and v["name"] == "MREC" for v in rows.values())
           and any(v["brief"]["text_overrides"].get(cta_id) == "MUA NGAY"
                   for v in rows.values()),
           before=n_variants, after=len(variants),
           rows=[(k, v["name"], v["status"], v["error"], v["brief"]["text_overrides"])
                 for k, v in rows.items()])
    page.wait_for_selector("#review-row:not(.hidden)")
    page.select_option("#review-row", "row2")
    page.wait_for_timeout(300)
    j.step("review filters by campaign row",
           len(page.query_selector_all(".variant-card")) == 1
           and "Tuần 2" in page.inner_text(".variant-card .row-tag"))
    page.click(".variant-card [data-open]")
    page.wait_for_selector("#detail-plan:has-text('Direction asked')")
    plan_line = page.inner_text("#detail-plan")
    j.step("detail view shows the layout family and the honoured direction",
           "Layout " in plan_line and "text side left" in plan_line and "honoured" in plan_line
           and "not met" not in plan_line, text=plan_line)
    page.click("#detail-close")
    page.select_option("#review-row", "all")
    j.shot("09_campaign_review", full=True)
    j.sign_out()

    # 10) sign in again as the administrator, small screen, no errors
    page.reload()
    page.wait_for_selector("#auth-login:not(.hidden)")
    j.sign_in("ada", "correct horse")
    page.set_viewport_size({"width": 900, "height": 800})
    page.click("#project-list .project-tile")
    page.wait_for_selector("#view-design:not(.hidden)")
    page.click("header nav button[data-view='review']")
    page.wait_for_selector("#view-review:not(.hidden)")
    page.wait_for_timeout(500)
    j.shot("08_review_900", full=True)
    horiz = page.evaluate(
        "document.documentElement.scrollWidth > document.documentElement.clientWidth + 2"
    )
    j.step("no horizontal scroll at 900px", not horiz)
    j.step("no console/page errors or failed responses", not j.errors, errors=j.errors[:5])
    if os.environ.get("AUTOBANNER_E2E_SHOTS"):
        shutil.copy(logo, j.shots / "logo.png")
