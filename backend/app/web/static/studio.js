/* AutoBanner Studio - a thin client over the public REST API (/v1/*). */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const KEY_STORAGE = "autobanner.apiKey";

  const state = {
    config: null,
    presets: [],
    packs: {},
    selected: new Set(),
    custom: [],
    file: null,
    analysis: null,
    job: null,
    pollTimer: null,
    blobUrls: [],
    analyzeSeq: 0,
  };

  /* ---------- tiny DOM helper (never uses innerHTML for data) ---------- */
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  /* ---------- API ---------- */
  function apiKey() {
    try { return localStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; }
  }

  async function api(path, opts = {}) {
    const headers = new Headers(opts.headers || {});
    const key = apiKey();
    if (key) headers.set("X-API-Key", key);
    const res = await fetch(path, { ...opts, headers });
    if (res.status === 401 && state.config && state.config.auth_required) {
      openKeyDialog();
    }
    if (!res.ok) {
      let message = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (body && body.error && body.error.message) message = body.error.message;
      } catch { /* non-JSON body */ }
      const err = new Error(message);
      err.status = res.status;
      throw err;
    }
    return res;
  }

  /* ---------- UI helpers ---------- */
  function showError(message) {
    const box = $("error");
    box.textContent = message;
    box.hidden = !message;
  }

  function ratioIcon(w, h) {
    const max = 16;
    const s = max / Math.max(w, h);
    return el("span", {
      class: "ratio",
      style: `width:${Math.max(3, Math.round(w * s))}px;height:${Math.max(3, Math.round(h * s))}px`,
      "aria-hidden": "true",
    });
  }

  function fmtKb(kb) {
    return kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb} KB`;
  }

  const PLATFORM_LABELS = {
    iab: "IAB display", google: "Google Ads", meta: "Meta", tiktok: "TikTok",
    linkedin: "LinkedIn", x: "X (Twitter)", youtube: "YouTube", pinterest: "Pinterest",
    web: "Web", email: "Email",
  };

  /* ---------- selection ---------- */
  function targetCount() {
    return state.selected.size + state.custom.length;
  }

  function refreshSelection() {
    $("selected-count").textContent = `${targetCount()} selected`;
    for (const input of document.querySelectorAll(".preset-item input")) {
      input.checked = state.selected.has(input.value);
    }
    for (const chip of document.querySelectorAll("#packs .chip")) {
      const ids = state.packs[chip.dataset.pack].presets;
      chip.setAttribute("aria-pressed", String(ids.every((id) => state.selected.has(id))));
    }
    refreshGenerate();
  }

  function refreshGenerate() {
    const btn = $("generate");
    const busy = state.job && ["queued", "running"].includes(state.job.status);
    const limit = state.config ? state.config.limits.max_targets_per_job : 60;
    let hint = "";
    if (!state.file) hint = "Upload a design to begin.";
    else if (!state.analysis) hint = "Analyzing design…";
    else if (targetCount() === 0) hint = "Choose at least one output size.";
    else if (targetCount() > limit) hint = `At most ${limit} sizes per job.`;
    else if (busy) hint = "Rendering…";
    else hint = `Ready to render ${targetCount()} size${targetCount() === 1 ? "" : "s"}.`;
    btn.disabled = !state.analysis || targetCount() === 0 || targetCount() > limit || busy;
    btn.textContent = busy ? "Rendering…" : "Generate";
    $("form-hint").textContent = hint;
  }

  function renderPacks() {
    const box = $("packs");
    box.replaceChildren();
    for (const [id, pack] of Object.entries(state.packs)) {
      box.append(el("button", {
        type: "button", class: "chip", "data-pack": id, "aria-pressed": "false",
        title: pack.description,
        onclick: () => {
          const all = pack.presets.every((p) => state.selected.has(p));
          pack.presets.forEach((p) => (all ? state.selected.delete(p) : state.selected.add(p)));
          refreshSelection();
        },
      }, id.replace(/-/g, " ")));
    }
  }

  function renderPresets() {
    const q = $("preset-filter").value.trim().toLowerCase();
    const groups = new Map();
    for (const p of state.presets) {
      const hay = `${p.name} ${p.platform} ${p.id} ${p.width}x${p.height}`.toLowerCase();
      if (q && !hay.includes(q)) continue;
      if (!groups.has(p.platform)) groups.set(p.platform, []);
      groups.get(p.platform).push(p);
    }
    const list = $("preset-list");
    list.replaceChildren();
    if (groups.size === 0) {
      list.append(el("p", { class: "muted small", style: "padding:0.5rem 0.75rem" }, "No presets match."));
      return;
    }
    for (const [platform, items] of groups) {
      const group = el("div", { class: "preset-group" },
        el("h3", { text: PLATFORM_LABELS[platform] || platform }));
      for (const p of items) {
        const input = el("input", {
          type: "checkbox", value: p.id,
          onchange: (e) => {
            e.target.checked ? state.selected.add(p.id) : state.selected.delete(p.id);
            refreshSelection();
          },
        });
        input.checked = state.selected.has(p.id);
        const extra = [p.max_kb ? `≤ ${fmtKb(p.max_kb)}` : "", p.safe_zone && (p.safe_zone.top || p.safe_zone.bottom) ? "safe zone" : ""]
          .filter(Boolean).join(" · ");
        group.append(el("label", { class: "preset-item", title: extra || null },
          input, ratioIcon(p.width, p.height), el("span", { text: p.name }),
          el("span", { class: "dims", text: `${p.width}×${p.height}` })));
      }
      list.append(group);
    }
  }

  function renderCustom() {
    const box = $("custom-list");
    box.replaceChildren(...state.custom.map((size, i) => el("button", {
      type: "button", class: "chip", "aria-pressed": "true", "aria-label": `Remove ${size}`,
      onclick: () => { state.custom.splice(i, 1); renderCustom(); refreshSelection(); },
    }, size, el("span", { class: "x", "aria-hidden": "true" }, "×"))));
  }

  /* ---------- upload & analyze ---------- */
  async function handleFile(file) {
    if (!file) return;
    // Only the latest selection may update state (responses can arrive out of order).
    const seq = ++state.analyzeSeq;
    state.file = file;
    state.analysis = null;
    showError("");
    const preview = $("source-preview");
    const isRaster = /\.(png|jpe?g|webp)$/i.test(file.name);
    if (preview.src) URL.revokeObjectURL(preview.src);
    if (isRaster) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      $("dropzone-empty").hidden = true;
    } else {
      preview.hidden = true;
      $("dropzone-empty").hidden = false;
      $("dropzone-empty").replaceChildren(el("strong", { text: file.name }),
        el("span", { class: "muted small", text: "Layered PSD" }));
    }
    refreshGenerate();

    $("auto-layers-field").hidden = !isRaster;
    await analyze(file, seq);
  }

  async function analyze(file, seq) {
    const body = new FormData();
    body.append("file", file);
    const auto = !$("auto-layers-field").hidden && $("auto-layers").checked;
    try {
      const res = await api(`/v1/analyze?auto_layers=${auto}`, { method: "POST", body });
      const analysis = await res.json();
      if (seq !== state.analyzeSeq) return;
      state.analysis = analysis;
      renderSourceInfo();
    } catch (err) {
      if (seq !== state.analyzeSeq) return;
      state.file = null;
      showError(`Could not read this file: ${err.message}`);
    }
    refreshGenerate();
  }

  function renderSourceInfo() {
    const a = state.analysis;
    const info = $("source-info");
    const roles = {};
    for (const e of a.elements) roles[e.role] = (roles[e.role] || 0) + 1;
    const roleText = Object.entries(roles).map(([r, n]) => `${r}×${n}`).join(", ");
    info.replaceChildren(
      el("dt", { text: "File" }), el("dd", { text: a.file }),
      el("dt", { text: "Size" }), el("dd", { text: `${a.width}×${a.height}px` }),
      el("dt", { text: "Type" }), el("dd", { text: {
        flat_image: "Flat image",
        auto_layers: `Flat image · ${a.layers - 1} elements detected`,
      }[a.source_type] || `Layered (${a.layers} layers)` }),
      el("dt", { text: "Roles" }), el("dd", { text: roleText || "-" }),
    );
    info.hidden = false;
    $("anchor-field").hidden = a.source_type !== "flat_image";
  }

  /* ---------- render job ---------- */
  function collectOptions() {
    const maxKb = parseInt($("max-kb").value, 10);
    const opts = {
      presets: [...state.selected],
      sizes: [...state.custom],
      mode: document.querySelector('input[name="mode"]:checked').value,
      format: $("format").value,
      respect_platform_limits: $("platform-limits").checked,
      enforce_safe_zones: $("safe-zones").checked,
    };
    if (Number.isFinite(maxKb) && maxKb > 0) opts.max_kb = maxKb;
    if (!$("anchor-field").hidden) opts.anchor_preset = $("anchor-preset").value;
    if (!$("auto-layers-field").hidden) opts.auto_layers = $("auto-layers").checked;
    return opts;
  }

  function clearResults() {
    state.blobUrls.forEach((u) => URL.revokeObjectURL(u));
    state.blobUrls = [];
    $("gallery").replaceChildren();
    $("download-all").hidden = true;
    $("summary").textContent = "";
  }

  async function generate() {
    showError("");
    clearResults();
    $("empty").hidden = true;
    const body = new FormData();
    body.append("file", state.file);
    body.append("options", JSON.stringify(collectOptions()));
    try {
      const res = await api("/v1/jobs", { method: "POST", body });
      state.job = await res.json();
      renderSkeletons(state.job.progress.total);
      updateProgress();
      poll();
    } catch (err) {
      showError(err.message);
      $("empty").hidden = $("gallery").children.length > 0;
    }
    refreshGenerate();
  }

  function renderSkeletons(n) {
    $("gallery").replaceChildren(...Array.from({ length: n }, () =>
      el("li", { class: "card", "aria-hidden": "true" },
        el("div", { class: "card-media skeleton" }),
        el("div", { class: "card-body" }, el("span", { class: "muted small", text: "Queued" })))));
  }

  function updateProgress() {
    const job = state.job;
    const box = $("progress");
    if (!job || !["queued", "running"].includes(job.status)) { box.hidden = true; return; }
    box.hidden = false;
    const { done, total, current } = job.progress;
    $("progress-fill").style.width = `${total ? Math.round((done / total) * 100) : 0}%`;
    $("progress-text").textContent = job.status === "queued"
      ? "Waiting for a render slot…"
      : `Rendering ${Math.min(done + 1, total)} of ${total}${current && current !== "done" ? ` - ${current}` : ""}`;
  }

  function poll() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      try {
        const res = await api(`/v1/jobs/${encodeURIComponent(state.job.id)}`);
        state.job = await res.json();
      } catch (err) {
        showError(err.message);
        state.job = null;
        updateProgress();
        refreshGenerate();
        return;
      }
      updateProgress();
      if (state.job.status === "succeeded") await renderResults();
      else if (state.job.status === "failed") {
        showError(state.job.error ? state.job.error.message : "Render failed");
        $("gallery").replaceChildren();
        $("empty").hidden = false;
      } else poll();
      refreshGenerate();
    }, 700);
  }

  async function fetchBlobUrl(path) {
    const res = await api(path);
    const url = URL.createObjectURL(await res.blob());
    state.blobUrls.push(url);
    return url;
  }

  async function renderResults() {
    const job = state.job;
    const m = job.manifest;
    const s = m.summary;
    $("summary").textContent =
      `${s.succeeded}/${s.total} sizes in ${(m.duration_ms / 1000).toFixed(1)}s` +
      (s.with_warnings ? ` · ${s.with_warnings} with warnings` : "") +
      (s.failed ? ` · ${s.failed} failed` : "");

    const cards = m.assets.map((a) => buildCard(a));
    $("gallery").replaceChildren(...cards.map((c) => c.node));

    $("download-all").hidden = s.succeeded === 0;

    await Promise.all(cards.map(async ({ asset, img, link }) => {
      if (!img) return;
      try {
        const url = await fetchBlobUrl(job.links.assets[asset.file]);
        img.src = url;
        link.href = url;
        link.hidden = false;
      } catch { img.alt = "Preview unavailable"; }
    }));
  }

  function buildCard(a) {
    const title = el("div", { class: "card-title" },
      el("span", { text: a.name }), el("span", { class: "dims", text: `${a.width}×${a.height}` }));
    if (a.status !== "ok") {
      return {
        asset: a,
        node: el("li", { class: "card" },
          el("div", { class: "card-media" }, el("span", { class: "badge err", text: "Failed" })),
          el("div", { class: "card-body" }, title, el("p", { class: "small", text: a.error || "Unknown error" }))),
      };
    }
    const ex = a.export;
    const budget = ex.max_kb
      ? el("span", { class: `badge ${ex.within_budget ? "ok" : "warn"}`, text: `${fmtKb(ex.size_kb)} / ${fmtKb(ex.max_kb)}` })
      : el("span", { class: "badge", text: fmtKb(ex.size_kb) });
    const img = el("img", { alt: `${a.name} ${a.width}×${a.height} render`, loading: "lazy" });
    const link = el("a", { class: "btn", download: a.file, hidden: true }, "Download");
    const meta = el("div", { class: "meta" }, budget, el("span", { text: ex.format.toUpperCase() }),
      ex.quality ? el("span", { text: `q${ex.quality}` }) : null,
      a.used_fallback ? el("span", { class: "badge warn", text: "fallback" }) : null);
    const warnings = a.warnings.length
      ? el("ul", { class: "warnings" }, ...a.warnings.map((w) => el("li", { text: w })))
      : null;
    return {
      asset: a, img, link,
      node: el("li", { class: "card" },
        el("div", { class: "card-media" }, img),
        el("div", { class: "card-body" }, title, meta, warnings, el("div", { class: "card-actions" }, link))),
    };
  }

  // The ZIP is fetched only when asked for: most users preview first, and the
  // individual assets are already downloaded for the gallery.
  async function downloadZip() {
    const job = state.job;
    if (!job || !job.links) return;
    const btn = $("download-all");
    btn.disabled = true;
    try {
      const url = await fetchBlobUrl(job.links.download);
      const a = el("a", {
        href: url,
        download: `${(job.manifest.source.file || "design").replace(/\.[^.]+$/, "")}_autobanner.zip`,
      });
      document.body.append(a);
      a.click();
      a.remove();
    } catch (err) {
      showError(`ZIP unavailable: ${err.message}`);
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------- API key dialog ---------- */
  function openKeyDialog() {
    const dlg = $("key-dialog");
    if (dlg.open) return;
    $("key-input").value = apiKey();
    dlg.showModal();
  }

  /* ---------- boot ---------- */
  async function boot() {
    try {
      state.config = await (await fetch("/v1/config")).json();
      $("version").textContent = `v${state.config.version}`;
      $("api-key-btn").hidden = !state.config.auth_required;
      const catalog = await (await fetch("/v1/presets")).json();
      state.presets = catalog.presets;
      state.packs = catalog.packs;
      (catalog.packs.starter ? catalog.packs.starter.presets : []).forEach((p) => state.selected.add(p));
    } catch (err) {
      showError(`Cannot reach the AutoBanner server: ${err.message}`);
    }
    renderPacks();
    renderPresets();
    refreshSelection();
    if (state.config && state.config.auth_required && !apiKey()) openKeyDialog();
  }

  /* ---------- events ---------- */
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => handleFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("is-over")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });
  $("preset-filter").addEventListener("input", renderPresets);
  $("custom-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("custom-size");
    const m = input.value.match(/^\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*$/);
    if (!m) { input.setCustomValidity("Use WIDTHxHEIGHT, e.g. 1200x628"); input.reportValidity(); return; }
    input.setCustomValidity("");
    const size = `${+m[1]}x${+m[2]}`;
    if (!state.custom.includes(size)) state.custom.push(size);
    input.value = "";
    renderCustom();
    refreshSelection();
  });
  $("custom-size").addEventListener("input", (e) => e.target.setCustomValidity(""));
  $("generate").addEventListener("click", generate);
  $("auto-layers").addEventListener("change", () => {
    if (!state.file) return;
    state.analysis = null;
    refreshGenerate();
    analyze(state.file, ++state.analyzeSeq);
  });
  $("download-all").addEventListener("click", downloadZip);
  $("api-key-btn").addEventListener("click", openKeyDialog);
  $("key-dialog").addEventListener("close", () => {
    if ($("key-dialog").returnValue === "save") {
      try { localStorage.setItem(KEY_STORAGE, $("key-input").value.trim()); } catch { /* storage blocked */ }
    }
  });

  boot();
})();
