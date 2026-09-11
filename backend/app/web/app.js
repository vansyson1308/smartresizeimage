/* AutoBanner web UI — dependency-free. Talks to /api. */
(() => {
  "use strict";

  const state = {
    campaignRows: [], rowSeq: 0, rowFilter: "all",
    view: "projects",
    project: null,        // payload from /api/projects/{id}
    selectedId: null,
    presets: [],
    fonts: [],
    customTargets: [],
    selectedPresets: new Set(["ig_story", "ig_square", "fb_link"]),
    job: null,
    jobTimer: null,
    filter: "all",
    detailId: null,
    previewNonce: 0,
    stageScale: 1,
    drag: null,
    auth: null,           // /api/auth/status payload
    can: { edit: true, approve: true, admin: true },
  };

  const ROLES = ["headline", "subheadline", "body_text", "cta", "badge", "label", "logo", "hero_image",
    "icon", "photo", "illustration", "decoration", "background", "background_pattern", "overlay", "group", "unknown"];
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------- helpers ----------
  function toast(msg, bad) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.toggle("bad", !!bad);
    t.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.add("hidden"), bad ? 6000 : 3000);
  }
  function setStatus(msg) { $("#status-bar").textContent = msg; }
  async function api(path, opts = {}) {
    // Same-origin requests carry the session cookie; the custom header marks them as
    // ours (cookie-authenticated writes without it are refused: CSRF protection).
    const headers = Object.assign({ "X-Requested-With": "fetch" }, opts.headers || {});
    const res = await fetch(path, Object.assign({ credentials: "same-origin" }, opts, { headers }));
    if (res.status === 204) return null;
    const ct = res.headers.get("content-type") || "";
    const body = ct.includes("application/json") ? await res.json() : await res.blob();
    if (!res.ok) {
      if (res.status === 401 && state.auth && state.auth.mode !== "open" && !path.startsWith("/api/auth/")) {
        showAuth(false);
        throw new Error("Sign in required");
      }
      const detail = body && body.detail ? body.detail : res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }
  const json = (obj) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) });
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const verdictBadge = (v) => {
    const map = { accepted: ["ok", "Accepted"], needs_review: ["warn", "Needs review"], failed: ["bad", "Failed"], not_evaluated: ["neutral", "Not checked"] };
    const [cls, label] = map[v] || ["neutral", v];
    return `<span class="badge ${cls}">${label}</span>`;
  };
  const approvalBadge = (a) => a === "approved" ? '<span class="badge ok">Approved</span>' : a === "rejected" ? '<span class="badge bad">Rejected</span>' : "";
  function downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ---------- navigation ----------
  function updateHash() {
    const id = state.project ? state.project.project.id : "";
    const next = id ? `#p=${encodeURIComponent(id)}&view=${state.view}` : "";
    if (location.hash !== next) history.replaceState(null, "", next || location.pathname);
  }
  function parseHash() {
    const m = /p=([^&]+)/.exec(location.hash || "");
    const v = /view=([a-z]+)/.exec(location.hash || "");
    return { id: m ? decodeURIComponent(m[1]) : null, view: v ? v[1] : "design" };
  }
  function showView(name) {
    state.view = name;
    updateHash();
    $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${name}`));
    $$("header nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    if (name === "projects") loadProjects();
    if (name === "design") renderDesign();
    if (name === "brief") renderBrief();
    if (name === "review") renderReview();
  }
  $$("header nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  function enableProjectViews(on) { $$("header nav button").forEach((b) => { if (b.dataset.view !== "projects") b.disabled = !on; }); }

  // ---------- projects ----------
  async function loadProjects() {
    renderWorkspace();
    try {
      const data = await api("/api/projects");
      const list = $("#project-list");
      list.innerHTML = "";
      $("#projects-empty").classList.toggle("hidden", data.projects.length > 0);
      for (const p of data.projects) {
        const el = document.createElement("div");
        el.className = "card project-tile";
        el.dataset.id = p.id;
        el.innerHTML = `<strong>${esc(p.name)}</strong><div class="small muted">${esc(p.brand || "")} · ${p.canvas.width}×${p.canvas.height} · ${p.element_count} elements · ${p.variant_count} variants</div><div class="small muted">Updated ${esc(p.updated_at || "")}</div>`;
        el.addEventListener("click", () => openProject(p.id));
        list.appendChild(el);
      }
    } catch (e) { toast(e.message, true); }
  }
  async function openProject(id, view) {
    setStatus("Opening project…");
    try {
      state.project = await api(`/api/projects/${id}`);
      state.selectedId = null;
      enableProjectViews(true);
      showView(view || "design");
      setStatus(`Opened ${state.project.project.name}`);
    } catch (e) { toast(e.message, true); setStatus("Ready"); }
  }
  async function uploadMaster(file) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const name = $("#new-name").value.trim(); if (name) fd.append("name", name);
    const brand = $("#new-brand").value.trim(); if (brand) fd.append("brand", brand);
    setStatus(`Importing ${file.name}…`);
    try {
      state.project = await api("/api/projects", { method: "POST", body: fd });
      enableProjectViews(true);
      toast("Design imported. Check roles and text before generating.");
      showView("design");
      setStatus(`Opened ${state.project.project.name}`);
    } catch (e) { toast(`Import failed: ${e.message}`, true); setStatus("Ready"); }
  }
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => uploadMaster(e.dataTransfer.files[0]));
  $("#pick-file").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => uploadMaster(e.target.files[0]));
  $("#import-project").addEventListener("click", () => $("#import-input").click());
  $("#import-input").addEventListener("change", async (e) => {
    const file = e.target.files[0]; if (!file) return;
    const fd = new FormData(); fd.append("file", file);
    try {
      state.project = await api("/api/projects/import", { method: "POST", body: fd });
      enableProjectViews(true); toast("Project reopened."); showView("design");
    } catch (err) { toast(err.message, true); }
  });

  $("#blank-create").addEventListener("click", async () => {
    try {
      state.project = await api("/api/projects/blank", json({ name: $("#new-name").value.trim() || "Untitled", brand: $("#new-brand").value.trim() || null, width: Number($("#blank-w").value), height: Number($("#blank-h").value), background: $("#blank-bg").value }));
      enableProjectViews(true); toast("Blank canvas created. Add text and images."); showView("design");
    } catch (e) { toast(e.message, true); }
  });

  // ---------- design view ----------
  const doc = () => state.project && state.project.document;
  const pid = () => state.project && state.project.project.id;
  const elementById = (id) => doc().elements.find((e) => e.id === id);

  async function refreshProject() {
    state.project = await api(`/api/projects/${pid()}`);
  }
  async function patchDocument(ops, label) {
    try {
      state.project = await api(`/api/projects/${pid()}/document`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ops, label }) });
      renderDesign();
    } catch (e) { toast(e.message, true); }
  }

  function renderDesign() {
    if (!doc()) return;
    const d = doc();
    $("#design-title").textContent = `${state.project.project.name} · ${d.canvas_width}×${d.canvas_height}`;
    state.previewNonce += 1;
    const img = $("#master-img");
    img.onload = () => { drawOverlay(); };
    img.src = `/api/projects/${pid()}/preview.png?max_side=1400&n=${state.previewNonce}`;
    // import notes
    const notes = (d.metadata && d.metadata.import_notes) || [];
    $("#import-notes-wrap").classList.toggle("hidden", notes.length === 0);
    $("#import-notes").innerHTML = notes.map((n) => `<li>${esc(n)}</li>`).join("");
    renderLayerList();
    renderElementPanel();
    renderConstraints();
    renderFonts();
    renderHistory();
    drawOverlay();
  }

  function renderLayerList() {
    const list = $("#layer-list");
    const d = doc();
    const items = [...d.elements].sort((a, b) => b.z_index - a.z_index);
    list.innerHTML = items.map((e) => {
      const low = e.role_confidence < 0.6 && !isBg(e);
      const rec = e.provenance && e.provenance.origin === "recovered" && !isBg(e);
      return `<li data-id="${esc(e.id)}" class="${e.id === state.selectedId ? "selected" : ""}">
        <span class="grow">${esc(e.name)}${e.locked ? " 🔒" : ""}${e.visible ? "" : " (hidden)"}</span>
        <span class="role">${esc(e.role)}${rec ? ' <span class="badge neutral" title="Recovered from a flat image">recovered</span>' : ""}${low ? ' <span class="badge warn">check</span>' : ""}</span></li>`;
    }).join("");
    list.querySelectorAll("li").forEach((li) => li.addEventListener("click", () => selectElement(li.dataset.id)));
  }
  const isBg = (e) => ["background", "background_pattern", "overlay"].includes(e.role);

  function selectElement(id) {
    state.selectedId = id;
    renderLayerList();
    renderElementPanel();
    drawOverlay();
  }

  function renderElementPanel() {
    const panel = $("#element-panel");
    const e = state.selectedId && elementById(state.selectedId);
    panel.classList.toggle("hidden", !e);
    if (!e) return;
    $("#el-name").value = e.name;
    const roleSel = $("#el-role");
    roleSel.innerHTML = ROLES.map((r) => `<option value="${r}" ${r === e.role ? "selected" : ""}>${r}</option>`).join("");
    const conf = Math.round((e.role_confidence || 0) * 100);
    const badge = $("#el-confidence");
    badge.textContent = conf >= 90 ? "confirmed" : `${conf}% sure`;
    badge.className = `badge ${conf >= 90 ? "ok" : conf >= 60 ? "neutral" : "warn"}`;
    const isText = e.kind === "text" && e.text;
    $("#el-text-wrap").classList.toggle("hidden", !isText);
    const recovered = !isText && e.effects && e.effects.recovered_text;
    $("#el-recovered").classList.toggle("hidden", !recovered);
    if (recovered) $("#el-recovered-text").value = e.effects.recovered_text;
    if (isText) {
      const st = e.text.runs[0] ? e.text.runs[0].style : {};
      $("#el-text").value = e.text.runs.map((r) => r.text).join("");
      const runsNote = $("#el-runs-note");
      runsNote.classList.toggle("hidden", e.text.runs.length <= 1);
      if (e.text.runs.length > 1) {
        const parts = e.text.runs.map((r) => `"${r.text.length > 18 ? r.text.slice(0, 18) + "…" : r.text}" ${r.style.weight || "regular"} ${Math.round(r.style.font_size || 0)}px ${r.style.color || ""}`);
        runsNote.textContent = `${e.text.runs.length} styled runs (kept on edit; the fields below show the first run and apply changes to all): ${parts.join(" · ")}`;
      }
      $("#el-font").value = st.font_family || "";
      $("#el-size").value = Math.round(st.font_size || 24);
      $("#el-weight").value = st.weight || "regular";
      $("#el-align").value = st.align || "left";
      $("#el-color").value = st.color || "#000000";
      $("#el-protected").checked = !!e.text.protected;
      $("#el-translations").value = Object.entries(e.text.translations || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
    }
    $("#el-locked").checked = !!e.locked;
    $("#el-visible").checked = !!e.visible;
    $("#el-scale-free").checked = !!(e.allowed && e.allowed.scale_free);
    $("#el-crop").checked = !!(e.allowed && e.allowed.crop);
    $("#el-priority").value = e.priority;
    const p = e.provenance || {};
    $("#el-provenance").textContent = `Source: ${p.origin || "?"}${p.source_ref ? " · " + p.source_ref : ""} · ${p.notes || ""}`;
    $("#font-list").innerHTML = state.fonts.map((f) => `<option value="${esc(f)}">`).join("");
  }

  $("#el-apply").addEventListener("click", async () => {
    const e = elementById(state.selectedId); if (!e) return;
    const ops = [];
    ops.push({ op: "set_flags", element_id: e.id, name: $("#el-name").value, locked: $("#el-locked").checked, visible: $("#el-visible").checked, priority: Number($("#el-priority").value) });
    if ($("#el-role").value !== e.role) ops.push({ op: "set_role", element_id: e.id, role: $("#el-role").value });
    ops.push({ op: "set_allowed", element_id: e.id, allowed: { scale_free: $("#el-scale-free").checked, crop: $("#el-crop").checked } });
    if (e.kind === "text") {
      const translations = {};
      $("#el-translations").value.split("\n").forEach((line) => { const m = /^\s*([A-Za-z0-9-]+)\s*=\s*(.+)$/.exec(line); if (m) translations[m[1]] = m[2].trim(); });
      ops.push({ op: "set_text", element_id: e.id, text: $("#el-text").value, protected: $("#el-protected").checked, translations,
        style: { font_family: $("#el-font").value || "DejaVu Sans", font_size: Number($("#el-size").value) || 24, weight: $("#el-weight").value, align: $("#el-align").value, color: $("#el-color").value || "#000000" } });
    }
    await patchDocument(ops, `edit ${e.name}`);
    toast("Saved.");
  });
  $("#el-convert").addEventListener("click", async () => {
    const e = elementById(state.selectedId); if (!e) return;
    const text = $("#el-recovered-text").value.trim();
    if (!text) { toast("Enter the text first.", true); return; }
    await patchDocument([{ op: "convert_to_text", element_id: e.id, text }], `convert ${e.name} to text`);
    toast("Converted to editable text. Check font, size and colour.");
  });
  $("#el-delete").addEventListener("click", async () => {
    const e = elementById(state.selectedId); if (!e) return;
    if (!confirm(`Remove "${e.name}" from the design? You can undo.`)) return;
    state.selectedId = null;
    await patchDocument([{ op: "delete_element", element_id: e.id }], `remove ${e.name}`);
  });
  $("#el-front").addEventListener("click", () => { const e = elementById(state.selectedId); if (e) patchDocument([{ op: "set_z", element_id: e.id, z_index: e.z_index + 1.5 }], "reorder"); });
  $("#el-back").addEventListener("click", () => { const e = elementById(state.selectedId); if (e) patchDocument([{ op: "set_z", element_id: e.id, z_index: e.z_index - 1.5 }], "reorder"); });
  $("#btn-undo").addEventListener("click", async () => {
    try { state.project = await api(`/api/projects/${pid()}/undo`, { method: "POST" }); renderDesign(); toast(state.project.undone ? "Undone." : "Nothing to undo."); } catch (e) { toast(e.message, true); }
  });
  $("#btn-save-project").addEventListener("click", async () => {
    try { const blob = await api(`/api/projects/${pid()}/export/project`); downloadBlob(blob, `${state.project.project.name}.autobanner.zip`); } catch (e) { toast(e.message, true); }
  });
  $("#btn-to-brief").addEventListener("click", () => showView("brief"));
  async function addElement(fd) {
    try {
      state.project = await api(`/api/projects/${pid()}/elements`, { method: "POST", body: fd });
      const added = doc().elements[doc().elements.length - 1];
      state.selectedId = added ? added.id : null;
      renderDesign();
      toast("Element added. Set its role and position.");
    } catch (e) { toast(e.message, true); }
  }
  $("#add-text").addEventListener("click", () => {
    const text = prompt("Text content", "Your headline");
    if (!text) return;
    const d = doc();
    const fd = new FormData();
    fd.append("kind", "text"); fd.append("name", text.slice(0, 24)); fd.append("role", "headline");
    fd.append("x", String(Math.round(d.canvas_width * 0.06))); fd.append("y", String(Math.round(d.canvas_height * 0.1)));
    fd.append("width", String(Math.round(d.canvas_width * 0.5))); fd.append("height", String(Math.round(d.canvas_height * 0.15)));
    fd.append("text", text);
    fd.append("style", JSON.stringify({ font_family: "DejaVu Sans", weight: "bold", color: "#ffffff", font_size: Math.round(d.canvas_height * 0.09) }));
    addElement(fd);
  });
  $("#add-image").addEventListener("click", () => $("#add-image-input").click());
  $("#add-image-input").addEventListener("change", (e) => {
    const file = e.target.files[0]; if (!file) return;
    const d = doc();
    const fd = new FormData();
    fd.append("kind", "image"); fd.append("name", file.name.replace(/\.[^.]+$/, "")); fd.append("role", "logo");
    fd.append("x", String(Math.round(d.canvas_width * 0.75))); fd.append("y", String(Math.round(d.canvas_height * 0.05)));
    fd.append("width", String(Math.round(d.canvas_width * 0.18))); fd.append("height", String(Math.round(d.canvas_height * 0.12)));
    fd.append("file", file);
    addElement(fd);
    e.target.value = "";
  });

  // canvas overlay with drag/resize
  function drawOverlay() {
    const d = doc(); if (!d) return;
    const img = $("#master-img");
    const svg = $("#overlay");
    const W = d.canvas_width, H = d.canvas_height;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    state.stageScale = img.clientWidth ? img.clientWidth / W : 1;
    const parts = [];
    const sorted = [...d.elements].sort((a, b) => a.z_index - b.z_index);
    for (const e of sorted) {
      const g = e.geometry;
      const cls = ["el-box"];
      if (isBg(e)) cls.push("bg");
      if (e.role_confidence < 0.6 && !isBg(e)) cls.push("low");
      if (e.locked) cls.push("locked");
      if (e.id === state.selectedId) cls.push("selected");
      parts.push(`<rect data-id="${esc(e.id)}" class="${cls.join(" ")}" x="${g.x}" y="${g.y}" width="${g.width}" height="${g.height}"></rect>`);
      if (e.id === state.selectedId && !isBg(e)) {
        const fs = Math.max(11, W / 90);
        parts.push(`<text class="el-label" x="${g.x + 4}" y="${Math.max(fs, g.y - 4)}" font-size="${fs}">${esc(e.name)} · ${esc(e.role)}</text>`);
        if (!e.locked) {
          const hs = Math.max(8, W / 100);
          parts.push(`<rect class="handle" data-handle="se" data-id="${esc(e.id)}" x="${g.x + g.width - hs / 2}" y="${g.y + g.height - hs / 2}" width="${hs}" height="${hs}"></rect>`);
        }
      }
    }
    svg.innerHTML = parts.join("");
    svg.querySelectorAll("rect.el-box").forEach((r) => {
      r.addEventListener("pointerdown", (ev) => startDrag(ev, r.dataset.id, "move"));
    });
    svg.querySelectorAll("rect.handle").forEach((r) => {
      r.addEventListener("pointerdown", (ev) => startDrag(ev, r.dataset.id, "resize"));
    });
  }
  function svgPoint(ev) {
    const svg = $("#overlay");
    const rect = svg.getBoundingClientRect();
    const d = doc();
    return { x: (ev.clientX - rect.left) / rect.width * d.canvas_width, y: (ev.clientY - rect.top) / rect.height * d.canvas_height };
  }
  function startDrag(ev, id, mode) {
    ev.preventDefault();
    const e = elementById(id);
    if (!e) return;
    if (state.selectedId !== id) selectElement(id);
    if (e.locked || isBg(e) || !state.can.edit) return;
    const p = svgPoint(ev);
    state.drag = { id, mode, start: p, orig: { ...e.geometry }, moved: false };
    $("#overlay").setPointerCapture(ev.pointerId);
  }
  $("#overlay").addEventListener("pointermove", (ev) => {
    const dr = state.drag; if (!dr) return;
    const e = elementById(dr.id); if (!e) return;
    const p = svgPoint(ev);
    const dx = p.x - dr.start.x, dy = p.y - dr.start.y;
    const d = doc();
    if (dr.mode === "move") {
      e.geometry.x = Math.round(Math.max(0, Math.min(d.canvas_width - dr.orig.width, dr.orig.x + dx)));
      e.geometry.y = Math.round(Math.max(0, Math.min(d.canvas_height - dr.orig.height, dr.orig.y + dy)));
    } else {
      const keepAspect = !(e.allowed && e.allowed.scale_free);
      let w = Math.max(8, dr.orig.width + dx), h = Math.max(8, dr.orig.height + dy);
      if (keepAspect) { const s = Math.max(w / dr.orig.width, h / dr.orig.height); w = dr.orig.width * s; h = dr.orig.height * s; }
      e.geometry.width = Math.round(Math.min(w, d.canvas_width - e.geometry.x));
      e.geometry.height = Math.round(Math.min(h, d.canvas_height - e.geometry.y));
    }
    dr.moved = true;
    drawOverlay();
  });
  async function endDrag() {
    const dr = state.drag; if (!dr) return;
    state.drag = null;
    const e = elementById(dr.id);
    if (dr.moved && e) {
      await patchDocument([{ op: "set_geometry", element_id: e.id, geometry: e.geometry }], `${dr.mode} ${e.name}`);
    }
  }
  $("#overlay").addEventListener("pointerup", endDrag);
  $("#overlay").addEventListener("pointercancel", endDrag);
  document.addEventListener("keydown", async (ev) => {
    if (state.view !== "design") return;
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") { ev.preventDefault(); $("#btn-undo").click(); return; }
    const e = state.selectedId && elementById(state.selectedId);
    if (!e || e.locked || !state.can.edit || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    const step = ev.shiftKey ? 10 : 1;
    const g = { ...e.geometry };
    if (ev.key === "ArrowLeft") g.x -= step; else if (ev.key === "ArrowRight") g.x += step; else if (ev.key === "ArrowUp") g.y -= step; else if (ev.key === "ArrowDown") g.y += step; else return;
    ev.preventDefault();
    await patchDocument([{ op: "set_geometry", element_id: e.id, geometry: g }], `nudge ${e.name}`);
  });
  window.addEventListener("resize", () => { if (state.view === "design") drawOverlay(); });

  // constraints
  const CONSTRAINT_LABELS = {
    keep_visible: (c) => `${nameOf(c.elements[0])} must stay visible`,
    clear_space: (c) => `Clear space around ${nameOf(c.elements[0])} (${c.params.ratio}× its height)`,
    allowed_overlap: (c) => `${nameOf(c.elements[0])} may overlap ${nameOf(c.elements[1])}`,
    order_below: (c) => `${nameOf(c.elements[0])} follows ${nameOf(c.elements[1])}`,
    keep_group: (c) => `Keep together: ${c.elements.map(nameOf).join(", ")}`,
    min_text_size: (c) => `${nameOf(c.elements[0])} at least ${c.params.px}px`,
    anchor_edge: (c) => `${nameOf(c.elements[0])} anchored to ${c.params.edge}`,
    scale_range: (c) => `${nameOf(c.elements[0])} scale ${c.params.min}–${c.params.max}`,
  };
  const nameOf = (id) => { const e = elementById(id); return e ? e.name : id; };
  function renderConstraints() {
    const d = doc();
    const wrap = $("#constraint-list");
    if (!d.constraints.length) { wrap.innerHTML = '<div class="small muted">No rules yet.</div>'; }
    else {
      wrap.innerHTML = d.constraints.map((c) => {
        const fn = CONSTRAINT_LABELS[c.type] || (() => c.type);
        const src = c.provenance && c.provenance.origin === "generated" ? '<span class="badge warn" title="Proposed automatically; confirm or remove">proposed</span>' : '<span class="badge ok">confirmed</span>';
        return `<div class="constraint"><input type="checkbox" data-cid="${esc(c.id)}" ${c.enabled ? "checked" : ""} style="width:auto"><span class="desc">${esc(fn(c))} ${src} ${c.hard ? "" : '<span class="badge neutral">soft</span>'}</span><button data-confirm="${esc(c.id)}" class="small" type="button" ${c.provenance.origin === "generated" ? "" : "disabled"}>Confirm</button><button data-remove="${esc(c.id)}" class="small danger" type="button">Remove</button></div>`;
      }).join("");
    }
    wrap.querySelectorAll("input[data-cid]").forEach((cb) => cb.addEventListener("change", () => patchDocument([{ op: "set_constraint", constraint_id: cb.dataset.cid, enabled: cb.checked }], "toggle rule")));
    wrap.querySelectorAll("button[data-confirm]").forEach((b) => b.addEventListener("click", () => patchDocument([{ op: "set_constraint", constraint_id: b.dataset.confirm, enabled: true }], "confirm rule")));
    wrap.querySelectorAll("button[data-remove]").forEach((b) => b.addEventListener("click", () => patchDocument([{ op: "remove_constraint", constraint_id: b.dataset.remove }], "remove rule")));
    const opts = d.elements.filter((e) => !isBg(e)).map((e) => `<option value="${esc(e.id)}">${esc(e.name)}</option>`).join("");
    $("#c-a").innerHTML = opts; $("#c-b").innerHTML = opts;
    updateConstraintForm();
    renderLearned();
  }
  async function renderLearned() {
    const wrap = $("#learned-list");
    try {
      const res = await api(`/api/projects/${pid()}/learned`);
      const corrections = res.corrections || [];
      const unresolved = res.unresolved_corrections || [];
      if (!res.families.length && !corrections.length && !unresolved.length) { wrap.innerHTML = ""; return; }
      let html = "";
      if (res.families.length) {
        html += `<div><strong>Learned from ${res.examples.length} approved variant${res.examples.length === 1 ? "" : "s"}</strong> (applied to the next generation):</div>` +
          res.families.map((f) => `<div>· ${esc(f.aspect)}: text ${esc(f.text_align)}, ${f.subject_first ? "subject above text" : "text before subject"}, confidence ${f.confidence}${f.constraints.length ? ", proposes " + esc(f.constraints.join(", ")) : ""}${f.examples === 1 ? ' <span class="badge warn" title="One example is under-determined; approve another size of this orientation to confirm">single example</span>' : ""}</div>`).join("");
      }
      const CORR = { scale_range: (c) => `${nameOf(c.element_id)} scale ${c.params.min}–${c.params.max}`, min_text_size: (c) => `${nameOf(c.element_id)} at least ${c.params.px}px (measured at ${(c.params.measured_on || []).join("×")})`, clear_space: (c) => `clear space around ${nameOf(c.element_id)}` };
      if (corrections.length) {
        html += `<div style="margin-top:4px"><strong>From your rejections</strong> (add as a rule to apply on the next generation):</div>` +
          corrections.map((c, i) => `<div>· "${esc(c.reason)}" → ${esc((CORR[c.kind] || (() => c.kind))(c))} <span class="muted">(${esc(c.evidence)})</span> <button data-apply-correction="${i}" class="small" type="button">Add as rule</button></div>`).join("");
      }
      if (unresolved.length) {
        html += unresolved.map((u) => `<div class="muted">· "${esc(u.reason)}": ${esc(u.note)}</div>`).join("");
      }
      wrap.innerHTML = html;
      wrap.querySelectorAll("button[data-apply-correction]").forEach((b) => b.addEventListener("click", async () => {
        try { state.project = await api(`/api/projects/${pid()}/learned/corrections/${b.dataset.applyCorrection}/apply`, { method: "POST" }); renderDesign(); toast("Rule added from correction"); } catch (e) { toast(e.message, true); }
      }));
    } catch (e) { wrap.innerHTML = ""; }
  }
  function updateConstraintForm() {
    const t = $("#c-type").value;
    $("#c-b").classList.toggle("hidden", !["allowed_overlap", "order_below", "keep_group"].includes(t));
    $("#c-param").classList.toggle("hidden", !["min_text_size", "clear_space"].includes(t));
    $("#c-param").placeholder = t === "clear_space" ? "ratio (e.g. 0.5)" : "px";
  }
  $("#c-type").addEventListener("change", updateConstraintForm);
  $("#c-add").addEventListener("click", () => {
    const t = $("#c-type").value;
    const els = [$("#c-a").value];
    if (!$("#c-b").classList.contains("hidden")) els.push($("#c-b").value);
    const params = {};
    if (t === "min_text_size") params.px = Number($("#c-param").value) || 10;
    if (t === "clear_space") params.ratio = Number($("#c-param").value) || 0.5;
    if (els.length === 2 && els[0] === els[1]) { toast("Pick two different elements.", true); return; }
    patchDocument([{ op: "add_constraint", constraint_id: null, constraint: { type: t, elements: els, params, hard: true } }], "add rule");
  });
  function renderFonts() {
    const d = doc();
    $("#font-status").innerHTML = (d.fonts || []).map((f) => `<li>${esc(f.family)} ${esc(f.weight)} — ${f.status === "available" ? '<span class="badge ok">available</span>' : `<span class="badge warn">${esc(f.status)}${f.substitute ? " → " + esc(f.substitute) : ""}</span>`}</li>`).join("") || '<li class="muted">No text elements.</li>';
  }
  function renderHistory() {
    $("#history-list").innerHTML = (state.project.history || []).slice().reverse().map((h) => `<li>v${h.version} · ${esc(h.label)} · ${esc(h.saved_at)}</li>`).join("");
  }

  // ---------- brief view ----------
  async function loadPresets() {
    try { const data = await api("/api/presets"); state.presets = data.presets; $("#preset-note").textContent = `${data.note} (preset list ${data.version})`; } catch (e) { toast(e.message, true); }
    try { const f = await api("/api/fonts"); state.fonts = f.families; } catch (e) { /* optional */ }
    return true;
  }
  function renderBrief() {
    const grid = $("#preset-grid");
    grid.innerHTML = state.presets.map((p) => `<label><input type="checkbox" data-preset="${p.id}" ${state.selectedPresets.has(p.id) ? "checked" : ""}><span class="grow">${esc(p.name)}<div class="small muted">${esc(p.channel)} · ${p.width}×${p.height}${p.verified ? "" : " · unverified"}</div></span></label>`).join("");
    grid.querySelectorAll("input").forEach((cb) => cb.addEventListener("change", () => { if (cb.checked) state.selectedPresets.add(cb.dataset.preset); else state.selectedPresets.delete(cb.dataset.preset); updateCampaignCount(); }));
    renderCustomTargets();
    renderCampaign();
    const d = doc();
    const texts = d.elements.filter((e) => e.kind === "text");
    $("#copy-overrides").innerHTML = texts.map((e) => {
      const cur = e.text.runs.map((r) => r.text).join("");
      return `<label>${esc(e.name)} <span class="muted">(${esc(e.role)})</span></label>${e.text.protected ? `<input type="text" value="${esc(cur)}" disabled title="Verbatim copy">` : `<input type="text" data-override="${esc(e.id)}" placeholder="${esc(cur)}">`}`;
    }).join("") || '<div class="small muted">No editable text in this design.</div>';
  }
  function renderCustomTargets() {
    $("#custom-list").innerHTML = state.customTargets.map((t, i) => `<li>${esc(t.name)} ${t.width}×${t.height} <button data-rm="${i}" class="small" type="button">remove</button></li>`).join("");
    $("#custom-list").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.customTargets.splice(Number(b.dataset.rm), 1); renderCustomTargets(); }));
    updateCampaignCount();
  }
  $("#custom-add").addEventListener("click", () => {
    const w = Number($("#custom-w").value), h = Number($("#custom-h").value);
    if (!(w > 0 && h > 0)) { toast("Enter a width and height.", true); return; }
    state.customTargets.push({ width: w, height: h, name: $("#custom-name").value.trim() || `${w}×${h}` });
    $("#custom-name").value = "";
    renderCustomTargets();
  });
  $("#btn-generate").addEventListener("click", async () => {
    const overrides = {};
    $$("#copy-overrides input[data-override]").forEach((i) => { if (i.value.trim()) overrides[i.dataset.override] = i.value; });
    const spec = { preset_ids: [...state.selectedPresets], targets: state.customTargets, text_overrides: overrides, locale: $("#brief-locale").value.trim() || null };
    if (!spec.preset_ids.length && !spec.targets.length) { toast("Pick at least one size.", true); return; }
    if (state.campaignRows.length) spec.rows = state.campaignRows.map((r) => ({ id: r.id, label: r.label, text_overrides: r.text_overrides, locale: r.locale }));
    try {
      const res = await api(`/api/projects/${pid()}/variants`, json(spec));
      state.project = res;
      state.job = res.job;
      trackJob(res.job.id);
      toast(`Generating ${res.job.items.length} variants…`);
    } catch (e) { toast(e.message, true); }
  });

  // ---------- campaign table ----------
  function campaignTexts() { return doc().elements.filter((e) => e.kind === "text" && e.text && !e.text.protected); }
  function updateCampaignCount() {
    const sizes = state.selectedPresets.size + state.customTargets.length;
    const rows = state.campaignRows.length;
    $("#campaign-count").textContent = rows ? `${rows} row${rows === 1 ? "" : "s"} × ${sizes} size${sizes === 1 ? "" : "s"} = ${rows * sizes} variants` : "";
  }
  function renderCampaign() {
    const wrap = $("#campaign-table-wrap");
    const texts = campaignTexts();
    if (!state.campaignRows.length) {
      wrap.innerHTML = '<div class="small muted">No rows: the sizes above are generated once with the copy fields. Read a table or add a row to generate a campaign.</div>';
      updateCampaignCount();
      return;
    }
    const head = `<tr><th>Row</th>${texts.map((e) => `<th>${esc(e.name)} <span class="muted">(${esc(e.role)})</span></th>`).join("")}<th>Locale</th><th></th></tr>`;
    const body = state.campaignRows.map((r, i) => `<tr><td><input data-row="${i}" data-field="label" value="${esc(r.label)}"></td>${texts.map((e) => `<td><input data-row="${i}" data-el="${esc(e.id)}" value="${esc(r.text_overrides[e.id] || "")}" placeholder="${esc(e.text.runs.map((x) => x.text).join(""))}"></td>`).join("")}<td><input class="narrow" data-row="${i}" data-field="locale" value="${esc(r.locale || "")}" placeholder="en"></td><td><button class="small" data-rmrow="${i}" type="button">remove</button></td></tr>`).join("");
    wrap.innerHTML = `<div class="table-wrap"><table class="campaign">${head}${body}</table></div>`;
    wrap.querySelectorAll("input").forEach((inp) => inp.addEventListener("input", () => {
      const r = state.campaignRows[Number(inp.dataset.row)]; if (!r) return;
      if (inp.dataset.el) { if (inp.value.trim()) r.text_overrides[inp.dataset.el] = inp.value; else delete r.text_overrides[inp.dataset.el]; }
      else if (inp.dataset.field === "locale") r.locale = inp.value.trim() || null;
      else r.label = inp.value;
    }));
    wrap.querySelectorAll("[data-rmrow]").forEach((b) => b.addEventListener("click", () => { state.campaignRows.splice(Number(b.dataset.rmrow), 1); renderCampaign(); }));
    updateCampaignCount();
  }
  $("#campaign-parse").addEventListener("click", async () => {
    const csv = $("#campaign-csv").value;
    if (!csv.trim()) { toast("Paste a table first.", true); return; }
    try {
      const res = await api(`/api/projects/${pid()}/campaign/rows`, json({ csv }));
      state.campaignRows = res.rows; state.rowSeq = res.rows.length;
      $("#campaign-note").textContent = res.notes.join(" ");
      renderCampaign();
      toast(`${res.rows.length} row${res.rows.length === 1 ? "" : "s"} read.`);
    } catch (e) { toast(e.message, true); }
  });
  $("#campaign-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    $("#campaign-csv").value = await f.text();
    ev.target.value = "";
    $("#campaign-parse").click();
  });
  $("#campaign-add-row").addEventListener("click", () => {
    state.rowSeq += 1;
    state.campaignRows.push({ id: `row${state.rowSeq}`, label: `Row ${state.rowSeq}`, text_overrides: {}, locale: null });
    renderCampaign();
  });
  $("#campaign-clear").addEventListener("click", () => { state.campaignRows = []; $("#campaign-note").textContent = ""; renderCampaign(); });
  $("#btn-cancel").addEventListener("click", async () => { if (state.job) { try { await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }); toast("Cancelling…"); } catch (e) { toast(e.message, true); } } });
  function trackJob(jobId) {
    $("#job-progress").classList.remove("hidden"); $("#btn-cancel").classList.remove("hidden"); $("#btn-generate").disabled = true;
    clearInterval(state.jobTimer);
    const poll = async () => {
      try {
        const { job } = await api(`/api/jobs/${jobId}`);
        state.job = job;
        $("#job-bar").style.width = `${Math.round(job.progress * 100)}%`;
        const done = job.items.filter((i) => i.status === "done").length;
        const running = job.items.find((i) => i.status === "running");
        $("#job-text").textContent = `${done}/${job.items.length} done` + (running ? ` · ${running.label}: ${running.stage}` : "") + ` · ${job.status}`;
        setStatus(`Generating… ${done}/${job.items.length}`);
        if (["done", "failed", "cancelled", "partial"].includes(job.status)) {
          clearInterval(state.jobTimer); state.jobTimer = null;
          $("#btn-cancel").classList.add("hidden"); $("#btn-generate").disabled = false;
          await refreshProject();
          setStatus(`Job ${job.status}`);
          toast(job.status === "done" ? "Variants ready for review." : `Job ${job.status}: some variants are missing.`, job.status === "failed");
          showView("review");
        }
      } catch (e) { clearInterval(state.jobTimer); toast(e.message, true); $("#btn-generate").disabled = false; }
    };
    poll();
    state.jobTimer = setInterval(poll, 1200);
  }

  // ---------- review view ----------
  $$(".filters button").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.filter; $$(".filters button").forEach((x) => x.classList.toggle("active", x === b)); renderReview(); }));
  $("#review-row").addEventListener("change", (e) => { state.rowFilter = e.target.value; renderReview(); });
  function renderReview() {
    if (!state.project) return;
    const variants = state.project.variants || [];
    const rowOf = (v) => (v.brief && v.brief.row) || null;
    const rowIds = [], rowLabels = {};
    variants.forEach((v) => { const r = rowOf(v); if (r && !rowLabels[r.id]) { rowIds.push(r.id); rowLabels[r.id] = r.label; } });
    const rowSel = $("#review-row");
    rowSel.innerHTML = '<option value="all">All rows</option>' + rowIds.map((id) => `<option value="${esc(id)}">${esc(rowLabels[id])}</option>`).join("");
    state.rowFilter = rowIds.includes(state.rowFilter) ? state.rowFilter : "all";
    rowSel.value = state.rowFilter;
    rowSel.classList.toggle("hidden", rowIds.length === 0);
    const filtered = variants.filter((v) => (state.filter === "all" ? true : state.filter === "approved" ? v.approval === "approved" : v.verdict === state.filter) && (state.rowFilter === "all" || (rowOf(v) || {}).id === state.rowFilter));
    const counts = { accepted: 0, needs_review: 0, failed: 0, approved: 0, pending: 0 };
    variants.forEach((v) => { if (counts[v.verdict] !== undefined) counts[v.verdict]++; if (v.approval === "approved") counts.approved++; if (["pending", "running"].includes(v.status)) counts.pending++; });
    $("#review-summary").textContent = `${variants.length} variants${rowIds.length ? ` in ${rowIds.length} rows` : ""} · ${counts.accepted} accepted · ${counts.needs_review} need review · ${counts.failed} failed · ${counts.approved} approved${counts.pending ? ` · ${counts.pending} in progress` : ""}`;
    $("#review-empty").classList.toggle("hidden", variants.length > 0);
    const grid = $("#variant-grid");
    grid.innerHTML = filtered.map((v) => {
      const thumb = v.image_path ? `<div class="thumb" data-open="${esc(v.id)}"><img src="/api/projects/${pid()}/variants/${v.id}/image.png?n=${esc(v.updated_at)}" alt="${esc(v.name)}"></div>` : `<div class="thumb placeholder">${esc(v.status)}${v.error ? ": " + esc(v.error) : ""}</div>`;
      const ed = state.can.edit ? "" : "disabled", ap = state.can.approve ? "" : "disabled";
      const rowTag = rowOf(v) ? `<span class="row-tag">${esc(rowOf(v).label)}</span>` : "";
      return `<div class="card variant-card">${thumb}<div class="row"><strong class="grow">${esc(v.name)}</strong>${rowTag}<span class="small muted">${v.width}×${v.height}</span></div><div class="row">${verdictBadge(v.verdict)} ${approvalBadge(v.approval)}</div><div class="row"><button data-open="${esc(v.id)}" class="small" type="button">Review</button><button data-approve="${esc(v.id)}" class="small" type="button" ${v.status !== "done" || ap ? "disabled" : ""}>Approve</button><button data-regen="${esc(v.id)}" class="small" type="button" ${ed}>Regenerate</button><button data-del="${esc(v.id)}" class="small danger" type="button" ${ed}>Delete</button></div></div>`;
    }).join("");
    grid.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => openDetail(b.dataset.open)));
    grid.querySelectorAll("[data-approve]").forEach((b) => b.addEventListener("click", () => setApproval(b.dataset.approve, "approved", "")));
    grid.querySelectorAll("[data-regen]").forEach((b) => b.addEventListener("click", () => regenerate(b.dataset.regen)));
    grid.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => { if (!confirm("Delete this variant?")) return; try { await api(`/api/projects/${pid()}/variants/${b.dataset.del}`, { method: "DELETE" }); await refreshProject(); renderReview(); } catch (e) { toast(e.message, true); } }));
    if (state.detailId && !variants.find((v) => v.id === state.detailId)) closeDetail();
  }
  async function setApproval(id, approval, reason) {
    try { await api(`/api/projects/${pid()}/variants/${id}/approval`, json({ approval, reason })); await refreshProject(); renderReview(); if (state.detailId === id) openDetail(id); toast(approval === "approved" ? "Approved." : "Rejected."); } catch (e) { toast(e.message, true); }
  }
  async function regenerate(id, spec) {
    try { const res = await api(`/api/projects/${pid()}/variants/${id}/regenerate`, spec ? json(spec) : { method: "POST" }); trackJob(res.job.id); toast("Regenerating…"); } catch (e) { toast(e.message, true); }
  }
  async function openDetail(id) {
    try {
      const data = await api(`/api/projects/${pid()}/variants/${id}`);
      state.detailId = id;
      const v = data.variant;
      $("#variant-detail").classList.remove("hidden");
      $("#detail-title").textContent = `${v.name} · ${v.width}×${v.height}`;
      const src = v.image_path ? `/api/projects/${pid()}/variants/${id}/image.png?n=${encodeURIComponent(v.updated_at)}` : "";
      $("#detail-img").src = src; $("#compare-variant").src = src;
      $("#compare-master").src = `/api/projects/${pid()}/preview.png?max_side=900&n=${state.previewNonce}`;
      $("#detail-verdict").innerHTML = `${verdictBadge(v.verdict)} ${approvalBadge(v.approval)}${v.approval_reason ? ` <span class="small muted">${esc(v.approval_reason)}</span>` : ""}`;
      const q = data.quality || {};
      const issues = (q.checks || []).filter((c) => c.status !== "pass").sort((a, b) => (a.status === "fail" ? 0 : a.status === "needs_review" ? 1 : 2) - (b.status === "fail" ? 0 : b.status === "needs_review" ? 1 : 2));
      $("#detail-issues").innerHTML = issues.length ? issues.map((c) => `<li class="${c.status}">${esc(c.message)}</li>`).join("") : '<li class="muted">No issues found by the automatic checks.</li>';
      const plan = data.plan || {};
      const overrides = (v.brief && v.brief.text_overrides) || {};
      const texts = doc().elements.filter((e) => e.kind === "text" && !e.text.protected);
      $("#detail-overrides").innerHTML = texts.map((e) => `<label>${esc(e.name)}</label><input type="text" data-dov="${esc(e.id)}" value="${esc(overrides[e.id] || "")}" placeholder="${esc(e.text.runs.map((r) => r.text).join(""))}">`).join("") || '<div class="small muted">No editable text.</div>';
      $("#detail-json").textContent = JSON.stringify({ plan, quality_summary: q.summary, fonts: plan.fonts, warnings: plan.warnings, repair_steps: plan.repair_steps, config: q.config }, null, 2);
      $("#variant-detail").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { toast(e.message, true); }
  }
  function closeDetail() { state.detailId = null; $("#variant-detail").classList.add("hidden"); }
  $("#detail-close").addEventListener("click", closeDetail);
  $("#detail-compare").addEventListener("change", (e) => { $("#detail-single").classList.toggle("hidden", e.target.checked); $("#detail-compare-wrap").classList.toggle("hidden", !e.target.checked); });
  $("#detail-approve").addEventListener("click", () => setApproval(state.detailId, "approved", ""));
  $("#detail-reject").addEventListener("click", () => { const r = $("#detail-reason").value.trim(); if (!r) { toast("Give a reason for rejecting.", true); return; } setApproval(state.detailId, "rejected", r); });
  $("#detail-regenerate").addEventListener("click", () => regenerate(state.detailId, { keep_layout: $("#detail-keep-layout").checked }));
  $("#detail-apply-overrides").addEventListener("click", () => { const o = {}; $$("#detail-overrides input[data-dov]").forEach((i) => { if (i.value.trim()) o[i.dataset.dov] = i.value; }); regenerate(state.detailId, { text_overrides: o, keep_layout: $("#detail-keep-layout").checked }); });
  async function exportZip(only) {
    try { const fmt = $("#export-format").value; const blob = await api(`/api/projects/${pid()}/export?format=${fmt}&only=${only}`); downloadBlob(blob, `${state.project.project.name}_${only}.zip`); } catch (e) { toast(e.message, true); }
  }
  $("#btn-export-approved").addEventListener("click", () => exportZip("approved"));
  $("#btn-export-all").addEventListener("click", () => exportZip("all"));

  // ---------- auth ----------
  function applyRoleGates() {
    const role = state.auth ? state.auth.role : "admin";
    const mode = state.auth ? state.auth.mode : "open";
    state.can = {
      edit: mode === "open" || ["editor", "admin"].includes(role),
      approve: mode === "open" || ["approver", "admin"].includes(role),
      admin: mode === "open" || role === "admin",
    };
    $$("[data-needs]").forEach((el) => { el.disabled = !state.can[el.dataset.needs]; el.title = state.can[el.dataset.needs] ? "" : `Your role (${role}) cannot do this`; });
    const chip = $("#user-chip");
    if (state.auth && state.auth.authenticated && state.auth.user) {
      chip.classList.remove("hidden");
      $("#user-name").textContent = `${state.auth.user} · ${state.auth.workspace}`;
      $("#user-role").textContent = role;
    } else chip.classList.add("hidden");
    $("#open-badge").classList.toggle("hidden", mode !== "open");
  }
  function showAuth(setup) {
    state.project = null;
    $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== "view-auth"));
    $$("header nav button").forEach((b) => { b.disabled = true; b.classList.remove("active"); });
    $("#auth-setup").classList.toggle("hidden", !setup);
    $("#auth-login").classList.toggle("hidden", !!setup);
    $("#auth-error").classList.add("hidden");
    $("#user-chip").classList.add("hidden");
    setStatus(setup ? "First run: create the administrator" : "Sign in");
    (setup ? $("#setup-username") : $("#login-username")).focus();
  }
  function authError(msg) { const el = $("#auth-error"); el.textContent = msg; el.classList.remove("hidden"); }
  async function loadAuth() {
    try { state.auth = await api("/api/auth/status"); } catch (e) { state.auth = { mode: "open", authenticated: true, role: "admin" }; }
    applyRoleGates();
    return state.auth.authenticated;
  }
  async function enterApp() {
    $$("header nav button").forEach((b) => { b.disabled = b.dataset.view !== "projects"; });
    await loadPresets();
    const { id, view } = parseHash();
    if (id) openProject(id, ["design", "brief", "review"].includes(view) ? view : "design");
    else showView("projects");
  }
  async function submitAuth(path, body) {
    try {
      await api(path, json(body));
      await loadAuth();
      toast(path.endsWith("setup") ? "Administrator created. You are signed in." : "Signed in.");
      await enterApp();
    } catch (e) { authError(e.message); }
  }
  $("#setup-submit").addEventListener("click", () => submitAuth("/api/auth/setup", {
    workspace: $("#setup-workspace").value.trim() || "default", username: $("#setup-username").value.trim(),
    password: $("#setup-password").value, setup_token: $("#setup-token").value.trim(),
  }));
  $("#login-submit").addEventListener("click", () => submitAuth("/api/auth/login", { username: $("#login-username").value.trim(), password: $("#login-password").value }));
  ["#setup-token", "#login-password"].forEach((sel) => $(sel).addEventListener("keydown", (ev) => { if (ev.key === "Enter") (sel === "#login-password" ? $("#login-submit") : $("#setup-submit")).click(); }));
  $("#btn-logout").addEventListener("click", async () => {
    try { await api("/api/auth/logout", { method: "POST" }); } catch (e) { /* session may already be gone */ }
    await loadAuth();
    showAuth(false);
  });
  async function renderWorkspace() {
    const card = $("#workspace-card");
    const a = state.auth;
    if (!a || a.mode !== "local" || !a.authenticated) { card.classList.add("hidden"); return; }
    card.classList.remove("hidden");
    $("#workspace-name").textContent = a.workspace || "";
    $("#members-wrap").classList.toggle("hidden", !state.can.admin);
    $("#tokens-wrap").classList.toggle("hidden", !a.user);
    if (state.can.admin) {
      try {
        const { users } = await api("/api/auth/users");
        const tbody = $("#member-table tbody");
        tbody.innerHTML = users.map((u) => `<tr><td>${esc(u.username)}${u.username === a.user ? ' <span class="muted">(you)</span>' : ""}</td><td><select data-role-for="${esc(u.username)}">${["viewer", "editor", "approver", "admin"].map((r) => `<option ${r === u.role ? "selected" : ""}>${r}</option>`).join("")}</select></td><td>${u.username === a.user ? "" : `<button class="small danger" data-remove-user="${esc(u.username)}" type="button">Remove</button>`}</td></tr>`).join("");
        tbody.querySelectorAll("select[data-role-for]").forEach((sel) => sel.addEventListener("change", async () => {
          try { await api(`/api/auth/users/${encodeURIComponent(sel.dataset.roleFor)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ role: sel.value }) }); toast("Role updated."); } catch (e) { toast(e.message, true); renderWorkspace(); }
        }));
        tbody.querySelectorAll("button[data-remove-user]").forEach((b) => b.addEventListener("click", async () => {
          if (!confirm(`Remove ${b.dataset.removeUser} from the workspace?`)) return;
          try { await api(`/api/auth/users/${encodeURIComponent(b.dataset.removeUser)}`, { method: "DELETE" }); renderWorkspace(); } catch (e) { toast(e.message, true); }
        }));
      } catch (e) { toast(e.message, true); }
    }
    if (a.user) {
      try {
        const me = await api("/api/auth/me");
        $("#token-list").innerHTML = (me.tokens || []).map((t) => `<li>${esc(t.name)} <span class="muted">· created ${esc(t.created_at)}${t.last_used ? " · last used " + esc(t.last_used) : ""}</span> <button class="small danger" data-revoke="${esc(t.id)}" type="button">Revoke</button></li>`).join("") || '<li class="muted">No tokens.</li>';
        $("#token-list").querySelectorAll("button[data-revoke]").forEach((b) => b.addEventListener("click", async () => { try { await api(`/api/auth/tokens/${b.dataset.revoke}`, { method: "DELETE" }); renderWorkspace(); } catch (e) { toast(e.message, true); } }));
      } catch (e) { /* tokens are optional */ }
    }
  }
  $("#member-add").addEventListener("click", async () => {
    try {
      await api("/api/auth/users", json({ username: $("#member-username").value.trim(), password: $("#member-password").value, role: $("#member-role").value }));
      $("#member-username").value = ""; $("#member-password").value = "";
      toast("Member added."); renderWorkspace();
    } catch (e) { toast(e.message, true); }
  });
  $("#token-create").addEventListener("click", async () => {
    try {
      const res = await api("/api/auth/tokens", json({ name: $("#token-name").value.trim() || "token" }));
      const pre = $("#token-new"); pre.textContent = `Copy it now, it is not shown again:\n${res.token}`; pre.classList.remove("hidden");
      $("#token-name").value = ""; renderWorkspace();
    } catch (e) { toast(e.message, true); }
  });

  // ---------- boot ----------
  loadAuth().then((authenticated) => {
    if (!authenticated) showAuth(!!state.auth.setup_required);
    else enterApp();
  });
})();
