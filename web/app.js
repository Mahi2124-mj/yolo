/* Poketenashi dashboard - every page of "/". Needs ui.js.
   ES2015 on purpose (no async/await, no ?. or ??): see ui.js.
   Every change - cameras, zones, settings, users, passwords - goes through a
   confirmation popup that shows what will change before anything is sent. */
(function () {
"use strict";
const {$, $$, esc, icon, api, modal, confirm, kv, toast, fmt, store} = window.UI;

/* ======================================================================
   Constants
   ====================================================================== */
/* Which rules exist, what each is called and whether it is really running all
   come from the monitor itself (/api/state, built from poketenashi/ruleinfo.py).
   No name is written down here: this page, the TV page and the alert log show
   the one label the server sends, so they cannot describe the same rule
   differently.

   These files are served from disk while the monitor keeps running in memory,
   so a page can be newer than the API it talks to. ruleCatalog() is the one
   place that reads the rule list out of whatever /api/state actually returns:
   a monitor that knows per-rule status sends `rule_status` (code, label and
   whether it can run); one running older code sends only `rules`, the codes
   switched on - then the labels and the reasons are simply unknown, and the
   pages say that instead of going blank or waiting on a monitor that is
   answering. `active: null` means "the monitor did not say". */
let RULES = [];                     // rule codes, in the order the server lists them
let RULE_NAMES = {};                // code -> operator-facing label ({} on older code)
const rootCss = getComputedStyle(document.documentElement);
const COLOR = {};
function ruleCatalog(s) {
  const st = (s && s.rule_status) || [];
  if (st.length) return {list: st, mode: "status"};
  const codes = (s && s.rules) || [];
  return {list: codes.map(code => ({rule: code, label: "", active: null, reason: null})),
          mode: codes.length ? "codes" : "none"};
}
function learnRules(list) {
  const sig = list.map(r => r.rule + "|" + r.label + "|" + (r.active === null ? "?" : r.active ? 1 : 0) + "|" + (r.reason || "")).join(",");
  if (sig === learnRules.sig) return false;
  learnRules.sig = sig;
  RULES = list.map(r => r.rule);
  RULE_NAMES = {};
  list.forEach(r => {
    if (r.label) RULE_NAMES[r.rule] = r.label;
    if (!COLOR[r.rule]) COLOR[r.rule] = rootCss.getPropertyValue("--" + r.rule).trim() || "#3987e5";
  });
  return true;
}
// A rule the camera does not check (Cameras page -> Rules…) is not shown on
// that camera at all: only the rules it was given are listed.
const camChecks_ = list => list.filter(r => !/^not checked on this camera/.test(r.reason || ""));
const ruleColor = r => COLOR[r] || "#3987e5";
const badge = r => `<span class="badge" style="background:${ruleColor(r)}">${esc(r)}</span>`;

const PAGES = {
  dash:  {title: "Dashboard",   sub: "What is happening on the floor right now", icon: "dashboard"},
  live:  {title: "Live view",   sub: "The cameras' annotated pictures, live", icon: "live"},
  evid:  {title: "Evidence",    sub: "Every violation with its picture and clip", icon: "evidence"},
  cams:  {title: "Cameras",     sub: "Add, change and switch the cameras", icon: "camera"},
  zones: {title: "Zone editor", sub: "Where each rule applies - per camera", icon: "zones"},
  set:   {title: "Settings",    sub: "Retention, live picture, users and the audit log", icon: "settings"},
  tv:    {title: "TV view",     sub: "A full-screen page for wall screens and TVs", icon: "tv"},
};
const NAV = [{sec: "Monitor"}, "dash", "live", "evid", {sec: "Configure"}, "cams", "zones", "set",
             {sec: "Display"}, "tv"];
const SET_TABS = [
  {id: "general", label: "Retention", icon: "database"},
  {id: "picture", label: "Live picture", icon: "layers"},
  {id: "users", label: "Users", icon: "users"},
  {id: "screens", label: "TV screens", icon: "tv"},
  {id: "account", label: "My account", icon: "key"},
  {id: "audit", label: "Audit log", icon: "history"},
  {id: "system", label: "System", icon: "server"},
];
const OVERLAYS = {
  keypoints: ["Body keypoints", "The pose skeleton drawn on each person"],
  hands: ["Hand skeletons", "The finger points found by the hand model"],
  zones: ["Zones", "The outlines of the gangway, crossing and other zones"],
  trail: ["Wrist trail", "Recent wrist movement - for checking PO"],
  enhance: ["Brighten dark pictures", "Contrast boost of the shown picture (detection is not affected)"],
  sharpen: ["Sharpen", "A sharper shown picture (detection is not affected)"],
};

/* ======================================================================
   State
   ====================================================================== */
const S = {
  me: {user: null, role: null, can_admin: false, setup_needed: false, local: false, lan: [], port: location.port},
  up: false, uptime: null, st: null, cams: [], camSig: "",
  sel: store.get("pk_cam", ""), grid: false,
  page: "dash", sub: "general", ruleView: "today",
  stats: null, statsSig: "", sys: null,
  ev: {cam: "", rule: "", limit: 60},
  alertSeq: -1, sound: store.get("pk_sound", "0") === "1",
};

const camById = id => S.cams.find(c => c.id === id);
const camName = id => { const c = camById(id); return c ? c.name : (id || ""); };
const maskUrl = s => String(s || "").replace(/(\/\/[^/:@]+):[^@/]*@/, "$1:***@");
const evidenceImg = v => v.snapshot ? "/snapshots/" + encodeURIComponent(v.snapshot)
  : (v.poster ? "/clips/" + encodeURIComponent(v.poster) : "");
const sw = (id, title, help, on, disabled) =>
  `<label class="switch"><input type="checkbox" id="${id}"${on ? " checked" : ""}${disabled ? " disabled" : ""}>` +
  `<span class="track"></span><span class="txt"><b>${esc(title)}</b>${help ? `<span>${esc(help)}</span>` : ""}</span></label>`;
const emptyBox = (ic, msg) => `<div class="empty">${icon(ic)}<span>${esc(msg)}</span></div>`;
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");

function statusPill(c) {
  if (!c.enabled) return `<span class="pill"><span class="dot"></span>Switched off</span>`;
  if (c.camera_ok && S.up) return `<span class="pill ok"><span class="dot ok"></span>Online</span>`;
  return `<span class="pill bad"><span class="dot bad"></span>Offline</span>`;
}
function hintFor(err) {
  const e = String(err || "");
  if (/answers, but the stream/i.test(e)) return "The camera answers, but not on this stream path - every camera brand uses its own. Open Edit and press “Find stream”: the monitor asks the camera itself.";
  if (/not reachable/i.test(e)) return "Nothing answers at this IP address: check the camera's power and cable, and that the IP is right.";
  if (/stopped sending|no new frame/i.test(e)) return "The picture stopped; the monitor reconnects by itself.";
  return "";
}

/* ======================================================================
   Navigation
   ====================================================================== */
function parseHash() {
  const h = location.hash.replace(/^#/, "").split("/");
  return {page: PAGES[h[0]] ? h[0] : "dash", sub: h[1] || ""};
}
function setHash() {
  const h = "#" + S.page + (S.page === "set" ? "/" + S.sub : "");
  if (location.hash !== h) history.replaceState(null, "", h);
}
function go(page, sub) {
  if (!PAGES[page]) page = "dash";
  if (S.page === "zones" && page !== "zones" && ZE.dirty()) {
    ZE.askDiscard().then(ok => {
      if (ok) { ZE.revert(); go(page, sub); } else setHash();
    });
    return;
  }
  if (sub && SET_TABS.some(t => t.id === sub)) S.sub = sub;
  const changed = S.page !== page;
  S.page = page;
  setHash();
  $$("section.page[id^='p-']").forEach(p => p.classList.toggle("hide", p.id !== "p-" + page));
  $("#pageTitle").textContent = PAGES[page].title;
  $("#pageSub").textContent = PAGES[page].sub;
  document.title = PAGES[page].title + " · Poketenashi";
  renderNav(true);
  $("#rail").classList.remove("open");
  if (changed) window.scrollTo(0, 0);
  if (page !== "live") stopStream();
  if (page === "dash") { renderDashTiles(); renderKpis(); renderRuleBars(); renderCamBars(); drawHourly(); refreshLatest(); renderDashSys(); }
  if (page === "live") renderLive();
  if (page === "evid") renderEvidence(true);
  if (page === "cams") renderCams();
  if (page === "zones") ZE.open();
  if (page === "set") renderSettings();
  if (page === "tv") renderTv();
}
window.addEventListener("hashchange", () => {
  const p = parseHash();
  if (p.page !== S.page || (p.page === "set" && p.sub && p.sub !== S.sub)) go(p.page, p.sub);
});

let navHtml = "";
function navCount(n) {
  if (n === "evid" && S.stats && S.stats.today) return `<span class="count" title="Violations today">${S.stats.today}</span>`;
  if (n === "cams") {
    const off = S.cams.filter(c => c.enabled && !c.camera_ok).length;
    if (off && S.up) return `<span class="count bad" title="${off} camera(s) offline">${off}</span>`;
  }
  return "";
}
function renderNav(force) {
  const html = NAV.map(n => typeof n === "object" ? `<div class="nav-sec">${n.sec}</div>` :
    `<button class="nav${n === S.page ? " on" : ""}" data-go="${n}"${n === S.page ? ' aria-current="page"' : ""}>` +
    `${icon(PAGES[n].icon)}<span>${PAGES[n].title}</span>${navCount(n)}</button>`).join("");
  if (!force && html === navHtml) return;
  navHtml = html;
  $("#nav").innerHTML = html;
}

document.addEventListener("click", e => {
  const g = e.target.closest("[data-go]");
  if (g) { e.preventDefault(); go(g.getAttribute("data-go"), g.getAttribute("data-sub")); return; }
  const cp = e.target.closest("[data-copy]");
  if (cp) {
    e.preventDefault();
    toast(UI.copy(cp.getAttribute("data-copy")) ? "Copied to the clipboard." : "Could not copy - select the text instead.",
          "success");
    return;
  }
  const sh = e.target.closest("[data-shot]");
  if (sh) {
    const host = sh.closest("[data-rows]");
    if (host && host.__rows) viewer(host.__rows, +sh.getAttribute("data-shot"));
    return;
  }
  const t = e.target.closest("[data-tile]");
  if (t) { S.sel = t.getAttribute("data-tile"); store.set("pk_cam", S.sel); S.grid = false; go("live"); }
});
$("#menuBtn").onclick = e => { e.stopPropagation(); $("#rail").classList.toggle("open"); };
// On a narrow screen the menu is a drawer: a click anywhere else closes it.
document.addEventListener("click", e => {
  const rail = $("#rail");
  if (rail.classList.contains("open") && !rail.contains(e.target)) rail.classList.remove("open");
});

/* ======================================================================
   Who am I - login, user menu, admin guard
   ====================================================================== */
let afterLogin = null;
function loadMe() {
  return api.get("/api/me").then(m => {
    const was = S.me.user, couldView = S.me.may_view;
    S.me = m;
    renderUser();
    updateWall();
    if (m.may_view && couldView === false) { pollState(); refreshStats(); pollAlerts(); refreshSys(); }
    if (was && !m.user) toast("Your login ended (the monitor restarted or you were logged out elsewhere).", "warning");
    if (S.page === "set") renderSettings();
    if (S.page === "cams") renderCams();
    if (S.page === "zones") ZE.authChanged();
  }).catch(() => {});
}
// Looking needs a login since 7 Oct 2026 (or a TV screen's link): without one,
// a wall with a login button instead of a login popup every 2 seconds.
function updateWall() {
  const need = S.me.may_view === false && !S.me.setup_needed;
  let w = $("#wall");
  if (need && !w) {
    w = document.createElement("div");
    w.id = "wall";
    w.innerHTML = `<div class="card wall-card"><div class="card-body" style="text-align:center;display:flex;flex-direction:column;gap:14px;align-items:center">
      <span class="avatar" style="width:52px;height:52px">${icon("lock")}</span>
      <h2>Log in to see your cameras</h2>
      <p class="muted">Every person sees the cameras the administrator gave them. A TV screen uses its own link (Settings → TV screens).</p>
      <button class="btn btn-primary" id="wallLogin">${icon("login")}<span>Log in</span></button></div></div>`;
    document.body.appendChild(w);
    $("#wallLogin").onclick = () => login("Log in to see your cameras.");
  }
  if (w) w.classList.toggle("hide", !need);
  document.body.classList.toggle("walled", need);
}
const initials = n => String(n || "?").replace(/[^A-Za-z0-9]/g, "").slice(0, 2).toUpperCase() || "?";
function renderUser() {
  const box = $("#user"), m = S.me;
  if (m.user) {
    box.innerHTML = `<button class="userbtn" id="userBtn" aria-haspopup="menu" aria-expanded="false" title="${esc(m.user)} (${esc(m.role)})">` +
      `<span class="avatar">${esc(initials(m.user))}</span><span class="nm">${esc(m.user)}</span>${icon("chevron-down", "sm")}</button>`;
    $("#userBtn").onclick = e => { e.stopPropagation(); toggleUserMenu(); };
  } else {
    box.innerHTML = (m.display ? `<span class="pill" title="This screen's link shows its own cameras">${icon("tv", "sm")}${esc(m.display)}</span> ` : "") +
      `<button class="btn btn-secondary btn-sm" id="loginBtn">${icon("login")}<span>Log in</span></button>`;
    $("#loginBtn").onclick = () => login();
  }
}
function toggleUserMenu() {
  const open = $("#userMenu");
  if (open) { open.remove(); return; }
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.id = "userMenu";
  menu.setAttribute("role", "menu");
  menu.innerHTML = `<div class="who"><span class="avatar">${esc(initials(S.me.user))}</span><div><b>${esc(S.me.user)}</b>` +
    `<span>${S.me.role === "admin" ? "Administrator" : "Viewer"}</span></div></div>` +
    `<button role="menuitem" data-act="account">${icon("key")}My account &amp; password</button>` +
    (S.me.role === "admin" ? `<button role="menuitem" data-act="users">${icon("users")}Users</button>` +
                             `<button role="menuitem" data-act="audit">${icon("history")}Audit log</button>` : "") +
    `<button role="menuitem" data-act="logout">${icon("logout")}Log out</button>`;
  $("#user").appendChild(menu);
  menu.onclick = e => {
    const b = e.target.closest("button");
    if (!b) return;
    menu.remove();
    const a = b.getAttribute("data-act");
    if (a === "logout") logout(); else go("set", a);
  };
}
document.addEventListener("click", e => {
  const m = $("#userMenu");
  if (m && !m.contains(e.target) && !e.target.closest("#userBtn")) m.remove();
});
function logout() {
  api.post("/api/logout", {}).then(() => { toast("You are logged out.", "info"); return loadMe(); })
    .catch(err => toast(err.message, "error"));
}

function login(reason, then) {
  if (then) afterLogin = then;
  if (S.me.setup_needed) return setupAdmin();
  if ($("#lgForm")) return;
  modal({
    title: "Log in", subtitle: reason || "Viewers see their cameras; admins change cameras, zones, settings and users.",
    icon: "lock", size: "sm",
    body: `<div id="lgForm" style="display:flex;flex-direction:column;gap:14px">
      <div class="field"><label for="lgU">User name</label><input class="input" id="lgU" autocomplete="username" autofocus></div>
      <div class="field"><label for="lgP">Password</label><input class="input" id="lgP" type="password" autocomplete="current-password"></div></div>`,
    onClose: r => { if (r !== true) afterLogin = null; },
    actions: [
      {label: "Cancel", onClick: m => m.close()},
      {label: "Log in", kind: "primary", icon: "login", onClick: m => {
        const u = $("#lgU").value.trim(), p = $("#lgP").value;
        if (!u || !p) { m.error("Enter your user name and password."); return; }
        m.error("");
        m.busy(true, "Logging in…");
        api.post("/api/login", {user: u, password: p}).then(d => {
          m.busy(false);
          m.close(true);
          toast(`Logged in as ${d.user} (${d.role === "admin" ? "administrator" : "viewer"}).`, "success");
          return loadMe();
        }).then(() => {
          const f = afterLogin;
          afterLogin = null;
          if (f && S.me.can_admin) f();
        }).catch(err => { m.busy(false); m.error(err.message); });
      }}
    ]
  });
}
function setupAdmin() {
  if (!S.me.local) { toast("No users exist yet: create the admin account on the monitor PC itself.", "warning"); return; }
  modal({
    title: "Create the admin account", subtitle: "The first user can change everything and add more users.",
    icon: "shield", size: "sm",
    body: `<div class="field"><label for="suU">User name</label><input class="input" id="suU" autocomplete="username"></div>
      <div class="field"><label for="suP">Password</label><input class="input" id="suP" type="password" autocomplete="new-password"><span class="help">At least 6 characters</span></div>
      <div class="field"><label for="suP2">Password again</label><input class="input" id="suP2" type="password" autocomplete="new-password"></div>`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: "Create admin", kind: "primary", icon: "check", onClick: m => {
        const u = $("#suU").value.trim(), p = $("#suP").value;
        if (p !== $("#suP2").value) { m.error("The two passwords differ."); return; }
        m.busy(true, "Creating…");
        api.post("/api/setup", {user: u, password: p}).then(() => {
          m.busy(false); m.close(true); toast("Admin account created - you are logged in.", "success"); return loadMe();
        }).catch(err => { m.busy(false); m.error(err.message); });
      }}]
  });
}
// Wrap an admin action: run it, or ask for a login first and run it after.
function admin(fn, reason) {
  return function () {
    const args = arguments;
    if (S.me.can_admin) return fn.apply(null, args);
    if (S.me.user) { toast("Only an administrator can do this - you are logged in as a viewer.", "warning"); return; }
    login(reason || "Log in as an administrator to make changes.", () => fn.apply(null, args));
  };
}
api.onLogin = msg => loadMe().then(() => { if (S.me.may_view !== false) login(msg); });

/* ======================================================================
   Polling
   ====================================================================== */
function pollState() {
  return api.get("/api/state" + (S.sel ? "?cam=" + encodeURIComponent(S.sel) : "")).then(s => {
    const restarted = S.uptime !== null && s.uptime_sec < S.uptime;
    const wasDown = !S.up && S.everUp;          // a real outage, not the first answer
    S.up = S.everUp = true;
    S.uptime = s.uptime_sec;
    S.st = s;
    S.cams = s.cameras || [];
    const learned = learnRules(ruleCatalog(s).list);
    if (learned && S.page === "evid") renderEvidenceFilters();
    if (!S.sel || !camById(S.sel)) S.sel = S.cams.length ? S.cams[0].id : "";
    const sig = JSON.stringify(S.cams.map(c => [c.id, c.name, c.enabled, c.detect, c.zones, c.source]));
    if (sig !== S.camSig) { S.camSig = sig; camerasChanged(); }
    if (restarted) { toast("The monitor restarted.", "info"); loadMe(); }
    if (restarted || wasDown) reopenStream();
    updateStatus();
  }).catch(() => { S.up = false; updateStatus(); });
}
function camerasChanged() {
  if (S.page === "dash") { renderDashTiles(); renderCamBars(); }
  if (S.page === "live") renderLive();
  if (S.page === "evid") renderEvidenceFilters();
  if (S.page === "cams") renderCams();
  if (S.page === "zones") ZE.fillCams();
  if (S.page === "tv") renderTv();
}
function updateStatus() {
  const on = S.cams.filter(c => c.enabled), up = on.filter(c => c.camera_ok);
  const top = $("#topStatus");
  if (!S.up) {
    top.className = "pill bad";
    top.innerHTML = `<span class="dot bad"></span><span>Monitor unreachable</span>`;
  } else if (on.length && up.length === on.length) {
    top.className = "pill ok";
    top.innerHTML = `<span class="dot ok"></span><span>Monitoring · ${fmt.plural(on.length, "camera")}</span>`;
  } else {
    top.className = "pill warn";
    top.innerHTML = `<span class="dot warn"></span><span>${on.length - up.length} of ${on.length} cameras offline</span>`;
  }
  $("#sysDot").className = "dot " + (S.up ? "ok" : "bad");
  $("#sysText").textContent = S.up ? "Monitor running" : "Monitor unreachable";
  $("#sysUp").textContent = S.up && S.uptime !== null ? "Up " + fmt.dur(S.uptime) : "";
  $("#sysVer").textContent = S.me.version ? "Version " + S.me.version : "";
  renderNav();
  updateCamBits();
  if (S.page === "dash") { renderKpis(); renderRules(); }
  if (S.page === "live") updateLive();
}
function updateCamBits() {
  S.cams.forEach(c => {
    const ok = c.enabled && c.camera_ok && S.up;
    $$(`[data-cam-dot="${c.id}"]`).forEach(el => { el.className = "dot " + (!c.enabled ? "" : ok ? "ok" : "bad"); });
    $$(`[data-cam-meta="${c.id}"]`).forEach(el => {
      el.textContent = !c.enabled ? "switched off" : ok ? `${c.persons} in view · ${(+c.fps || 0).toFixed(1)} fps` : "offline";
    });
    $$(`[data-cam-status="${c.id}"]`).forEach(el => { el.innerHTML = statusPill(c); });
    $$(`[data-cam-facts="${c.id}"]`).forEach(el => {
      el.innerHTML = ok
        ? `<span>${icon("resolution")}${c.size ? c.size[0] + "×" + c.size[1] : "—"}</span>` +
          `<span>${icon("gauge")}${c.detect ? (+c.fps || 0).toFixed(1) + " fps detection" : "no detection"}</span>` +
          `<span>${icon("live")}${(+c.view_fps || 0).toFixed(0)} fps picture</span>` +
          `<span>${icon("people")}${fmt.plural(c.persons || 0, "person", "people")} in view</span>` +
          (c.gray_pct >= 1 ? `<span class="warn-text" title="Frames that arrived broken (gray) in the last minute and were skipped">${icon("alert")}${c.gray_pct}% broken frames skipped</span>` : "")
        : `<span>${icon("zones")}Zones: ${c.zones && c.zones.length ? esc(c.zones.join(", ")) : "none drawn"}</span>`;
    });
    $$(`[data-cam-warn="${c.id}"]`).forEach(el => {
      const html = ok && c.gray_pct >= 10
        ? `<div class="note warn">${icon("alert")}<div><b>${c.gray_pct}% of the frames arrive broken (gray)</b> and are skipped - the PC is too busy to keep up with this stream. A sub stream needs much less processing: Edit → Find stream.</div></div>` : "";
      if (el.__h !== html) { el.__h = html; el.innerHTML = html; }
    });
    $$(`[data-cam-err="${c.id}"]`).forEach(el => {
      const err = c.enabled && !ok ? (S.up ? (c.last_error || "connecting…") : "the monitor is not reachable") : "";
      const h = err ? hintFor(err) : "";
      const html = err ? `<div class="note bad">${icon("alert")}<div><b>Offline:</b> ${esc(err)}${h ? `<br><span class="dim">${esc(h)}</span>` : ""}</div></div>` : "";
      if (el.__h !== html) { el.__h = html; el.innerHTML = html; }
    });
    $$(`[data-ph="${c.id}"]`).forEach(el => {
      const img = el.parentNode.querySelector("img");
      el.classList.toggle("hide", !!(c.enabled && img && img.naturalWidth > 0));
      const span = el.querySelector("span");
      if (span) span.textContent = !c.enabled ? "Switched off" : ok ? "Waiting for the picture…" : "Offline - no picture";
    });
  });
}
// Camera pictures in tiles and cards: one JPEG at a time, never a stream (a
// browser holds only ~6 connections per server and the live view needs one).
function refreshSnaps() {
  const root = S.page === "dash" ? "#dashTiles" : S.page === "cams" ? "#camList" : (S.page === "live" && S.grid ? "#liveGrid" : null);
  if (!root || document.hidden) return;
  $$(root + " img[data-snap]").forEach(img => {
    const c = camById(img.getAttribute("data-snap"));
    if (img.__busy || !c || !c.camera_ok || !S.up) return;
    img.__busy = true;
    const n = new Image();
    n.onload = () => {
      img.src = n.src;
      img.__busy = false;
      const ph = img.parentNode.querySelector(".ph");
      if (ph) ph.classList.add("hide");
    };
    n.onerror = () => { img.__busy = false; };
    n.src = "/api/frame?cam=" + encodeURIComponent(c.id) + "&t=" + Date.now();
  });
}

function refreshStats() {
  return api.get("/api/stats").then(st => {
    S.stats = st;
    const sig = JSON.stringify(st);
    if (sig === S.statsSig) return;
    S.statsSig = sig;
    renderNav();
    if (S.page === "dash") { renderKpis(); renderRuleBars(); renderCamBars(); drawHourly(); }
  }).catch(() => {});
}
function refreshSys() {
  return api.get("/api/settings").then(s => {
    S.sys = s;
    if (S.page === "dash") renderDashSys();
  }).catch(() => {});
}
function pollAlerts() {
  return api.get("/api/alerts?since=" + Math.max(S.alertSeq, 0)).then(r => {
    if (S.alertSeq < 0 || r.seq < S.alertSeq) { S.alertSeq = r.seq; return; }   // first poll / restart
    S.alertSeq = r.seq;
    if (!r.alerts.length) return;
    r.alerts.slice(-3).forEach(showAlert);
    if (S.sound) UI.beep();
    clearTimeout(S.alertTimer);
    S.alertTimer = setTimeout(() => {
      refreshStats();
      if (S.page === "dash") refreshLatest();
      if (S.page === "evid") loadEvidence();
      if (S.page === "live" && !S.grid) refreshLiveRecent();
    }, 1200);
  }).catch(() => {});
}
function showAlert(a) {
  const box = $("#alerts");
  const el = document.createElement("div");
  el.className = "alert";
  el.setAttribute("role", "status");
  const pic = evidenceImg(a);
  el.innerHTML = `<span class="stripe" style="background:${ruleColor(a.rule)}"></span><div class="body">` +
    `<div class="r">${badge(a.rule)}<span>${esc(a.camera_name || camName(a.camera))} · person ${who(a)} · ${esc(fmt.hms(a.time))}</span></div>` +
    `<div class="d">${esc(RULE_NAMES[a.rule] || a.desc || a.rule)}</div></div>` +
    (pic ? `<img src="${pic}" alt="" onerror="this.remove()">` : "");
  el.onclick = () => { el.remove(); viewer([a], 0); };
  box.appendChild(el);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => { if (el.parentNode) el.remove(); }, 10000);
}

/* ======================================================================
   Shared pieces: tiles, evidence cards, the viewer, bars, chart
   ====================================================================== */
function tileHtml(c) {
  return `<button class="tile" data-tile="${esc(c.id)}" title="Open the live view of ${esc(c.name)}">
    <div class="stage">${c.enabled ? `<img data-snap="${esc(c.id)}" alt="">` : ""}
      <div class="ph" data-ph="${esc(c.id)}">${icon(c.enabled ? "live" : "power")}<span>${c.enabled ? "Waiting for the picture…" : "Switched off"}</span></div></div>
    <div class="tile-meta"><span class="dot" data-cam-dot="${esc(c.id)}"></span><b>${esc(c.name)}</b>
      <span class="r num" data-cam-meta="${esc(c.id)}"></span></div></button>`;
}
function shotHtml(v, i, showCam) {
  const pic = evidenceImg(v);
  return `<button class="shot" data-shot="${i}" title="${esc(RULE_NAMES[v.rule] || v.desc)} - ${esc(v.time)}">
    <div class="pic">${pic ? `<img src="${pic}" loading="lazy" alt="" onerror="this.style.visibility='hidden'">` : ""}
      ${showCam ? `<span class="cam">${esc(v.camera_name || camName(v.camera))}</span>` : ""}
      ${v.clip ? `<span class="play">${icon("play")}</span>` : ""}</div>
    <div class="m">${badge(v.rule)}<span>${who(v)}</span><span class="t">${esc(fmt.hms(v.time))}</span></div></button>`;
}
function renderShots(el, rows, emptyMsg, oneCam) {
  const sig = rows.map(v => v.time + v.track_id + v.rule + (v.snapshot || v.poster || "") + (v.clip || "")).join("|");
  if (el.__sig === sig) return;
  el.__sig = sig;
  el.__rows = rows;
  el.setAttribute("data-rows", "");
  el.innerHTML = rows.length ? rows.map((v, i) => shotHtml(v, i, !oneCam && S.cams.length > 1)).join("") : emptyBox("check", emptyMsg);
}

function viewer(list, start) {
  let idx = start;
  const m = modal({
    title: "", size: "xl", cls: "viewer",
    body: `<div id="vwMedia"></div><div class="viewer-meta" id="vwMeta"></div>`,
    actions: [
      {label: "Previous", icon: "chevron-left", key: "prev", onClick: () => show(idx - 1)},
      {label: "Next", icon: "chevron-right", key: "next", onClick: () => show(idx + 1)},
      {label: "Download", icon: "download", key: "dl", onClick: () => download()},
      {label: "Close", kind: "primary", onClick: mm => mm.close()}
    ],
    onClose: () => document.removeEventListener("keydown", keys)
  });
  const depth = UI.openPopups();
  const keys = e => {
    // Only while this viewer is the top popup, and not while a <video> has
    // the focus (there the arrows seek).
    if (m.closed || UI.openPopups() !== depth || (e.target && e.target.tagName === "VIDEO")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); show(idx - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); show(idx + 1); }
  };
  document.addEventListener("keydown", keys);
  if (list.length < 2) { m.button("prev").classList.add("hide"); m.button("next").classList.add("hide"); }
  function show(j) {
    if (j < 0 || j >= list.length) return;
    idx = j;
    const v = list[j], pic = evidenceImg(v);
    $("h2", m.el).innerHTML = `${badge(v.rule)} ${esc(RULE_NAMES[v.rule] || v.desc || v.rule)}`;
    $(".modal-titles p", m.el) || $(".modal-titles", m.el).insertAdjacentHTML("beforeend", "<p></p>");
    $(".modal-titles p", m.el).textContent = `${v.camera_name || camName(v.camera)} · ${v.time}`;
    const media = $("#vwMedia", m.el);
    if (v.clip) {
      media.innerHTML = `<video src="/clips/${encodeURIComponent(v.clip)}" ${pic ? `poster="${pic}"` : ""} controls autoplay loop playsinline muted></video>`;
      $("video", media).onerror = () => {
        media.innerHTML = (pic ? `<img src="${pic}" alt="">` : "") +
          `<div class="note">${icon("info")}<div>The clip is still being written (it lands a few seconds after the alert). <a href="#" id="vwRetry">Try again</a></div></div>`;
        const r = $("#vwRetry", media);
        if (r) r.onclick = e => { e.preventDefault(); show(idx); };
      };
    } else {
      media.innerHTML = pic ? `<img src="${pic}" alt="">` : emptyBox("evidence", "No picture was saved for this event.");
    }
    $("#vwMeta", m.el).innerHTML = `<span>${icon("user")}Person ${who(v)}</span><span>${icon("camera")}${esc(v.camera_name || camName(v.camera))}</span>` +
      `<span>${icon("calendar")}${esc(v.time)}</span>${list.length > 1 ? `<span class="muted">${j + 1} of ${list.length}</span>` : ""}`;
    m.button("prev").disabled = j <= 0;
    m.button("next").disabled = j >= list.length - 1;
    m.button("dl").disabled = !(v.clip || pic);
  }
  function download() {
    const v = list[idx];
    const a = document.createElement("a");
    a.href = v.clip ? "/clips/" + encodeURIComponent(v.clip) : evidenceImg(v);
    a.download = v.clip || v.snapshot || v.poster || "evidence";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
  show(idx);
}

function bars(rows, emptyMsg, wide) {
  if (!rows.length) return emptyBox("check", emptyMsg);
  const max = Math.max(1, ...rows.map(r => r.count)), tot = rows.reduce((s, r) => s + r.count, 0) || 1;
  return rows.map(r => `<div class="bar-row${wide ? " wide" : ""}" title="${esc(r.name)}: ${r.count}">
      <span class="lbl">${r.label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.count / max * 100).toFixed(1)}%;background:${r.color}"></div></div>
      <span class="val">${fmt.n(r.count)}</span>
      ${wide ? "" : `<span class="nm">${esc(r.name)} · ${Math.round(r.count / tot * 100)}%</span>`}</div>`).join("");
}
let tipEl = null;
function tip(e, html) {
  if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
  if (!html) { tipEl.style.display = "none"; return; }
  tipEl.innerHTML = html;
  tipEl.style.display = "block";
  tipEl.style.left = Math.min(e.clientX + 14, innerWidth - tipEl.offsetWidth - 10) + "px";
  tipEl.style.top = Math.max(e.clientY - 44, 8) + "px";
}
function niceMax(v) {
  if (v <= 4) return 4;
  const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

/* ======================================================================
   Dashboard
   ====================================================================== */
function renderDashTiles() {
  const el = $("#dashTiles");
  el.innerHTML = S.cams.length ? S.cams.map(tileHtml).join("") : emptyBox("camera", "No cameras yet - add one under Cameras.");
  updateCamBits();
  refreshSnaps();
}
function renderKpis() {
  const st = S.stats;
  if (st) {
    $("#kToday").textContent = fmt.n(st.today);
    const d = st.today - st.yesterday_so_far;
    $("#kTodayF").innerHTML = d > 0 ? `<span class="up">${icon("arrow-up")}${d} more</span>&nbsp;than yesterday by this time`
      : d < 0 ? `<span class="down">${icon("arrow-down")}${-d} fewer</span>&nbsp;than yesterday by this time`
      : "The same as yesterday by this time";
    $("#k24").textContent = fmt.n(st.last24);
    const peak = st.by_hour.reduce((a, b) => b.count > a.count ? b : a, {count: 0, label: ""});
    $("#k24F").textContent = peak.count ? `Busiest hour: ${peak.label} (${peak.count})` : "No violations in the last 24 hours";
  }
  const on = S.cams.filter(c => c.enabled), up = on.filter(c => c.camera_ok);
  const down = on.filter(c => !c.camera_ok).map(c => c.name);
  $("#kCams").innerHTML = S.up ? `${up.length}<small> / ${on.length}</small>` : "—";
  $("#kCamsF").innerHTML = !S.up ? "The monitor is not reachable"
    : down.length ? `<span class="up">${icon("alert")}</span>Offline: ${esc(down.join(", "))}`
    : on.length ? "All cameras live" : "No camera is switched on";
  const ppl = S.cams.reduce((n, c) => n + (c.persons || 0), 0);
  $("#kPeople").textContent = S.up ? fmt.n(ppl) : "—";
  $("#kPeopleF").textContent = S.up ? `Right now, on ${fmt.plural(up.length, "live camera")}` : "";
}
// What the monitor is really judging with. A rule that cannot fire - its zones
// are not drawn, it is switched off, the camera is switched off - is named with
// the reason the server gives, so it is never silently absent from the row.
function renderRules() {
  const el = $("#dashRules");
  if (!el) return;
  const cat = ruleCatalog(S.st), list = camChecks_(cat.list);
  const cam = camById(S.sel);
  const where = cam ? cam.name : "this camera";
  const chips = list.map(r => `<span class="chip">` +
    (r.active === null ? "" : `<span class="dot ${r.active ? "ok" : "warn"}"></span>`) +
    `${esc(r.rule)}${r.label ? " · " + esc(r.label) : ""}</span>`).join("");
  if (cat.mode === "none") {
    $("#rulesSub").textContent = "What this camera is really running";
    el.innerHTML = `<div class="help">` + (S.up
      ? "This monitor did not report its rules."
      : "The monitor is not reachable.") + `</div>`;
    return;
  }
  if (cat.mode === "codes") {          // an older monitor: codes only, no status
    $("#rulesSub").textContent = `Rules on ${where} (the monitor does not report per-rule status)`;
    el.innerHTML = `<div class="chips">${chips}</div>` +
      `<div class="note info" style="margin-top:10px">${icon("info")}<div>` +
      `The monitor is running older code: it reports the ${list.length} rules that are switched on, not which of them can actually fire. ` +
      `Restart it to see a rule with no zone drawn named as off instead of listed.` +
      `<div class="help">This page is served from disk, so it can be newer than the monitor that answers it - it renders either.</div>` +
      `</div></div>`;
    return;
  }
  const on = list.filter(r => r.active), off = list.filter(r => !r.active);
  $("#rulesSub").textContent = `${on.length} of ${list.length} running on ${where}`;
  el.innerHTML = `<div class="chips">` + (on.length ? on.map(r => `<span class="chip">` +
      `<span class="dot ok"></span>${esc(r.rule)} · ${esc(r.label)}</span>`).join("")
      : `<span class="help">No rule is running on this camera.</span>`) + `</div>` +
    (off.length ? `<div class="note warn" style="margin-top:10px">${icon("alert")}<div>` +
      off.map(r => `<b>${esc(r.rule)} · ${esc(r.label)}</b> — ${esc(r.reason)}`).join("<br>") +
      `<div class="help">A rule with no zone drawn never judges anyone. Draw the zone in the Zone editor, or untick the rule in Cameras → Rules….</div>` +
      `</div></div>` : "");
}
function renderRuleBars() {
  const st = S.stats;
  if (!st) return;
  const rows = (S.ruleView === "all" ? st.by_rule : st.today_by_rule)
    .map(r => ({label: badge(r.rule), name: RULE_NAMES[r.rule] || r.rule, count: r.count, color: ruleColor(r.rule)}));
  $("#ruleBars").innerHTML = bars(rows, S.ruleView === "all" ? "No violations recorded yet." : "No violations today.");
}
$("#ruleSeg").onclick = e => {
  const b = e.target.closest("button");
  if (!b) return;
  S.ruleView = b.getAttribute("data-r");
  $$("#ruleSeg button").forEach(x => x.classList.toggle("on", x === b));
  renderRuleBars();
};
function renderCamBars() {
  const st = S.stats;
  if (!st) return;
  const rows = Object.keys(st.by_camera).map(id => ({label: esc(camName(id)), name: camName(id), count: st.by_camera[id], color: "var(--accent)"}))
    .sort((a, b) => b.count - a.count);
  $("#camBars").innerHTML = bars(rows, "No violations recorded yet.", true);
}
function drawHourly() {
  const el = $("#hourly");
  if (!el || !S.stats || S.page !== "dash" || !el.clientWidth) return;
  const data = S.stats.by_hour;
  if (!data.some(b => b.count)) { el.innerHTML = emptyBox("activity", "No violations in the last 24 hours."); return; }
  const W = el.clientWidth, H = 190, L = 34, B = 24, T = 10, pw = W - L, ph = H - B - T;
  const max = niceMax(Math.max(...data.map(b => b.count))), slot = pw / data.length;
  const bw = Math.max(4, Math.min(30, slot - 4));
  let g = "";
  [0, max / 2, max].forEach(t => {
    const y = T + ph - t / max * ph;
    g += `<line class="grid" x1="${L}" x2="${W}" y1="${y}" y2="${y}"/><text class="axis" x="${L - 8}" y="${y + 3.5}" text-anchor="end">${Math.round(t)}</text>`;
  });
  const every = W > 820 ? 3 : 6;
  data.forEach((b, i) => {
    const x = L + i * slot + (slot - bw) / 2, h = b.count ? Math.max(2, b.count / max * ph) : 0, y = T + ph - h, r = Math.min(4, bw / 2, h);
    g += `<rect class="hit" data-i="${i}" x="${L + i * slot}" y="${T}" width="${slot}" height="${ph}" fill="transparent"/>`;
    if (h) g += `<path class="col" pointer-events="none" d="M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${y + h}Z"/>`;
    if (i % every === 0) g += `<text class="axis" x="${L + i * slot + slot / 2}" y="${H - 6}" text-anchor="middle">${b.label}</text>`;
  });
  el.innerHTML = `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Violations per hour over the last 24 hours">${g}</svg>`;
  const svg = $("svg", el);
  svg.addEventListener("mousemove", e => {
    const t = e.target.closest(".hit");
    if (!t) { tip(e, ""); return; }
    const b = data[+t.getAttribute("data-i")];
    tip(e, `<b>${b.count}</b> ${b.count === 1 ? "violation" : "violations"} · ${b.label}–${String((+b.label.slice(0, 2) + 1) % 24).padStart(2, "0")}:00`);
  });
  svg.addEventListener("mouseleave", e => tip(e, ""));
}
if (window.ResizeObserver) new ResizeObserver(() => drawHourly()).observe($("#hourly"));
else window.addEventListener("resize", drawHourly);
function refreshLatest() {
  return api.get("/api/violations?limit=8").then(rows => renderShots($("#latest"), rows, "No violations recorded yet."))
    .catch(() => {});
}
function renderDashSys() {
  const s = S.sys, el = $("#dashSys");
  if (!s) { el.innerHTML = `<div class="skel" style="height:58px"></div><div class="skel" style="height:58px"></div>`; return; }
  const items = [
    ["Running for", fmt.dur(s.system.uptime_sec), "since " + s.system.started.slice(5, 16)],
    ["Disk free", s.disk ? s.disk.free_gb + " GB" : "—", s.disk ? "of " + s.disk.total_gb + " GB" : ""],
    ["Evidence stored", fmt.mb(s.evidence.mb), fmt.plural(s.evidence.files, "file")],
    ["Kept for", s.retention.days > 0 ? fmt.plural(+s.retention.days, "day") : "ever", s.retention.days > 0 ? "then deleted automatically" : "nothing is deleted"],
  ];
  el.innerHTML = items.map(([l, v, sub]) => `<div class="stat"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="help">${esc(sub)}</div></div>`).join("");
}

// Who broke the rule: the employee code face ID gave the person (7 Oct), else
// the tracker's number for them on that camera.
function who(v) { return v.person ? esc(v.person) : "#" + esc(v.track_id); }

/* ======================================================================
   Live view
   ====================================================================== */
let streamCam = null, liveOkPrev = null;
function renderLive() {
  $("#liveSeg").innerHTML = S.cams.map(c =>
    `<button data-cam="${esc(c.id)}" class="${!S.grid && c.id === S.sel ? "on" : ""}" role="tab" aria-selected="${!S.grid && c.id === S.sel}"><span class="dot ${!S.grid && c.id === S.sel ? "ok" : ""}"></span>${esc(c.name)}</button>`).join("") +
    (S.cams.length > 1 ? `<button data-cam="*" class="${S.grid ? "on" : ""}">${icon("grid", "sm")}All cameras</button>` : "");
  $("#liveOne").classList.toggle("hide", S.grid);
  $("#liveGrid").classList.toggle("hide", !S.grid);
  ["#snapBtn", "#fullBtn", "#liveRecentCard"].forEach(s => $(s).classList.toggle("hide", S.grid));
  if (S.grid) {
    stopStream();
    $("#liveGrid").innerHTML = S.cams.map(tileHtml).join("");
    refreshSnaps();
  } else {
    startStream();
    refreshLiveRecent();
  }
  updateCamBits();
  updateLive();
}
$("#liveSeg").onclick = e => {
  const b = e.target.closest("button");
  if (!b) return;
  const id = b.getAttribute("data-cam");
  if (id === "*") S.grid = true;
  else {
    // a different camera: drop the old picture at once, so it can never show
    // under the new tab while the new stream (if any) starts
    if (id !== S.sel) stopStream();
    S.grid = false; S.sel = id; store.set("pk_cam", id); liveOkPrev = null;
  }
  renderLive();
  pollState();
};
function startStream(force) {
  const c = camById(S.sel);
  if (!c || S.page !== "live" || S.grid) return;
  if (!c.enabled) { stopStream(); return; }      // switched off: no stream, a black stage
  if (!force && streamCam === S.sel) return;
  streamCam = S.sel;
  const img = $("#liveImg");
  img.src = "/api/stream?cam=" + encodeURIComponent(S.sel) + "&t=" + Date.now();
}
function stopStream() {
  streamCam = null;
  const img = $("#liveImg");
  if (img && img.getAttribute("src")) img.removeAttribute("src");
}
function reopenStream() { if (S.page === "live" && !S.grid) startStream(true); }
// How much of the software is judging this camera, right beside where the
// operator watches it. The rules that are NOT running are named, with the
// server's reason in the tooltip.
function rulesChip() {
  const cat = ruleCatalog(S.st), list = camChecks_(cat.list);
  if (cat.mode === "none") return "";
  if (cat.mode === "codes") {          // older monitor: say so, do not invent status
    return `<span class="mut" title="This monitor reports the rules that are switched on, ` +
      `not which can actually fire - it is running older code.">` +
      `${list.length} rules, status not reported</span>`;
  }
  const off = list.filter(r => !r.active);
  const title = off.length ? off.map(r => `${r.rule}: ${r.reason}`).join("; ") : "Every rule has what it needs";
  return `<span class="${off.length ? "warn-text" : ""}" title="${esc(title)}">` +
    `${icon("check")}${list.length - off.length} of ${list.length} rules running` +
    (off.length ? ` (off: ${off.map(r => esc(r.rule)).join(", ")})` : "") + `</span>`;
}
function updateLive() {
  if (S.page !== "live" || S.grid) return;
  const c = camById(S.sel);
  if (!c) { $("#livePh span").textContent = "No camera."; return; }
  const ok = !!(c.enabled && c.camera_ok && S.up);
  if (!c.enabled) stopStream();                     // switched off while watched
  else if (streamCam !== S.sel) startStream();      // switched back on
  if (ok && liveOkPrev === false) reopenStream();
  liveOkPrev = ok;
  const img = $("#liveImg"), has = !!(c.enabled && img.getAttribute("src") && img.naturalWidth > 0);
  // no picture: hide the <img> too, or the browser draws its broken-image
  // icon and alt text ("Live camera picture") on the stage
  img.style.visibility = has ? "" : "hidden";
  $("#livePh").classList.toggle("hide", has);
  // a switched-off camera is just a black stage - its tab dot already says it is off
  $("#livePh").classList.toggle("blank", !c.enabled);
  $("#livePh span").textContent = !c.enabled ? "" : ok ? "Waiting for the picture…"
    : "Camera offline: " + (S.up ? (c.last_error || "connecting…") : "the monitor is not reachable");
  const tag = $("#liveTag");
  tag.classList.toggle("hide", !has);
  tag.innerHTML = `<span class="dot ${ok ? "live" : "bad"}"></span>${esc(c.name)} · ${ok ? "LIVE" : "OFFLINE - last picture"}`;
  $("#liveInfo").innerHTML = [
    c.size ? `<span>${icon("resolution")}${c.size[0]}×${c.size[1]}</span>` : "",
    `<span>${icon("gauge")}${c.detect ? (ok ? (+c.fps || 0).toFixed(1) + " fps detection" : "detection paused") : "rules off - live view only"}</span>`,
    rulesChip(),
    `<span>${icon("live")}${ok ? (+c.view_fps || 0).toFixed(0) : 0} fps picture</span>`,
    `<span>${icon("people")}${ok ? fmt.plural(c.persons || 0, "person", "people") : "—"} in view</span>`,
    ok && c.gray_pct >= 1 ? `<span class="warn-text">${icon("alert")}${c.gray_pct}% broken frames skipped</span>` : "",
    `<span class="mono" title="${esc(c.source)}">${icon("link")}${esc(c.source)}</span>`
  ].join("");
}
function refreshLiveRecent() {
  if (!S.sel) return;
  return api.get("/api/violations?limit=6&cam=" + encodeURIComponent(S.sel))
    .then(rows => renderShots($("#liveRecent"), rows, "No violations on this camera yet.", true)).catch(() => {});
}
$("#liveRecentAll").onclick = () => { S.ev.cam = S.sel; S.ev.rule = ""; go("evid"); };
$("#fullBtn").onclick = () => {
  const d = document, st = $("#liveStage");
  if (d.fullscreenElement || d.webkitFullscreenElement) (d.exitFullscreen || d.webkitExitFullscreen).call(d);
  else if (st.requestFullscreen) st.requestFullscreen();
  else if (st.webkitRequestFullscreen) st.webkitRequestFullscreen();
};
$("#snapBtn").onclick = () => {
  const img = $("#liveImg");
  if (!img.naturalWidth) { toast("There is no picture yet.", "warning"); return; }
  const cv = document.createElement("canvas");
  cv.width = img.naturalWidth;
  cv.height = img.naturalHeight;
  cv.getContext("2d").drawImage(img, 0, 0);
  const a = document.createElement("a");
  a.download = `poketenashi_${S.sel}_${stamp()}.jpg`;
  a.href = cv.toDataURL("image/jpeg", 0.92);
  a.click();
  toast("Snapshot saved to your downloads.", "success");
};
$("#ovBtn").onclick = admin(() => {
  api.get("/api/settings").then(s => {
    const cur = s.overlays;
    modal({
      title: "Live picture overlays", subtitle: "What is drawn on the live picture - for every viewer, the TV included",
      icon: "layers", size: "md",
      body: `<div class="switch-list">${Object.keys(OVERLAYS).map(k => sw("ov_" + k, OVERLAYS[k][0], OVERLAYS[k][1], cur[k])).join("")}</div>`,
      actions: [{label: "Cancel", onClick: m => m.close()},
        {label: "Apply…", kind: "primary", icon: "check", onClick: m => {
          const next = {};
          Object.keys(OVERLAYS).forEach(k => { next[k] = $("#ov_" + k, m.el).checked; });
          confirmOverlays(cur, next).then(ok => { if (ok) m.close(true); });
        }}]
    });
  }).catch(err => toast(err.message, "error"));
}, "Log in as an administrator to change the live picture.");
function confirmOverlays(cur, next) {
  const changed = Object.keys(OVERLAYS).filter(k => !!next[k] !== !!cur[k]);
  if (!changed.length) { toast("Nothing changed.", "info"); return Promise.resolve(true); }
  const body = {};
  changed.forEach(k => { body[k] = next[k]; });
  return confirm({
    title: "Change the live picture?", icon: "layers",
    message: "Every viewer sees this, the TV included. It is saved in config.yaml and kept after a restart.",
    html: kv(changed.map(k => [OVERLAYS[k][0], cur[k] ? "On" : "Off", next[k] ? "On" : "Off"])),
    confirm: "Apply changes", confirmIcon: "check", busy: "Applying…",
    action: () => api.post("/api/settings", {overlays: body})
  }).then(r => { if (r) { toast("The live picture is updated.", "success"); refreshSys(); } return !!r; });
}

/* ======================================================================
   Evidence
   ====================================================================== */
function renderEvidenceFilters() {
  $("#evCam").innerHTML = `<option value="">All cameras</option>` + S.cams.map(c =>
    `<option value="${esc(c.id)}"${c.id === S.ev.cam ? " selected" : ""}>${esc(c.name)}</option>`).join("");
  $("#evCam").classList.toggle("hide", S.cams.length < 2 && !S.ev.cam);
  $("#evRules").innerHTML = [["", "All rules"]].concat(RULES.map(r => [r, r])).map(([r, l]) => {
    const on = S.ev.rule === r, c = r ? ruleColor(r) : "var(--accent)";
    return `<button class="chip${on ? " on" : ""}" data-rule="${r}" title="${esc(r ? RULE_NAMES[r] : "Every rule")}"` +
      (on ? ` style="background:${c};border-color:${c}"` : "") + `>${l}</button>`;
  }).join("");
}
function renderEvidence(reset) {
  renderEvidenceFilters();
  if (reset) $("#evList").__sig = null;
  loadEvidence();
}
$("#evCam").onchange = () => { S.ev.cam = $("#evCam").value; S.ev.limit = 60; renderEvidence(true); };
$("#evRules").onclick = e => {
  const b = e.target.closest("[data-rule]");
  if (!b) return;
  S.ev.rule = b.getAttribute("data-rule");
  S.ev.limit = 60;
  renderEvidence(true);
};
$("#evMore").onclick = () => { S.ev.limit = Math.min(1000, S.ev.limit + 60); loadEvidence(); };
function loadEvidence() {
  const q = `/api/violations?limit=${S.ev.limit}` + (S.ev.rule ? "&rule=" + S.ev.rule : "") +
            (S.ev.cam ? "&cam=" + encodeURIComponent(S.ev.cam) : "");
  const el = $("#evList");
  if (el.__sig == null && !el.innerHTML) el.innerHTML = `<div class="shots">${'<div class="skel" style="aspect-ratio:16/11"></div>'.repeat(8)}</div>`;
  return api.get(q).then(rows => {
    $("#evCount").textContent = fmt.plural(rows.length, "event") + (rows.length >= S.ev.limit ? " (newest)" : "");
    $("#evMore").classList.toggle("hide", rows.length < S.ev.limit || S.ev.limit >= 1000);
    const sig = rows.map(v => v.time + v.track_id + v.rule + (v.clip || "")).join("|");
    if (el.__sig === sig) return;
    el.__sig = sig;
    el.__rows = rows;
    el.setAttribute("data-rows", "");
    if (!rows.length) { el.innerHTML = `<div class="card">${emptyBox("evidence", "No violations match these filters.")}</div>`; return; }
    const today = fmt.today(), groups = [];
    rows.forEach((v, i) => {
      const d = String(v.time).slice(0, 10), g = groups[groups.length - 1];
      if (g && g.d === d) g.items.push([v, i]); else groups.push({d, items: [[v, i]]});
    });
    const many = S.cams.length > 1 && !S.ev.cam;
    el.innerHTML = groups.map(g => `<div class="day">${g.d === today ? "Today" : esc(g.d)} <span class="muted">(${g.items.length})</span></div>` +
      `<div class="shots">${g.items.map(([v, i]) => shotHtml(v, i, many)).join("")}</div>`).join("");
  }).catch(err => toast(err.message, "error"));
}

/* ======================================================================
   Cameras
   ====================================================================== */
function renderCams() {
  const on = S.cams.filter(c => c.enabled), up = on.filter(c => c.camera_ok);
  $("#camSummary").innerHTML = S.cams.length
    ? `<b>${fmt.plural(S.cams.length, "camera")}</b> · ${up.length} of ${on.length} switched-on online · up to 8 cameras`
    : "No cameras yet.";
  const list = $("#camList");
  list.innerHTML = S.cams.length ? S.cams.map(camCard).join("")
    : `<div class="card">${emptyBox("camera", "No cameras yet - press “Add camera”.")}</div>`;
  updateCamBits();
  refreshSnaps();
}
function camCard(c) {
  const id = esc(c.id), gangway = (c.zones || []).indexOf("gangway") >= 0;
  return `<div class="card cam-card" data-camcard="${id}">
    <div class="stage">${c.enabled ? `<img data-snap="${id}" alt="">` : ""}
      <div class="ph" data-ph="${id}">${icon(c.enabled ? "live" : "power")}<span>${c.enabled ? "Waiting for the picture…" : "Switched off"}</span></div></div>
    <div class="cam-info">
      <div class="cam-title"><h3>${esc(c.name)}</h3><span class="tag">${id}</span>
        <span data-cam-status="${id}">${statusPill(c)}</span>
        <span class="pill ${c.detect ? "info" : ""}">${icon(c.detect ? "shield" : "eye", "sm")}${c.detect ? "Rules on" : "Live view only"}</span></div>
      <div class="cam-addr" title="${esc(c.source)}">${esc(c.source)}</div>
      ${c.detect ? `<div class="muted" style="font-size:12.5px">Checks: ${(c.rules_enabled || RULES).map(r => badge(r)).join(" ") || "no rule"}</div>` : ""}
      <div class="cam-facts" data-cam-facts="${id}"></div>
      <div data-cam-err="${id}"></div>
      <div data-cam-warn="${id}"></div>
      ${c.detect && !gangway ? `<div class="note warn">${icon("alert")}<div><b>No gangway zone drawn.</b> PO, KE, RUN and FALL judge everyone in the picture until it is. <a href="#zones" data-act="zones" data-id="${id}">Draw the zones</a></div></div>` : ""}
      <div class="cam-actions">
        <button class="btn btn-secondary btn-sm" data-act="live" data-id="${id}">${icon("live")}<span>Live view</span></button>
        <button class="btn btn-secondary btn-sm" data-act="zones" data-id="${id}">${icon("zones")}<span>Zones</span></button>
        <button class="btn btn-secondary btn-sm" data-act="edit" data-id="${id}">${icon("edit")}<span>Edit</span></button>
        <button class="btn btn-secondary btn-sm" data-act="rules" data-id="${id}">${icon("shield")}<span>Rules…</span></button>
        <button class="btn btn-secondary btn-sm" data-act="power" data-id="${id}">${icon("power")}<span>${c.enabled ? "Switch off" : "Switch on"}</span></button>
        <button class="btn btn-danger-ghost btn-sm" data-act="remove" data-id="${id}"${S.cams.length <= 1 ? ' disabled title="The last camera cannot be removed"' : ""}>${icon("trash")}<span>Remove</span></button>
      </div></div></div>`;
}
$("#camList").onclick = e => {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  e.preventDefault();
  const c = camById(b.getAttribute("data-id"));
  if (!c) return;
  const a = b.getAttribute("data-act");
  if (a === "live") { S.sel = c.id; store.set("pk_cam", c.id); S.grid = false; go("live"); }
  if (a === "zones") { ZE.want = c.id; go("zones"); }
  if (a === "edit") editCam(c);
  if (a === "rules") camRules(c);
  if (a === "power") powerCam(c);
  if (a === "remove") removeCam(c);
};
$("#addCamBtn").onclick = admin(() => cameraForm(null), "Log in as an administrator to add a camera.");
const editCam = admin(c => cameraForm(c), "Log in as an administrator to change a camera.");
const powerCam = admin(c => {
  const on = !c.enabled;
  confirm({
    title: on ? `Switch ${c.name} on?` : `Switch ${c.name} off?`, icon: "power", danger: !on,
    message: on ? "The monitor connects to the camera again and runs its rules."
                : "Its live picture and its rules stop until it is switched on again. Nothing is deleted.",
    confirm: on ? "Switch on" : "Switch off", confirmIcon: "power", busy: "Saving…",
    action: () => api.post("/api/cameras/update", {id: c.id, enabled: on})
  }).then(r => { if (r) { toast(`${c.name} is switched ${on ? "on" : "off"}.`, "success"); afterCamChange(); } });
}, "Log in as an administrator to switch cameras.");
const removeCam = admin(c => {
  confirm({
    title: `Remove ${c.name}?`, danger: true, icon: "trash",
    message: "The camera stops being monitored and leaves the list. Its saved evidence stays. This cannot be undone.",
    html: kv([["Camera", `${c.name} (${c.id})`], ["Address", c.source]]),
    confirm: "Remove camera", confirmIcon: "trash", busy: "Removing…",
    action: () => api.post("/api/cameras/delete", {id: c.id})
  }).then(r => { if (r) { toast(`${c.name} was removed.`, "success"); afterCamChange(); } });
}, "Log in as an administrator to remove a camera.");
function afterCamChange() {
  pollState();
  setTimeout(pollState, 2500);
  setTimeout(pollState, 7000);
}

// The add / edit form. A camera can be given by IP address (+ login): the
// monitor then asks the camera itself which stream it serves - nobody needs to
// know RTSP addresses. Or by full address (RTSP URL, video file, webcam number).
function cameraForm(cam) {
  const edit = !!cam, rtsp = !edit || cam.kind === "rtsp";
  // A camera that answers but serves no stream at its path: open straight on
  // "Find stream" - that is what fixes it.
  const wrongPath = edit && rtsp && cam.enabled && !cam.camera_ok && /answers, but the stream/i.test(cam.last_error || "");
  const st = {mode: edit ? (wrongPath ? "find" : "keep") : "ip", found: null, sel: null, tested: null};
  const modes = edit
    ? [["keep", "Keep current"]].concat(rtsp ? [["find", "Find stream"]] : [], [["ip", "New IP / login"], ["url", "Full address"]])
    : [["ip", "By IP address"], ["url", "By full address"]];
  const m = modal({
    title: edit ? `Edit ${cam.name}` : "Add a camera",
    subtitle: edit ? `${cam.id} · ${cam.kind === "rtsp" ? "network camera" : cam.kind}` : "Connect a network camera with its IP address and login",
    icon: edit ? "edit" : "camera", size: "lg",
    body: `<div class="field"><label for="cfName">Camera name</label>
        <input class="input" id="cfName" maxlength="40" placeholder="e.g. Aisle 2 - north end" value="${esc(edit ? cam.name : "")}"></div>
      <div class="field"><span class="label">Connection</span>
        ${edit ? `<div class="addr">${icon("link")}<code title="${esc(cam.source)}">${esc(cam.source)}</code></div>` : ""}
        <div><div class="seg" id="cfMode">${modes.map(([k, l]) => `<button type="button" data-m="${k}" class="${k === st.mode ? "on" : ""}">${l}</button>`).join("")}</div></div>
      </div>
      <div id="cfPane" style="display:flex;flex-direction:column;gap:14px"></div>
      <div class="divider"></div>
      <div class="switch-list">
        ${sw("cfDetect", "Run the safety rules on this camera", "Off = live picture only. Cameras with rules on share this PC's processing - two get about half the frames each.", edit ? cam.detect : true)}
        ${edit ? sw("cfOn", "Camera switched on", "Off = no picture and no rules until it is switched on again.", cam.enabled) : ""}
      </div>`,
    actions: [{label: "Cancel", onClick: mm => mm.close()},
              {label: edit ? "Save changes…" : "Add camera…", kind: "primary", icon: edit ? "save" : "plus", onClick: mm => submit(mm)}]
  });
  const el = s => $(s, m.el);
  const val = s => { const x = el(s); return x ? x.value.trim() : ""; };

  function renderPane() {
    const p = el("#cfPane");
    if (st.mode === "keep") {
      p.innerHTML = `<p class="help">The camera keeps its current address and login.</p>`;
    } else if (st.mode === "find") {
      p.innerHTML = `<div class="note info">${icon("info")}<div>The monitor asks the camera which stream it serves, with the camera's <b>saved</b> login. Use this when the camera answers but shows no picture - its stream path does not match its brand.</div></div>
        <div><button type="button" class="btn btn-secondary" id="cfFind">${icon("search")}<span>Find stream</span></button></div>
        <div id="cfStreams"></div>`;
    } else if (st.mode === "ip") {
      p.innerHTML = `<div class="form-grid">
          <div class="field"><label for="cfHost">IP address</label><input class="input" id="cfHost" placeholder="192.168.1.64" value="${esc(edit && cam.host ? cam.host : "")}" autocomplete="off"></div>
          <div class="field"><label for="cfPort">RTSP port</label><input class="input num" id="cfPort" type="number" min="1" max="65535" value="${edit && cam.port ? cam.port : 554}"></div>
          <div class="field"><label for="cfUser">User name</label><input class="input" id="cfUser" autocomplete="off" placeholder="${edit ? "(keep the saved login)" : "admin"}"></div>
          <div class="field"><label for="cfPass">Password</label><input class="input" id="cfPass" type="password" autocomplete="new-password" placeholder="${edit ? "(keep the saved login)" : ""}"></div>
        </div>
        ${edit ? `<p class="help">Leave user name and password empty to keep the camera's saved login.</p>` : `<p class="help">The camera's own login (from its label or installer). The monitor finds the stream path for the camera's brand by itself.</p>`}
        <div><button type="button" class="btn btn-secondary" id="cfFind">${icon("search")}<span>Find stream</span></button></div>
        <div id="cfStreams"></div>`;
    } else {
      p.innerHTML = `<div class="field"><label for="cfSrc">Full address</label>
          <div class="input-group"><input class="input mono" id="cfSrc" autocomplete="off" placeholder="rtsp://user:password@192.168.1.64:554/Streaming/Channels/101">
          <button type="button" class="btn btn-secondary" id="cfTest">${icon("check")}<span>Test</span></button></div>
          <span class="help">An RTSP address, a video file path, or a webcam number (0, 1 …).</span></div>
        <div id="cfTestOut"></div>`;
    }
    const f = el("#cfFind");
    if (f) f.onclick = find;
    const t = el("#cfTest");
    if (t) t.onclick = test;
    const src = el("#cfSrc");
    if (src) src.oninput = () => { st.tested = null; };
  }
  el("#cfMode").onclick = e => {
    const b = e.target.closest("button");
    if (!b) return;
    st.mode = b.getAttribute("data-m");
    st.found = null; st.sel = null; st.tested = null;
    $$("#cfMode button", m.el).forEach(x => x.classList.toggle("on", x === b));
    m.error("");
    renderPane();
  };
  renderPane();

  function find() {
    let body;
    if (st.mode === "find") body = {id: cam.id};
    else {
      const host = val("#cfHost");
      if (!host) { m.error("Enter the camera's IP address."); el("#cfHost").focus(); return; }
      body = {host: host, port: val("#cfPort") || 554, user: val("#cfUser"), password: el("#cfPass").value};
      if (edit) body.id = cam.id;
    }
    m.error("");
    const out = el("#cfStreams"), btn = el("#cfFind");
    btn.disabled = true;
    btn.innerHTML = `<span class="spin"></span><span>Asking the camera…</span>`;
    out.innerHTML = `<div class="note">${'<span class="spin"></span>'}<div>Checking the login, then the stream paths of the common camera brands. This takes up to half a minute.</div></div>`;
    st.found = null; st.sel = null;
    api.post("/api/cameras/discover", body).then(d => {
      if (!d.ok) {
        out.innerHTML = `<div class="note ${d.login === "refused" ? "bad" : "warn"}">${icon("alert")}<div>${esc(d.error)}${d.login === "refused" ? "<br><span class='dim'>Check the user name and password - the camera checked them and said no.</span>" : ""}</div></div>`;
        return;
      }
      st.found = d.found;
      out.innerHTML = `<div class="note ok">${icon("check")}<div>The camera accepted the login and serves ${fmt.plural(d.found.length, "stream")}. Pick one:</div></div>
        <div class="streams" role="radiogroup">${d.found.map((f, i) => `<div class="stream-opt" data-i="${i}" role="radio" tabindex="0" aria-checked="false">
          <span class="radio"></span>${f.image ? `<img src="${f.image}" alt="">` : `<div class="stage" style="width:132px">${icon("alert")}</div>`}
          <div><b>${esc(f.label)}${f.width ? ` · ${f.width}×${f.height}` : ""}</b><span class="mono">${esc(f.path)}</span>
          ${f.error ? `<span style="color:#ff9d97">${esc(f.error)}</span>` : ""}</div></div>`).join("")}</div>
        <p class="help">A main stream gives the sharpest picture; a sub stream needs less processing.</p>`;
      const opts = $$(".stream-opt", out);
      const pick = i => {
        st.sel = i;
        opts.forEach((o, j) => { o.classList.toggle("on", j === i); o.setAttribute("aria-checked", j === i ? "true" : "false"); });
      };
      opts.forEach((o, i) => {
        o.onclick = () => pick(i);
        o.onkeydown = e => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); pick(i); } };
      });
      pick(0);
    }).catch(err => {
      out.innerHTML = `<div class="note bad">${icon("alert")}<div>${esc(err.message)}</div></div>`;
    }).then(() => {
      btn.disabled = false;
      btn.innerHTML = `${icon("search")}<span>Find stream again</span>`;
    });
  }
  function test() {
    const src = val("#cfSrc"), out = el("#cfTestOut"), btn = el("#cfTest");
    if (!src) { m.error("Enter the camera's address first."); return; }
    m.error("");
    btn.disabled = true;
    btn.innerHTML = `<span class="spin"></span><span>Testing…</span>`;
    out.innerHTML = "";
    api.post("/api/cameras/test", {source: src}).then(d => {
      if (d.ok) {
        st.tested = d;
        out.innerHTML = `<div class="note ok">${icon("check")}<div>Connected - the camera sends ${d.width}×${d.height}.</div></div><img class="preview" src="${d.image}" alt="Test picture">`;
      } else {
        st.tested = null;
        out.innerHTML = `<div class="note bad">${icon("alert")}<div>${esc(d.error)}</div></div>`;
      }
    }).catch(err => { out.innerHTML = `<div class="note bad">${icon("alert")}<div>${esc(err.message)}</div></div>`; })
      .then(() => { btn.disabled = false; btn.innerHTML = `${icon("check")}<span>Test</span>`; });
  }
  function submit(mm) {
    const name = val("#cfName");
    const detect = el("#cfDetect").checked, enabled = edit ? el("#cfOn").checked : true;
    let conn = null, addr = null, pic = null;
    if (st.mode === "ip" || st.mode === "find") {
      if (!st.found || st.sel == null) { mm.error("Press “Find stream” and pick the camera's stream first."); return; }
      const f = st.found[st.sel];
      pic = f.width ? `${f.width}×${f.height}` : null;
      if (st.mode === "find") {
        conn = {path: f.path};
        addr = cam.source.replace(/^(\w+:\/\/[^/]+).*$/, "$1") + f.path;
      } else {
        const user = val("#cfUser"), host = val("#cfHost"), port = val("#cfPort") || 554;
        conn = {host: host, port: port, user: user, password: el("#cfPass").value, path: f.path};
        addr = `rtsp://${user ? user + ":***@" : (edit && cam.has_login ? "(saved login)@" : "")}${host}:${port}${f.path}`;
      }
    } else if (st.mode === "url") {
      const src = val("#cfSrc");
      if (!src) { mm.error("Enter the camera's address."); return; }
      conn = {source: src};
      addr = maskUrl(src);
      pic = st.tested ? `${st.tested.width}×${st.tested.height}` : "not tested yet";
    }
    mm.error("");
    if (!edit) {
      confirm({
        title: "Add this camera?", icon: "camera", size: "md",
        message: "The monitor starts it within a few seconds.",
        html: kv([["Name", name || "(automatic)"], ["Address", addr], ["Picture", pic || "—"],
                  ["Safety rules", detect ? "On" : "Off - live view only"]]) +
          (detect ? `<div class="note">${icon("info")}<div>Cameras with rules on share this PC's processing. After adding it, draw its <b>gangway</b> zone in the Zone editor.</div></div>` : ""),
        confirm: "Add camera", confirmIcon: "plus", busy: "Adding…",
        action: () => api.post("/api/cameras/add", Object.assign({name: name, detect: detect}, conn))
      }).then(r => {
        if (!r) return;
        mm.close(true);
        toast(`${name || "The camera"} was added - it starts in a few seconds.`, "success", "Camera added");
        afterCamChange();
      });
      return;
    }
    const rows = [], payload = {id: cam.id};
    if (name && name !== cam.name) { rows.push(["Name", cam.name, name]); payload.name = name; }
    if (conn) { rows.push(["Address", addr]); Object.assign(payload, conn); }
    if (detect !== cam.detect) { rows.push(["Safety rules", cam.detect ? "On" : "Off", detect ? "On" : "Off"]); payload.detect = detect; }
    if (enabled !== cam.enabled) { rows.push(["Camera", cam.enabled ? "On" : "Off", enabled ? "On" : "Off"]); payload.enabled = enabled; }
    if (!rows.length) { toast("Nothing changed.", "info"); return; }
    const reconnects = conn || "detect" in payload || "enabled" in payload;
    confirm({
      title: `Save the changes to ${cam.name}?`, icon: "save", size: "md",
      html: kv(rows) + (reconnects ? `<div class="note">${icon("info")}<div>The camera reconnects: its picture is gone for a few seconds.</div></div>` : ""),
      confirm: "Save changes", confirmIcon: "save", busy: "Saving…",
      action: () => api.post("/api/cameras/update", payload)
    }).then(r => {
      if (!r) return;
      mm.close(true);
      toast(`${name || cam.name} is saved.`, "success", "Camera updated");
      afterCamChange();
    });
  }
}

/* ======================================================================
   Zone editor
   ====================================================================== */
const STANDARD = ["gangway", "crossing", "stairs", "handrail", "checkpoint", "restricted"];
const ZCOL = {gangway: "#22d3ee", crossing: "#f59e0b", stairs: "#a78bfa", handrail: "#f472b6",
              checkpoint: "#34d399", restricted: "#ef4444", walkway_center: "#94a3b8"};
const ZEXTRA = ["#60a5fa", "#fb923c", "#4ade80", "#e879f9", "#facc15", "#2dd4bf"];
const clone = o => JSON.parse(JSON.stringify(o));
const ZE = {
  built: false, loaded: false, cam: "", camName: "", want: "", W: 1920, H: 1080,
  zones: {}, base: {}, help: {}, dirs: {}, baseDirs: {}, active: null, img: null, imgAt: "",
  drag: null, selV: -1, hoverV: -1, undo: [], redo: [], hidden: {}, canSave: false,

  build() {
    $("#p-zones").innerHTML = `
      <div class="card"><div class="card-body toolbar">
        <select class="select" id="zeCam" style="width:auto;min-width:210px" aria-label="Camera"></select>
        <span id="zeState"></span>
        <span class="sp"></span>
        <button class="btn btn-secondary" id="zeDiscard">${icon("undo")}<span>Discard changes</span></button>
        <button class="btn btn-primary" id="zeSave">${icon("save")}<span>Save zones…</span></button>
      </div></div>
      <div class="note hide" id="zeRo">${icon("lock")}<div>View only - <b>log in as an administrator</b> to change zones. <a href="#" id="zeLogin">Log in</a></div></div>
      <div class="ze">
        <div class="card"><div class="card-body">
          <div class="ze-canvas"><canvas id="zeCv" width="1920" height="1080" aria-label="Camera picture with the zones"></canvas></div>
          <div class="ze-under"><span id="zePic">Loading the camera picture…</span><span id="zeXY" class="num"></span>
            <span style="flex:1"></span><button class="btn btn-ghost btn-sm" id="zeRefresh">${icon("refresh")}<span>New camera picture</span></button></div>
        </div></div>
        <div class="ze-side">
          <div class="card"><div class="card-head"><h2>Zones</h2><span class="sub" id="zeCount"></span></div>
            <div class="card-body"><div class="zlist" id="zeList"></div>
              <div class="input-group" style="margin-top:12px"><input class="input" id="zeNew" maxlength="32" placeholder="New zone name, e.g. loading_bay">
                <button class="btn btn-secondary" id="zeAdd">${icon("plus")}<span>Add</span></button></div></div></div>
          <div class="card"><div class="card-head"><h2 id="zeSelT">Selected zone</h2></div>
            <div class="card-body" style="display:flex;flex-direction:column;gap:12px">
              <p class="help" id="zeHelp"></p><div id="zeWarn"></div>
              <div id="zeDir" class="hide"><span class="label">Feet on the STOP-LOOK-GO poster point</span>
                <div class="seg" id="zeDirSeg" style="margin-top:6px">
                  <button data-d="0,-1" title="Up">↑ Up</button><button data-d="0,1" title="Down">↓ Down</button>
                  <button data-d="-1,0" title="Left">← Left</button><button data-d="1,0" title="Right">→ Right</button></div>
                <p class="help" style="margin-top:6px">SHI is judged only on people leaving the poster this way without a stop. Set it as the painted feet point on the picture.</p></div>
              <div class="toolbar">
                <button class="btn btn-secondary btn-sm" id="zeUndo" title="Ctrl+Z">${icon("undo")}<span>Undo</span></button>
                <button class="btn btn-secondary btn-sm" id="zeRedo" title="Ctrl+Y">${icon("redo")}<span>Redo</span></button>
                <button class="btn btn-danger-ghost btn-sm" id="zeClear">${icon("eraser")}<span>Clear zone</span></button>
              </div></div></div>
          <div class="card"><div class="card-head"><h2>How to draw</h2></div><div class="card-body"><ul class="keys">
            <li><span class="kbd">Click</span>&nbsp; add a corner</li>
            <li><span class="kbd">Drag</span>&nbsp; move a corner</li>
            <li><span class="kbd">Click an edge</span>&nbsp; insert a corner there</li>
            <li><span class="kbd">Right-click</span> or <span class="kbd">Del</span>&nbsp; delete a corner</li>
            <li><span class="kbd">Ctrl+Z</span> / <span class="kbd">Ctrl+Y</span>&nbsp; undo / redo</li>
          </ul></div></div>
        </div>
      </div>`;
    this.cv = $("#zeCv");
    this.ctx = this.cv.getContext("2d");
    $("#zeCam").onchange = () => {
      const want = $("#zeCam").value;
      if (want === this.cam) return;
      const go2 = () => this.load(want);
      if (this.dirty()) this.askDiscard().then(ok => { if (ok) go2(); else $("#zeCam").value = this.cam; });
      else go2();
    };
    $("#zeSave").onclick = () => this.save();
    $("#zeDiscard").onclick = () => {
      if (!this.dirty()) return;
      this.askDiscard().then(ok => { if (ok) { this.revert(); toast("The changes were discarded.", "info"); } });
    };
    $("#zeLogin").onclick = e => { e.preventDefault(); login("Log in as an administrator to change zones."); };
    $("#zeRefresh").onclick = () => this.loadPicture();
    $("#zeUndo").onclick = () => this.doUndo();
    $("#zeDirSeg").onclick = e => {
      const b = e.target.closest("[data-d]");
      if (!b || !this.editable() || !this.isPad(this.active)) return;
      const d = b.getAttribute("data-d").split(",").map(Number);
      const cur = this.dirs[this.active];
      this.dirs[this.active] = cur && cur[0] === d[0] && cur[1] === d[1] ? null : d;
      this.renderAll();
    };
    $("#zeRedo").onclick = () => this.doRedo();
    $("#zeClear").onclick = () => {
      if (!this.editable() || !this.active || !(this.zones[this.active] || []).length) return;
      this.snap(); this.zones[this.active] = []; this.selV = -1; this.renderAll();
    };
    const add = () => {
      if (!this.editable()) return;
      const name = $("#zeNew").value.trim().toLowerCase().replace(/[^a-z0-9_]+/g, "_");
      if (!/^[a-z_][a-z0-9_]{0,31}$/.test(name)) { toast("A zone name uses lowercase letters, digits and _.", "error"); return; }
      if (!(name in this.zones)) { this.snap(); this.zones[name] = []; }
      this.active = name; $("#zeNew").value = ""; this.selV = -1; this.renderAll();
    };
    $("#zeAdd").onclick = add;
    $("#zeNew").onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); add(); } };
    $("#zeList").onclick = e => {
      const eye = e.target.closest("[data-eye]");
      if (eye) { const n = eye.getAttribute("data-eye"); this.hidden[n] = !this.hidden[n]; this.renderAll(); return; }
      const row = e.target.closest("[data-zone]");
      if (row) { this.active = row.getAttribute("data-zone"); this.selV = -1; this.hidden[this.active] = false; this.renderAll(); }
    };
    this.bindCanvas();
    this.built = true;
  },
  editable() {
    if (this.canSave) return true;
    toast("View only - log in as an administrator to change zones.", "warning");
    return false;
  },
  fillCams() {
    if (!this.built) return;
    $("#zeCam").innerHTML = S.cams.map(c => `<option value="${esc(c.id)}"${c.id === this.cam ? " selected" : ""}>${esc(c.name)}</option>`).join("");
  },
  open() {
    if (!this.built) this.build();
    this.fillCams();
    const want = this.want || this.cam || S.sel || (S.cams[0] && S.cams[0].id) || "";
    this.want = "";
    if (!this.loaded || want !== this.cam) {
      if (this.loaded && this.dirty()) {
        this.askDiscard().then(ok => { if (ok) this.load(want); else this.renderAll(); });
      } else this.load(want);
    } else this.renderAll();
  },
  authChanged() {
    if (!this.loaded) return;
    this.canSave = !!S.me.can_admin;
    this.renderAll();
  },
  load(cam) {
    return api.get("/api/zones" + (cam ? "?cam=" + encodeURIComponent(cam) : "")).then(d => {
      this.cam = d.cam; this.camName = d.name;
      this.W = d.frame[0]; this.H = d.frame[1];
      this.cv.width = this.W; this.cv.height = this.H;
      const z = {};
      Object.keys(d.zones).forEach(k => { z[k] = (d.zones[k] || []).map(p => [p[0], p[1]]); });
      STANDARD.forEach(k => { if (!(k in z)) z[k] = []; });
      this.zones = z; this.base = clone(z); this.help = d.help || {};
      this.dirs = clone(d.go_dir || {}); this.baseDirs = clone(d.go_dir || {});
      this.canSave = !!d.can_save;
      this.undo = []; this.redo = []; this.selV = -1; this.hoverV = -1; this.drag = null;
      if (!this.active || !(this.active in z)) this.active = "gangway";
      this.img = null; this.loaded = true;
      this.fillCams();
      this.renderAll();
      this.loadPicture();
    }).catch(err => toast("Could not load the zones: " + err.message, "error"));
  },
  loadPicture() {
    const cam = this.cam, im = new Image();
    $("#zePic").textContent = "Loading the camera picture…";
    im.onload = () => {
      if (cam !== this.cam) return;
      this.img = im;
      this.imgAt = new Date().toLocaleTimeString();
      $("#zePic").textContent = `Camera picture ${this.W}×${this.H} without overlays · taken ${this.imgAt}`;
      this.draw();
    };
    im.onerror = () => { $("#zePic").textContent = "No camera picture right now - the zones can still be edited."; };
    im.src = "/api/rawframe?cam=" + encodeURIComponent(cam) + "&t=" + Date.now();
  },
  isPad(n) { return !!n && (n === "checkpoint" || n.indexOf("checkpoint_") === 0); },
  dirName(d) { return !d ? "not set" : ({"0,-1": "up", "0,1": "down", "-1,0": "left", "1,0": "right"})[d.join(",")] || "?"; },
  dirChanges() {
    const ks = Object.keys(Object.assign({}, this.dirs, this.baseDirs)).filter(k => this.isPad(k));
    return ks.filter(k => JSON.stringify(this.dirs[k] || null) !== JSON.stringify(this.baseDirs[k] || null));
  },
  dirty() { return this.loaded && (JSON.stringify(this.zones) !== JSON.stringify(this.base) || this.dirChanges().length > 0); },
  changes() {
    const names = Object.keys(Object.assign({}, this.base, this.zones)).sort();
    return names.map(n => {
      const b = this.base[n] || [], a = this.zones[n] || [];
      if (JSON.stringify(a) === JSON.stringify(b)) return null;
      return {name: n, before: b.length, after: a.length, kind: !b.length ? "new" : !a.length ? "off" : "changed"};
    }).filter(Boolean);
  },
  diffHtml(ch) {
    const desc = n => n ? fmt.plural(n, "corner") : "off";
    return `<table class="diff">${ch.map(c => `<tr><td><span class="z"><span class="sw" style="width:11px;height:11px;border-radius:3px;background:${this.color(c.name)}"></span>${esc(c.name)}</span></td>
      <td class="dim">${desc(c.before)}<span class="arrow">${icon("arrow", "sm")}</span><b style="color:var(--text)">${desc(c.after)}</b></td>
      <td><span class="pill ${c.kind === "off" ? "warn" : c.kind === "new" ? "ok" : "info"}">${c.kind === "off" ? "switched off" : c.kind === "new" ? "new" : "changed"}</span></td></tr>`).join("")}` +
      this.dirChanges().map(k => `<tr><td><span class="z"><span class="sw" style="width:11px;height:11px;border-radius:3px;background:${this.color(k)}"></span>${esc(k)} feet</span></td>
      <td class="dim">${this.dirName(this.baseDirs[k])}<span class="arrow">${icon("arrow", "sm")}</span><b style="color:var(--text)">${this.dirName(this.dirs[k])}</b></td><td><span class="pill info">direction</span></td></tr>`).join("") + `</table>`;
  },
  askDiscard() {
    const ch = this.changes();
    return confirm({
      title: "Discard the unsaved zone changes?", danger: true, icon: "alert", size: "md",
      message: `${fmt.plural(ch.length, "zone")} of <b>${esc(this.camName)}</b> ${ch.length === 1 ? "has" : "have"} changes that are not saved.`,
      html: this.diffHtml(ch), confirm: "Discard changes", cancel: "Keep editing"
    });
  },
  revert() {
    this.zones = clone(this.base);
    this.dirs = clone(this.baseDirs);
    this.undo = []; this.redo = []; this.selV = -1;
    this.renderAll();
  },
  save() {
    if (!this.canSave) { login("Log in as an administrator to save zones.", () => this.save()); return; }
    const bad = Object.keys(this.zones).filter(n => this.zones[n].length && this.zones[n].length < 3);
    if (bad.length) {
      this.active = bad[0]; this.renderAll();
      toast(`${bad[0]}: a zone needs at least 3 corners (or none, to switch it off).`, "error");
      return;
    }
    const ch = this.changes(), dch = this.dirChanges();
    if (!ch.length && !dch.length) { toast("There are no changes to save.", "info"); return; }
    const gw = (this.zones.gangway || []).length;
    const warn = gw && gw !== 4 ? `<div class="note warn">${icon("alert")}<div>The gangway has ${gw} corners. NA measures crossing angles on its <b>4</b> corners - with ${gw} it is judged less precisely.</div></div>` : "";
    const offGw = ch.some(c => c.name === "gangway" && c.kind === "off") ? `<div class="note warn">${icon("alert")}<div>Without a gangway, PO, KE, RUN and FALL judge <b>everyone</b> in the picture.</div></div>` : "";
    confirm({
      title: "Save the zone changes?", subtitle: this.camName, icon: "zones", size: "md",
      html: this.diffHtml(ch) + warn + offGw +
        `<div class="note">${icon("info")}<div>The monitor uses the new zones at once. The previous ones are backed up.</div></div>`,
      confirm: "Save zones", confirmIcon: "save", busy: "Saving…",
      action: () => api.post("/api/zones", Object.assign({cam: this.cam, zones: this.zones},
        dch.length ? {go_dir: dch.reduce((o, k) => (o[k] = this.dirs[k] || null, o), {})} : {}))
    }).then(r => {
      if (!r) return;
      this.base = clone(this.zones);
      this.baseDirs = clone(this.dirs);
      this.undo = []; this.redo = [];
      this.renderAll();
      toast(`Zones of ${this.camName} are saved - the monitor is using them now.`, "success", "Zones saved");
      pollState();
    });
  },
  color(name) {
    if (ZCOL[name]) return ZCOL[name];
    const extra = Object.keys(this.zones).filter(n => !ZCOL[n]).sort();
    return ZEXTRA[Math.max(0, extra.indexOf(name)) % ZEXTRA.length];
  },
  snap() {
    this.undo.push(JSON.stringify(this.zones));
    if (this.undo.length > 200) this.undo.shift();
    this.redo = [];
  },
  doUndo() {
    if (!this.undo.length) return;
    this.redo.push(JSON.stringify(this.zones));
    this.zones = JSON.parse(this.undo.pop());
    if (this.active && !(this.active in this.zones)) this.active = "gangway";
    this.selV = -1;
    this.renderAll();
  },
  doRedo() {
    if (!this.redo.length) return;
    this.undo.push(JSON.stringify(this.zones));
    this.zones = JSON.parse(this.redo.pop());
    this.selV = -1;
    this.renderAll();
  },
  renderAll() { this.renderList(); this.renderSel(); this.renderState(); this.draw(); },
  renderState() {
    const n = this.changes().length + this.dirChanges().length, ro = !this.canSave;
    $("#zeState").innerHTML = ro ? `<span class="pill">${icon("lock", "sm")}View only</span>`
      : n ? `<span class="pill warn"><span class="dot warn"></span>${fmt.plural(n, "unsaved change")}</span>`
      : `<span class="pill ok">${icon("check", "sm")}Saved</span>`;
    $("#zeSave").disabled = ro || !n;
    $("#zeDiscard").disabled = !n;
    $("#zeRo").classList.toggle("hide", !ro);
    this.cv.classList.toggle("ro", ro);
    $("#zeUndo").disabled = ro || !this.undo.length;
    $("#zeRedo").disabled = ro || !this.redo.length;
    $("#zeNew").disabled = ro;
    $("#zeAdd").disabled = ro;
  },
  renderList() {
    const names = Object.keys(this.zones).sort((a, b) => {
      const ia = STANDARD.indexOf(a), ib = STANDARD.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    $("#zeList").innerHTML = names.map(n => {
      const k = (this.zones[n] || []).length, mod = JSON.stringify(this.zones[n] || []) !== JSON.stringify(this.base[n] || []);
      const warn = (k > 0 && k < 3) || (n === "gangway" && k && k !== 4);
      return `<div class="zrow${n === this.active ? " on" : ""}" data-zone="${esc(n)}" role="button" tabindex="0">
        <span class="sw" style="background:${this.color(n)}"></span><span class="nm">${esc(n)}</span>
        ${mod ? `<span class="mod" title="Changed - not saved yet"></span>` : ""}
        <span class="pts${warn ? " warn" : ""}">${k ? fmt.plural(k, "corner") : "off"}</span>
        <button class="eye" data-eye="${esc(n)}" title="${this.hidden[n] ? "Show" : "Hide"} on the picture">${icon(this.hidden[n] ? "eye-off" : "eye")}</button></div>`;
    }).join("");
    const drawn = names.filter(n => (this.zones[n] || []).length).length;
    $("#zeCount").textContent = `${drawn} of ${names.length} drawn`;
  },
  renderSel() {
    const a = this.active, k = a ? (this.zones[a] || []).length : 0;
    $("#zeSelT").textContent = a ? "Selected: " + a : "Selected zone";
    $("#zeHelp").textContent = a ? (this.help[a] || "Your own zone. The rules use the zones they know by name; this one is saved but not judged.") : "Pick a zone in the list.";
    let w = "";
    if (a === "gangway" && k && k !== 4) w = "NA measures crossing angles on the gangway's 4 corners: draw exactly 4.";
    else if (this.isPad(a) && k >= 3 && !this.dirs[a]) w = "Set the way the painted feet point - until then SHI judges people leaving the poster in ANY direction.";
    else if (k > 0 && k < 3) w = "A zone needs at least 3 corners - or none, to switch it off.";
    $("#zeWarn").innerHTML = w ? `<div class="note warn">${icon("alert")}<div>${esc(w)}</div></div>` : "";
    $("#zeClear").disabled = !this.canSave || !k;
    $("#zeDir").classList.toggle("hide", !this.isPad(a));
    const cur = (this.dirs[a] || []).join(",");
    $$("#zeDirSeg button").forEach(b => b.classList.toggle("on", b.getAttribute("data-d") === cur));
  },
  draw() {
    const ctx = this.ctx, W = this.W, H = this.H;
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    if (this.img) ctx.drawImage(this.img, 0, 0, W, H);
    else { ctx.fillStyle = "#0b0f15"; ctx.fillRect(0, 0, W, H); }
    const s = W / 1920;
    const order = Object.keys(this.zones).filter(n => n !== this.active).concat(this.active ? [this.active] : []);
    order.forEach(name => {
      const pts = this.zones[name] || [];
      if (!pts.length || this.hidden[name]) return;
      const c = this.color(name), on = name === this.active;
      ctx.beginPath();
      pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      if (pts.length >= 3) { ctx.closePath(); ctx.fillStyle = this.rgba(c, on ? 0.2 : 0.1); ctx.fill(); }
      ctx.lineWidth = (on ? 5 : 3) * s;
      ctx.strokeStyle = c;
      ctx.setLineDash(on ? [] : [14 * s, 8 * s]);
      ctx.stroke();
      ctx.setLineDash([]);
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      ctx.font = `600 ${Math.round(30 * s)}px Segoe UI, system-ui, sans-serif`;
      const tw = ctx.measureText(name).width;
      ctx.fillStyle = "rgba(0,0,0,.6)";
      ctx.fillRect(cx - tw / 2 - 8 * s, cy - 24 * s, tw + 16 * s, 34 * s);
      ctx.fillStyle = c;
      ctx.textAlign = "center";
      ctx.fillText(name, cx, cy + 2 * s);
      ctx.textAlign = "left";
      const dd = this.isPad(name) && this.dirs[name];
      if (dd && pts.length >= 3) {
        const L = 110 * s, x2 = cx + dd[0] * L, y2 = cy + 30 * s + dd[1] * L, x1 = cx, y1 = cy + 30 * s;
        ctx.strokeStyle = "#fde047"; ctx.fillStyle = "#fde047"; ctx.lineWidth = 7 * s;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        const ang = Math.atan2(y2 - y1, x2 - x1), hw = 22 * s;
        ctx.beginPath(); ctx.moveTo(x2 + Math.cos(ang) * hw, y2 + Math.sin(ang) * hw);
        ctx.lineTo(x2 + Math.cos(ang + 2.4) * hw, y2 + Math.sin(ang + 2.4) * hw);
        ctx.lineTo(x2 + Math.cos(ang - 2.4) * hw, y2 + Math.sin(ang - 2.4) * hw); ctx.closePath(); ctx.fill();
      }
      if (on) {
        pts.forEach((p, i) => {
          const hl = i === this.selV || i === this.hoverV;
          ctx.beginPath();
          ctx.arc(p[0], p[1], (hl ? 16 : 12) * s, 0, Math.PI * 2);
          ctx.fillStyle = i === 0 ? "#fff" : c;
          ctx.fill();
          ctx.lineWidth = (i === this.selV ? 4 : 2.5) * s;
          ctx.strokeStyle = i === this.selV ? "#fff" : "#000";
          ctx.stroke();
          ctx.font = `700 ${Math.round(18 * s)}px Segoe UI, system-ui, sans-serif`;
          ctx.fillStyle = "#000";
          ctx.textAlign = "center";
          ctx.fillText(String(i + 1), p[0], p[1] + 6 * s);
          ctx.textAlign = "left";
        });
      }
    });
  },
  rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  },
  toFrame(e) {
    const r = this.cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * this.W / r.width, y = (e.clientY - r.top) * this.H / r.height;
    return [Math.round(Math.min(Math.max(x, 0), this.W)), Math.round(Math.min(Math.max(y, 0), this.H))];
  },
  hitR() { return 14 * this.W / this.cv.getBoundingClientRect().width; },
  nearV(p) {
    const pts = this.zones[this.active] || [], R = this.hitR();
    let best = -1, bd = R;
    pts.forEach((q, i) => { const d = Math.hypot(q[0] - p[0], q[1] - p[1]); if (d <= bd) { bd = d; best = i; } });
    return best;
  },
  nearEdge(p) {
    const pts = this.zones[this.active] || [], R = this.hitR() * 0.8;
    if (pts.length < 2) return -1;
    const n = pts.length >= 3 ? pts.length : pts.length - 1;
    for (let i = 0; i < n; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length], dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
      if (Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]) <= R && t > 0.05 && t < 0.95) return i;
    }
    return -1;
  },
  bindCanvas() {
    const cv = this.cv;
    cv.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      if (!this.editable()) return;
      if (!this.active) { toast("Pick a zone in the list first.", "warning"); return; }
      if (this.hidden[this.active]) { this.hidden[this.active] = false; }
      const p = this.toFrame(e), pts = this.zones[this.active] = this.zones[this.active] || [];
      let i = this.nearV(p);
      this.snap();
      if (i < 0) {
        const edge = this.nearEdge(p);
        if (edge >= 0) { pts.splice(edge + 1, 0, p); i = edge + 1; }
        else { pts.push(p); i = pts.length - 1; }
      }
      this.drag = i; this.selV = i;
      cv.setPointerCapture(e.pointerId);
      this.renderAll();
    });
    cv.addEventListener("pointermove", e => {
      const p = this.toFrame(e);
      $("#zeXY").textContent = `x ${p[0]} · y ${p[1]}`;
      if (this.drag !== null && this.active) {
        this.zones[this.active][this.drag] = p;
        this.draw();
        return;
      }
      const h = this.active ? this.nearV(p) : -1;
      if (h !== this.hoverV) { this.hoverV = h; this.draw(); }
      cv.style.cursor = !this.canSave ? "default" : h >= 0 ? "move" : "crosshair";
    });
    cv.addEventListener("pointerup", () => { if (this.drag !== null) { this.drag = null; this.renderAll(); } });
    cv.addEventListener("pointerleave", () => { $("#zeXY").textContent = ""; if (this.hoverV >= 0) { this.hoverV = -1; this.draw(); } });
    cv.addEventListener("contextmenu", e => {
      e.preventDefault();
      if (!this.canSave || !this.active) return;
      const i = this.nearV(this.toFrame(e));
      if (i >= 0) { this.snap(); this.zones[this.active].splice(i, 1); this.selV = -1; this.renderAll(); }
    });
    document.addEventListener("keydown", e => {
      if (S.page !== "zones" || UI.openPopups() || /^(INPUT|SELECT|TEXTAREA)$/.test((e.target || {}).tagName || "")) return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z" && !e.shiftKey) { e.preventDefault(); this.doUndo(); }
      else if ((e.ctrlKey || e.metaKey) && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); this.doRedo(); }
      else if ((k === "delete" || k === "backspace") && this.canSave && this.active && this.selV >= 0) {
        e.preventDefault();
        this.snap(); this.zones[this.active].splice(this.selV, 1); this.selV = -1; this.renderAll();
      } else if (k === "escape") { this.selV = -1; this.draw(); }
    });
  }
};

/* ======================================================================
   Settings
   ====================================================================== */
function renderSettings() {
  if (!SET_TABS.some(t => t.id === S.sub)) S.sub = "general";
  $("#setNav").innerHTML = SET_TABS.map(t =>
    `<button data-go="set" data-sub="${t.id}" class="${t.id === S.sub ? "on" : ""}" role="tab" aria-selected="${t.id === S.sub}">${icon(t.icon, "sm")}${t.label}</button>`).join("");
  const body = $("#setBody");
  ({general: setGeneral, picture: setPicture, users: setUsers, screens: setScreens, account: setAccount,
    audit: setAudit, system: setSystem})[S.sub](body);
}
function lockedNote(what) {
  if (S.me.can_admin) return "";
  return `<div class="note">${icon("lock")}<div>${S.me.user ? `You are logged in as a viewer. Only an administrator can change ${what}.`
    : `View only - <a href="#" data-login>log in as an administrator</a> to change ${what}.`}</div></div>`;
}
document.addEventListener("click", e => {
  const l = e.target.closest("[data-login]");
  if (l) { e.preventDefault(); login(); }
});

function setGeneral(body) {
  body.innerHTML = lockedNote("the retention") + `<div class="card">
    <div class="card-head"><h2>Evidence retention</h2><span class="sub">How long pictures, clips and log rows are kept</span></div>
    <div class="card-body">
      <div class="form-grid">
        <div class="field"><label for="rDays">Keep evidence for (days)</label><input class="input num" id="rDays" type="number" min="0" max="365" step="1">
          <span class="help">Older evidence is deleted automatically. 0 = keep everything.</span></div>
        <div class="field"><label for="rSweep">Clean up every (minutes)</label><input class="input num" id="rSweep" type="number" min="5" max="1440" step="5">
          <span class="help">How often the monitor looks for old files.</span></div>
        <div class="span-2">${sw("rTrim", "Also remove old rows from the violation log", "Keeps the log in step with the evidence that still exists.", true)}</div>
      </div>
      <div class="stat-list" id="rStats" style="margin-top:18px"></div>
      <div class="field" style="margin-top:14px"><span class="label">Evidence folder</span>
        <div class="addr">${icon("hdd")}<code id="rDir">…</code></div></div>
    </div>
    <div class="card-foot"><span class="muted" id="rDirty" style="margin-right:auto"></span>
      <button class="btn btn-secondary" id="rReset" disabled>Reset</button>
      <button class="btn btn-primary" id="rSave" disabled>${icon("save")}<span>Save changes…</span></button></div></div>`;
  let cur = null;
  const read = () => ({days: Number($("#rDays").value), sweep_min: Number($("#rSweep").value), trim_log: $("#rTrim").checked});
  const fill = r => { $("#rDays").value = +r.days; $("#rSweep").value = +r.sweep_min; $("#rTrim").checked = !!r.trim_log; check(); };
  const check = () => {
    if (!cur) return;
    const v = read(), dirty = v.days !== +cur.days || v.sweep_min !== +cur.sweep_min || v.trim_log !== !!cur.trim_log;
    $("#rSave").disabled = !dirty || !S.me.can_admin;
    $("#rReset").disabled = !dirty;
    $("#rDirty").textContent = dirty ? "Unsaved changes" : "";
  };
  ["#rDays", "#rSweep", "#rTrim"].forEach(s => { $(s).oninput = check; $(s).onchange = check; $(s).disabled = !S.me.can_admin; });
  $("#rReset").onclick = () => fill(cur);
  api.get("/api/settings").then(s => {
    S.sys = s; cur = s.retention; fill(cur);
    $("#rStats").innerHTML = [
      ["Evidence stored", fmt.mb(s.evidence.mb), fmt.plural(s.evidence.files, "file")],
      ["Disk free", s.disk ? s.disk.free_gb + " GB" : "—", s.disk ? "of " + s.disk.total_gb + " GB" : ""],
      ["Kept now", +s.retention.days ? fmt.plural(+s.retention.days, "day") : "ever", +s.retention.days ? "older evidence is deleted" : "nothing is deleted"],
    ].map(([l, v, sub]) => `<div class="stat"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="help">${esc(sub)}</div></div>`).join("");
    $("#rDir").textContent = s.evidence.dir;
    $("#rDir").title = s.evidence.dir;
  }).catch(err => toast(err.message, "error"));
  $("#rSave").onclick = admin(() => {
    const v = read();
    if (!(v.days >= 0 && v.days <= 365)) { toast("Keep evidence for 0 to 365 days.", "error"); return; }
    if (!(v.sweep_min >= 5 && v.sweep_min <= 1440)) { toast("Clean up every 5 to 1440 minutes.", "error"); return; }
    api.get("/api/settings/preview?days=" + v.days).then(p => {
      const rows = [];
      if (v.days !== +cur.days) rows.push(["Keep evidence for", +cur.days ? fmt.plural(+cur.days, "day") : "ever", v.days ? fmt.plural(v.days, "day") : "ever"]);
      if (v.sweep_min !== +cur.sweep_min) rows.push(["Clean up every", cur.sweep_min + " min", v.sweep_min + " min"]);
      if (v.trim_log !== !!cur.trim_log) rows.push(["Trim the log", cur.trim_log ? "On" : "Off", v.trim_log ? "On" : "Off"]);
      const loses = p.files || (v.trim_log && p.log_rows);
      return confirm({
        title: "Save the retention?", icon: loses ? "alert" : "database", danger: !!loses, size: "md",
        html: kv(rows) + (loses
          ? `<div class="note bad">${icon("alert")}<div>Saving deletes <b>${fmt.plural(p.files, "file")}</b> (${fmt.mb(p.mb)})${v.trim_log && p.log_rows ? ` and <b>${fmt.plural(p.log_rows, "log row")}</b>` : ""} right away - evidence older than ${fmt.plural(v.days, "day")}. This cannot be undone.</div></div>`
          : `<div class="note ok">${icon("check")}<div>Nothing is deleted right now.</div></div>`),
        confirm: loses ? "Save and delete old evidence" : "Save", confirmIcon: "save", busy: "Saving…",
        action: () => api.post("/api/settings", {retention: v})
      });
    }).then(r => {
      if (!r) return;
      cur = r.retention || v; fill(cur);
      toast("The retention is saved.", "success");
      refreshSys();
    }).catch(err => toast(err.message, "error"));
  }, "Log in as an administrator to change the retention.");
}

function setPicture(body) {
  body.innerHTML = lockedNote("the live picture") + `<div class="card">
    <div class="card-head"><h2>Live picture</h2><span class="sub">What is drawn on the live picture - for every viewer, the TV included</span></div>
    <div class="card-body"><div class="switch-list" id="ovList"><div class="skel" style="height:120px"></div></div></div>
    <div class="card-foot"><span class="muted" id="ovDirty" style="margin-right:auto"></span>
      <button class="btn btn-secondary" id="ovReset" disabled>Reset</button>
      <button class="btn btn-primary" id="ovSave" disabled>${icon("save")}<span>Save changes…</span></button></div></div>`;
  let cur = null;
  const read = () => { const o = {}; Object.keys(OVERLAYS).forEach(k => { o[k] = $("#sp_" + k).checked; }); return o; };
  const check = () => {
    const n = Object.keys(OVERLAYS).filter(k => !!read()[k] !== !!cur[k]).length;
    $("#ovSave").disabled = !n || !S.me.can_admin;
    $("#ovReset").disabled = !n;
    $("#ovDirty").textContent = n ? fmt.plural(n, "unsaved change") : "";
  };
  const fill = o => {
    $("#ovList").innerHTML = Object.keys(OVERLAYS).map(k => sw("sp_" + k, OVERLAYS[k][0], OVERLAYS[k][1], o[k], !S.me.can_admin)).join("");
    $$("#ovList input").forEach(i => { i.onchange = check; });
    check();
  };
  api.get("/api/settings").then(s => { cur = s.overlays; fill(cur); }).catch(err => toast(err.message, "error"));
  $("#ovReset").onclick = () => fill(cur);
  $("#ovSave").onclick = admin(() => {
    const next = read();
    confirmOverlays(cur, next).then(ok => { if (ok) { cur = next; check(); } });
  }, "Log in as an administrator to change the live picture.");
}

function setUsers(body) {
  if (!S.me.can_admin) { body.innerHTML = lockedNote("users") || ""; return; }
  body.innerHTML = `<div class="card">
    <div class="card-head"><h2>Users</h2><span class="sub">Administrators change cameras, zones, settings and users; viewers only look</span>
      <div class="tools"><button class="btn btn-primary btn-sm" id="uAdd">${icon("plus")}<span>Add user</span></button></div></div>
    <div class="card-body"><div class="table-wrap"><table class="table"><thead><tr><th>User</th><th>Role</th><th>Cameras</th><th>Created</th><th></th></tr></thead>
      <tbody id="uRows"><tr><td colspan="5"><div class="skel" style="height:30px"></div></td></tr></tbody></table></div></div></div>
    <div class="note">${icon("info")}<div>Seeing the cameras needs a login: a viewer sees only the cameras given to them, an administrator sees all. A login lasts a week of not being used and survives a restart of the monitor. TV screens use their own link - <a href="#set/screens" data-go="set" data-sub="screens">TV screens</a>.</div></div>`;
  const load = () => api.get("/api/users").then(d => {
    const admins = d.users.filter(u => u.role === "admin").length;
    $("#uRows").innerHTML = d.users.map(u => {
      const me = u.name === S.me.user, lastAdmin = u.role === "admin" && admins <= 1;
      return `<tr><td><div style="display:flex;align-items:center;gap:10px"><span class="avatar">${esc(initials(u.name))}</span><b>${esc(u.name)}</b>${me ? '<span class="tag">you</span>' : ""}</div></td>
        <td>${u.role === "admin" ? `<span class="pill info">${icon("shield", "sm")}Administrator</span>` : `<span class="pill">${icon("eye", "sm")}Viewer</span>`}</td>
        <td>${camsLabel(u.role === "admin" ? ["*"] : u.cams)}</td>
        <td class="muted num nowrap">${esc(u.created || "—")}</td>
        <td class="actions"><div class="row-actions">
          ${u.role === "admin" ? "" : `<button class="btn btn-secondary btn-sm" data-u="cams" data-n="${esc(u.name)}">${icon("camera")}<span>Cameras…</span></button>`}
          <button class="btn btn-secondary btn-sm" data-u="pw" data-n="${esc(u.name)}">${icon("key")}<span>${me ? "Change password" : "Reset password"}</span></button>
          <button class="btn btn-secondary btn-sm" data-u="role" data-n="${esc(u.name)}"${lastAdmin ? ' disabled title="The last administrator stays an administrator"' : ""}>${u.role === "admin" ? "Make viewer" : "Make administrator"}</button>
          <button class="btn btn-danger-ghost btn-sm" data-u="del" data-n="${esc(u.name)}"${lastAdmin ? ' disabled title="The last administrator cannot be removed"' : ""}>${icon("trash")}<span>Remove</span></button></div></td></tr>`;
    }).join("");
    $("#uRows").__users = d.users;
  }).catch(err => toast(err.message, "error"));
  load();
  $("#uAdd").onclick = () => addUser(load);
  $("#uRows").onclick = e => {
    const b = e.target.closest("[data-u]");
    if (!b || b.disabled) return;
    const u = ($("#uRows").__users || []).find(x => x.name === b.getAttribute("data-n"));
    if (!u) return;
    const a = b.getAttribute("data-u");
    if (a === "cams") userCams(u, load);
    if (a === "pw") {
      if (u.name === S.me.user) { go("set", "account"); return; }
      resetPassword(u, load);
    }
    if (a === "role") {
      const to = u.role === "admin" ? "viewer" : "admin";
      confirm({
        title: to === "admin" ? `Make ${u.name} an administrator?` : `Make ${u.name} a viewer?`, icon: "shield",
        message: to === "admin" ? "Administrators can change cameras, zones, settings and users." : "Viewers can only look at the dashboard.",
        html: kv([["Role", u.role === "admin" ? "Administrator" : "Viewer", to === "admin" ? "Administrator" : "Viewer"]]) +
          `<div class="note">${icon("info")}<div>${esc(u.name)} is logged out and logs in again with the new role.</div></div>`,
        confirm: "Change role", busy: "Saving…",
        action: () => api.post("/api/users/update", {user: u.name, role: to})
      }).then(r => { if (r) { toast(`${u.name} is now ${to === "admin" ? "an administrator" : "a viewer"}.`, "success"); load(); if (u.name === S.me.user) loadMe(); } });
    }
    if (a === "del") {
      confirm({
        title: `Remove ${u.name}?`, danger: true, icon: "trash",
        message: "The user can no longer log in. This cannot be undone.",
        html: kv([["User", u.name], ["Role", u.role === "admin" ? "Administrator" : "Viewer"]]),
        confirm: "Remove user", confirmIcon: "trash", busy: "Removing…",
        action: () => api.post("/api/users/delete", {user: u.name})
      }).then(r => { if (r) { toast(`${u.name} was removed.`, "success"); load(); if (u.name === S.me.user) loadMe(); } });
    }
  };
}
function addUser(done) {
  modal({
    title: "Add a user", icon: "users", size: "sm",
    body: `<div class="field"><label for="nuName">User name</label><input class="input" id="nuName" maxlength="32" autocomplete="off" placeholder="e.g. supervisor1">
        <span class="help">2-32 letters, digits, '.', '_' or '-'</span></div>
      <div class="field"><label for="nuRole">Role</label><select class="select" id="nuRole">
        <option value="viewer">Viewer - can only look</option><option value="admin">Administrator - can change everything</option></select></div>
      <div class="field" id="nuCamsF"><span class="label">Cameras this user may see</span>${camChecks("nuc", ["*"])}</div>
      <div class="field"><label for="nuPass">Password</label><input class="input" id="nuPass" type="password" autocomplete="new-password"><span class="help">At least 6 characters</span></div>
      <div class="field"><label for="nuPass2">Password again</label><input class="input" id="nuPass2" type="password" autocomplete="new-password"></div>`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: "Add user…", kind: "primary", icon: "plus", onClick: m => {
        const name = $("#nuName").value.trim(), role = $("#nuRole").value, p1 = $("#nuPass").value;
        const ucams = role === "admin" ? ["*"] : readChecks("nuc");
        if (!ucams.length) { m.error("Pick at least one camera for this viewer."); return; }
        if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{1,31}$/.test(name)) { m.error("User name: 2-32 letters, digits, '.', '_' or '-'."); return; }
        if (p1.length < 6) { m.error("The password needs at least 6 characters."); return; }
        if (p1 !== $("#nuPass2").value) { m.error("The two passwords differ."); return; }
        m.error("");
        confirm({
          title: `Add ${name}?`, icon: "users",
          html: kv([["User name", name], ["Role", role === "admin" ? "Administrator" : "Viewer"], ["Cameras", camsText(ucams)]]) +
            (role === "admin" ? `<div class="note warn">${icon("alert")}<div>An administrator can change cameras, zones, settings and users.</div></div>` : ""),
          confirm: "Add user", confirmIcon: "plus", busy: "Adding…",
          action: () => api.post("/api/users/add", {user: name, password: p1, role: role, cams: ucams})
        }).then(r => { if (r) { m.close(true); toast(`${name} can now log in.`, "success", "User added"); done(); } });
      }}]
  });
}
function resetPassword(u, done) {
  modal({
    title: `New password for ${u.name}`, icon: "key", size: "sm",
    body: `<div class="field"><label for="rpA">New password</label><input class="input" id="rpA" type="password" autocomplete="new-password"><span class="help">At least 6 characters</span></div>
      <div class="field"><label for="rpB">New password again</label><input class="input" id="rpB" type="password" autocomplete="new-password"></div>`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: "Set password…", kind: "primary", icon: "key", onClick: m => {
        const a = $("#rpA").value;
        if (a.length < 6) { m.error("The password needs at least 6 characters."); return; }
        if (a !== $("#rpB").value) { m.error("The two passwords differ."); return; }
        m.error("");
        confirm({
          title: `Reset the password of ${u.name}?`, icon: "key",
          message: `${esc(u.name)} is logged out everywhere and logs in with the new password.`,
          confirm: "Reset password", busy: "Saving…",
          action: () => api.post("/api/users/update", {user: u.name, password: a})
        }).then(r => { if (r) { m.close(true); toast(`The password of ${u.name} is changed.`, "success"); if (done) done(); } });
      }}]
  });
}

// ---- cameras a user / a screen may see -----------------------------------
function camsText(list) {
  if (!list || list.indexOf("*") >= 0) return "All cameras";
  return list.map(camName).join(", ") || "None";
}
function camsLabel(list) {
  const all = !list || list.indexOf("*") >= 0;
  return all ? `<span class="pill info">${icon("camera", "sm")}All cameras</span>`
    : list.map(id => `<span class="pill">${icon("camera", "sm")}${esc(camName(id))}</span>`).join(" ");
}
function camChecks(prefix, sel) {
  const all = !sel || sel.indexOf("*") >= 0;
  return `<div class="switch-list" id="${prefix}List">` + sw(prefix + "_all", "All cameras", "Also every camera added later", all) +
    S.cams.map(c => sw(prefix + "_" + c.id, c.name, c.id + (c.enabled ? "" : " - switched off"), all || sel.indexOf(c.id) >= 0, all)).join("") + `</div>`;
}
function wireChecks(prefix) {
  const a = $("#" + prefix + "_all");
  if (!a) return;
  a.onchange = () => S.cams.forEach(c => { const i = $("#" + prefix + "_" + c.id); i.disabled = a.checked; if (a.checked) i.checked = true; });
}
function readChecks(prefix) {
  if ($("#" + prefix + "_all").checked) return ["*"];
  return S.cams.filter(c => $("#" + prefix + "_" + c.id).checked).map(c => c.id);
}
document.addEventListener("change", e => {
  const id = (e.target && e.target.id) || "";
  if (/_all$/.test(id)) wireChecks(id.slice(0, -4));
  if (id === "nuRole") $("#nuCamsF").classList.toggle("hide", e.target.value === "admin");
});
function userCams(u, done) {
  modal({
    title: `Cameras of ${u.name}`, icon: "camera", size: "sm",
    body: `<p class="help" style="margin:0 0 10px">${esc(u.name)} sees the live picture, the alerts and the evidence of these cameras only.</p>${camChecks("uc", u.cams)}`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: "Save…", kind: "primary", icon: "save", onClick: m => {
        const next = readChecks("uc");
        if (!next.length) { m.error("Pick at least one camera."); return; }
        confirm({
          title: `Change the cameras of ${u.name}?`, icon: "camera",
          html: kv([["Cameras", camsText(u.cams), camsText(next)]]),
          confirm: "Save", confirmIcon: "save", busy: "Saving…",
          action: () => api.post("/api/users/update", {user: u.name, cams: next})
        }).then(r => { if (r) { m.close(true); toast(`${u.name} now sees: ${camsText(next)}.`, "success"); done(); } });
      }}]
  });
  wireChecks("uc");
}

// ---- TV screens: a link per screen that shows its cameras without a login ----
function screenLinks(d) {
  const port = S.me.port || location.port;
  const hosts = (S.me.lan && S.me.lan.length ? S.me.lan : [location.hostname]);
  return hosts.map(h => `http://${h}${port ? ":" + port : ""}/tv?d=${encodeURIComponent(d.token)}`);
}
function setScreens(body) {
  if (!S.me.can_admin) { body.innerHTML = lockedNote("the TV screens") || ""; return; }
  body.innerHTML = `<div class="card">
    <div class="card-head"><h2>TV screens</h2><span class="sub">Each screen gets its own link and shows only its cameras - nobody logs in on the TV</span>
      <div class="tools"><button class="btn btn-primary btn-sm" id="dAdd">${icon("plus")}<span>Add screen</span></button></div></div>
    <div class="card-body" id="dRows"><div class="skel" style="height:60px"></div></div></div>
    <div class="note">${icon("info")}<div>Open a screen's link once in the TV's browser: the TV remembers it (also after a restart of the monitor) and shows only that screen's cameras. <b>New link</b> stops the old link from working - use it if a link got to the wrong place.</div></div>`;
  const load = () => api.get("/api/displays").then(d => {
    const rows = d.displays || [];
    $("#dRows").innerHTML = rows.length ? rows.map(x => `<div class="card" style="margin-bottom:12px"><div class="card-body">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><b style="font-size:15px">${icon("tv", "sm")} ${esc(x.name)}</b>${camsLabel(x.cams)}
        <span class="muted" style="margin-left:auto">${esc(x.created || "")}</span></div>
      <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px">${screenLinks(x).map(l =>
        `<div class="addr">${icon("link")}<code>${esc(l)}</code><button class="btn btn-secondary btn-sm" data-d="copy" data-l="${esc(l)}">${icon("copy")}<span>Copy</span></button></div>`).join("")}</div>
      <div class="row-actions" style="margin-top:10px">
        <button class="btn btn-secondary btn-sm" data-d="cams" data-id="${esc(x.id)}">${icon("camera")}<span>Cameras…</span></button>
        <button class="btn btn-secondary btn-sm" data-d="regen" data-id="${esc(x.id)}">${icon("refresh")}<span>New link…</span></button>
        <button class="btn btn-danger-ghost btn-sm" data-d="del" data-id="${esc(x.id)}">${icon("trash")}<span>Remove</span></button></div>
      </div></div>`).join("") : emptyBox("tv", "No screens yet - press “Add screen”.");
    $("#dRows").__d = rows;
  }).catch(err => toast(err.message, "error"));
  load();
  $("#dAdd").onclick = () => screenForm(null, load);
  $("#dRows").onclick = e => {
    const b = e.target.closest("[data-d]");
    if (!b) return;
    const a = b.getAttribute("data-d");
    if (a === "copy") {
      const l = b.getAttribute("data-l");
      (navigator.clipboard ? navigator.clipboard.writeText(l) : Promise.reject()).then(
        () => toast("Link copied - open it on the TV.", "success"),
        () => { window.prompt("Copy this link and open it on the TV:", l); });
      return;
    }
    const x = ($("#dRows").__d || []).find(y => y.id === b.getAttribute("data-id"));
    if (!x) return;
    if (a === "cams") screenForm(x, load);
    if (a === "regen") confirm({
      title: `New link for ${x.name}?`, icon: "refresh", danger: true,
      message: "The old link stops working at once: the TV showing it goes blank until the new link is opened on it.",
      confirm: "Make a new link", confirmIcon: "refresh", busy: "Saving…",
      action: () => api.post("/api/displays/update", {id: x.id, regenerate: true})
    }).then(r => { if (r) { toast(`${x.name} has a new link - open it on the TV.`, "success"); load(); } });
    if (a === "del") confirm({
      title: `Remove ${x.name}?`, icon: "trash", danger: true,
      message: "Its link stops working; the TV showing it goes blank.",
      html: kv([["Screen", x.name], ["Cameras", camsText(x.cams)]]),
      confirm: "Remove screen", confirmIcon: "trash", busy: "Removing…",
      action: () => api.post("/api/displays/delete", {id: x.id})
    }).then(r => { if (r) { toast(`${x.name} was removed.`, "success"); load(); } });
  };
}
function screenForm(x, done) {
  modal({
    title: x ? `Cameras of ${x.name}` : "Add a TV screen", icon: "tv", size: "sm",
    body: (x ? "" : `<div class="field"><label for="dsName">Screen name</label><input class="input" id="dsName" maxlength="40" placeholder="e.g. Toolroom TV"></div>`) +
      `<div class="field"><span class="label">Cameras on this screen</span>${camChecks("dc", x ? x.cams : [])}</div>`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: x ? "Save…" : "Add screen…", kind: "primary", icon: x ? "save" : "plus", onClick: m => {
        const name = x ? x.name : $("#dsName").value.trim(), cams = readChecks("dc");
        if (!name) { m.error("Give the screen a name."); return; }
        if (!cams.length) { m.error("Pick at least one camera."); return; }
        confirm({
          title: x ? `Change the cameras of ${name}?` : `Add the screen ${name}?`, icon: "tv",
          html: kv(x ? [["Cameras", camsText(x.cams), camsText(cams)]] : [["Screen", name], ["Cameras", camsText(cams)]]),
          confirm: x ? "Save" : "Add screen", confirmIcon: x ? "save" : "plus", busy: "Saving…",
          action: () => x ? api.post("/api/displays/update", {id: x.id, cams: cams}) : api.post("/api/displays/add", {name: name, cams: cams})
        }).then(r => { if (r) { m.close(true); toast(x ? `${name} now shows: ${camsText(cams)}.` : `${name} added - copy its link and open it on the TV.`, "success"); done(); } });
      }}]
  });
  wireChecks("dc");
}

// ---- the rules each camera checks --------------------------------------
const camRules = admin(c => {
  const all = RULES;
  const cur = c.rules_enabled || all;
  const need = {TE: "stairs + handrail", NA: "gangway", SHI: "checkpoint", RESTRICTED: "restricted"};
  modal({
    title: `Rules on ${c.name}`, icon: "shield", size: "sm",
    body: `<p class="help" style="margin:0 0 10px">The rules this camera checks. A rule that needs a zone runs once that zone is drawn.</p><div class="switch-list">` +
      RULES.map(r => sw("cr_" + r, `${r} - ${RULE_NAMES[r] || r}`, need[r] ? "needs the " + need[r] + " zone" : "", cur.indexOf(r) >= 0)).join("") + `</div>`,
    actions: [{label: "Cancel", onClick: m => m.close()},
      {label: "Save…", kind: "primary", icon: "save", onClick: m => {
        const next = RULES.filter(r => $("#cr_" + r).checked);
        const on = r => next.indexOf(r) >= 0, was = r => cur.indexOf(r) >= 0;
        const rows = RULES.filter(r => on(r) !== was(r)).map(r => [r + " - " + (RULE_NAMES[r] || r), was(r) ? "Checked" : "Off", on(r) ? "Checked" : "Off"]);
        if (!rows.length) { m.close(); return; }
        confirm({
          title: `Change the rules on ${c.name}?`, icon: "shield",
          html: kv(rows) + `<div class="note">${icon("info")}<div>The camera restarts its rules (a few seconds without detection).</div></div>`,
          confirm: "Save", confirmIcon: "save", busy: "Saving…",
          action: () => api.post("/api/cameras/update", {id: c.id, rules_enabled: next})
        }).then(r => { if (r) { m.close(true); toast(`${c.name} now checks: ${next.join(", ") || "no rule"}.`, "success"); afterCamChange(); } });
      }}]
  });
}, "Log in as an administrator to change a camera's rules.");

function setAccount(body) {
  if (!S.me.user) {
    body.innerHTML = `<div class="card">${emptyBox("user", "You are not logged in.")}<div class="card-foot" style="justify-content:center;border-top:0;padding-top:0">
      <button class="btn btn-primary" data-login>${icon("login")}<span>Log in</span></button></div></div>`;
    return;
  }
  body.innerHTML = `<div class="grid-2">
    <div class="card"><div class="card-head"><h2>Account</h2></div><div class="card-body">
      <div style="display:flex;align-items:center;gap:14px"><span class="avatar" style="width:46px;height:46px;font-size:16px">${esc(initials(S.me.user))}</span>
        <div><b style="font-size:16px">${esc(S.me.user)}</b><div class="muted">${S.me.role === "admin" ? "Administrator - can change everything" : "Viewer - can look at everything"}</div></div></div>
      <div class="divider" style="margin:16px 0"></div>
      <button class="btn btn-secondary" id="acLogout">${icon("logout")}<span>Log out</span></button></div></div>
    <div class="card"><div class="card-head"><h2>Change my password</h2><span class="sub">Saved on the monitor at once</span></div>
      <form class="card-body" id="pwForm" style="display:flex;flex-direction:column;gap:14px" autocomplete="on">
        <input type="text" autocomplete="username" value="${esc(S.me.user)}" class="hide" tabindex="-1" aria-hidden="true">
        <div class="field"><label for="pwCur">Current password</label><input class="input" id="pwCur" type="password" autocomplete="current-password"></div>
        <div class="field"><label for="pwNew">New password</label><input class="input" id="pwNew" type="password" autocomplete="new-password"><span class="help">At least 6 characters</span></div>
        <div class="field"><label for="pwNew2">New password again</label><input class="input" id="pwNew2" type="password" autocomplete="new-password"></div>
        <div class="field-err" id="pwErr"></div>
        <div><button class="btn btn-primary" type="submit">${icon("key")}<span>Change password…</span></button></div>
      </form></div></div>`;
  $("#acLogout").onclick = logout;
  $("#pwForm").onsubmit = e => {
    e.preventDefault();
    const cur = $("#pwCur").value, a = $("#pwNew").value;
    const err = !cur ? "Enter your current password." : a.length < 6 ? "The new password needs at least 6 characters."
      : a !== $("#pwNew2").value ? "The two new passwords differ." : a === cur ? "The new password is the same as the current one." : "";
    $("#pwErr").textContent = err;
    if (err) return;
    confirm({
      title: "Change your password?", icon: "key",
      message: "Use the new password from now on. Your logins in other browsers are signed out.",
      confirm: "Change password", busy: "Saving…",
      action: () => api.post("/api/password", {current: cur, password: a})
    }).then(r => {
      if (!r) return;
      ["#pwCur", "#pwNew", "#pwNew2"].forEach(s => { $(s).value = ""; });
      toast("Your password is changed.", "success", "Password changed");
    });
  };
}

function setAudit(body) {
  if (!S.me.can_admin) { body.innerHTML = lockedNote("anything - and to see the audit log"); return; }
  body.innerHTML = `<div class="card"><div class="card-head"><h2>Audit log</h2><span class="sub">Who changed what, and when - newest first</span>
      <div class="tools"><input class="input" id="auQ" placeholder="Filter…" style="width:200px;height:30px">
        <button class="btn btn-secondary btn-sm" id="auR">${icon("refresh")}<span>Refresh</span></button></div></div>
    <div class="card-body"><div class="table-wrap"><table class="table"><thead><tr><th>Time</th><th>User</th><th>What</th><th>Details</th><th>From</th></tr></thead>
      <tbody id="auRows"><tr><td colspan="5"><div class="skel" style="height:30px"></div></td></tr></tbody></table></div></div></div>`;
  let all = [];
  const show = () => {
    const q = $("#auQ").value.trim().toLowerCase();
    const rows = q ? all.filter(e => JSON.stringify(e).toLowerCase().indexOf(q) >= 0) : all;
    $("#auRows").innerHTML = rows.length ? rows.map(e => `<tr><td class="num muted" style="white-space:nowrap">${esc(e.time)}</td><td><b>${esc(e.user)}</b></td>
      <td>${/fail/.test(e.action) ? `<span class="pill bad">${esc(e.action)}</span>` : `<span class="tag">${esc(e.action)}</span>`}</td>
      <td>${esc(e.detail)}</td><td class="muted mono">${esc(e.ip)}</td></tr>`).join("")
      : `<tr><td colspan="5">${emptyBox("history", q ? "Nothing matches the filter." : "Nothing recorded yet.")}</td></tr>`;
  };
  const load = () => api.get("/api/audit?limit=400").then(d => { all = d.entries; show(); }).catch(err => toast(err.message, "error"));
  $("#auQ").oninput = show;
  $("#auR").onclick = load;
  load();
}

function setSystem(body) {
  body.innerHTML = `<div class="card"><div class="card-head"><h2>System</h2><span class="sub">The monitor on this PC</span></div>
    <div class="card-body"><div class="stat-list" id="syStats"><div class="skel" style="height:64px"></div></div></div></div>
    <div class="card"><div class="card-head"><h2>Dashboard addresses</h2><span class="sub">Open the dashboard from another PC or phone on the same network</span></div>
      <div class="card-body" id="syAddr" style="display:flex;flex-direction:column;gap:8px"></div></div>`;
  api.get("/api/settings").then(s => {
    S.sys = s;
    const y = s.system;
    $("#syStats").innerHTML = [
      ["Version", y.version, "Poketenashi safety monitor"],
      ["Running since", y.started, "up " + fmt.dur(y.uptime_sec)],
      ["Cameras", String(y.cameras), `${S.cams.filter(c => c.enabled && c.camera_ok).length} online now`],
      ["Users", String(y.users), "can log in"],
      ["Disk free", s.disk ? s.disk.free_gb + " GB" : "—", s.disk ? "of " + s.disk.total_gb + " GB" : ""],
      ["Evidence stored", fmt.mb(s.evidence.mb), fmt.plural(s.evidence.files, "file")],
    ].map(([l, v, sub]) => `<div class="stat"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="help">${esc(sub)}</div></div>`).join("");
    const port = y.port, hosts = y.lan.length ? y.lan : [location.hostname];
    $("#syAddr").innerHTML = hosts.map(h => addrRow(`http://${h}:${port}/`)).join("");
  }).catch(err => toast(err.message, "error"));
}
function addrRow(url) {
  return `<div class="addr">${icon("link")}<code>${esc(url)}</code><button class="btn btn-ghost btn-sm" data-copy="${esc(url)}">${icon("copy")}<span>Copy</span></button>` +
    `<a class="btn btn-secondary btn-sm" href="${esc(url)}" target="_blank" rel="noopener">${icon("external")}<span>Open</span></a></div>`;
}

/* ======================================================================
   TV view
   ====================================================================== */
function renderTv() {
  const port = S.me.port || location.port || 80;
  const hosts = S.me.lan && S.me.lan.length ? S.me.lan : [location.hostname];
  // The per-camera links use one of this PC's addresses - the one on the TV's
  // network (remembered on this computer).
  let host = store.get("pk_tvhost", "");
  if (hosts.indexOf(host) < 0) host = hosts[hosts.length - 1];
  $("#p-tv").innerHTML = `<div class="grid-2">
    <div class="card"><div class="card-head"><h2>Show it on a TV</h2><span class="sub">Live picture with the violations beside it</span></div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:12px">
        <p class="dim">The TV page needs no login, keeps itself up to date and reloads by itself after the monitor restarts. It also works on old TV browsers.
          Pick the camera in its top bar - the TV remembers it - and “‹ Dashboard” leads back here.</p>
        <ol style="margin:0;padding-left:18px;color:var(--text-2);display:flex;flex-direction:column;gap:4px">
          <li>Connect the TV to the same network as this PC.</li>
          <li>Open the TV's web browser and enter one of these addresses:</li></ol>
        ${hosts.map(h => addrRow(`http://${h}:${port}/tv`)).join("")}
        <p class="help">Use the address on the TV's own network - a Wi-Fi TV usually reaches this PC on its Wi-Fi address.</p>
        <div><a class="btn btn-primary" href="/tv" target="_blank" rel="noopener">${icon("external")}<span>Open the TV view here</span></a></div>
      </div></div>
    <div class="card"><div class="card-head"><h2>One camera per screen</h2><span class="sub">Only that camera and only its violations</span>
        ${hosts.length > 1 ? `<div class="tools"><select class="select" id="tvHost" style="width:auto;height:30px" aria-label="This PC's address on the TV's network">
          ${hosts.map(h => `<option${h === host ? " selected" : ""}>${esc(h)}</option>`).join("")}</select></div>` : ""}</div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:8px">
        ${S.cams.length ? S.cams.map(c => {
          const url = `http://${host}:${port}/tv?cam=${encodeURIComponent(c.id)}`;
          return `<div class="addr">${icon("camera")}<div style="flex:1;min-width:0"><b>${esc(c.name)}</b>
            <code style="display:block">${esc(url)}</code></div>
            <button class="btn btn-ghost btn-sm" data-copy="${esc(url)}">${icon("copy")}<span>Copy</span></button>
            <a class="btn btn-secondary btn-sm" href="/tv?cam=${encodeURIComponent(c.id)}" target="_blank" rel="noopener">${icon("external")}<span>Open</span></a></div>`;
        }).join("") : emptyBox("camera", "No cameras yet.")}
      </div></div></div>`;
  const sel = $("#tvHost");
  if (sel) sel.onchange = () => { store.set("pk_tvhost", sel.value); renderTv(); };
}

/* ======================================================================
   Top bar bits and start-up
   ====================================================================== */
function renderSound() {
  const b = $("#soundBtn");
  b.innerHTML = icon(S.sound ? "bell" : "bell-off");
  b.classList.toggle("on", S.sound);
  b.title = S.sound ? "Sound for new violations: on (this computer)" : "Sound for new violations: off";
}
$("#soundBtn").onclick = () => {
  S.sound = !S.sound;
  store.set("pk_sound", S.sound ? "1" : "0");
  renderSound();
  if (S.sound) UI.beep();
  toast(S.sound ? "A chime plays for every new violation - on this computer." : "Sound is off.", "info");
};
function tick() { $("#clock").textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"}); }
window.addEventListener("beforeunload", e => {
  if (ZE.dirty()) { e.preventDefault(); e.returnValue = ""; }
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) { pollState(); refreshStats(); } });

renderSound();
renderUser();
tick();
const first = parseHash();
if (first.sub && SET_TABS.some(t => t.id === first.sub)) S.sub = first.sub;
const qcam = /[?&]cam=([^&#]*)/.exec(location.search);       // /?cam=cam2#zones (old /zones links)
if (qcam) {
  S.sel = ZE.want = decodeURIComponent(qcam[1]);
  history.replaceState(null, "", location.pathname + location.hash);
}
renderDashSys();
go(first.page, first.sub);
loadMe().then(() => { if (S.me.setup_needed && S.me.local) setupAdmin(); });
pollState();
refreshStats();
refreshSys();
pollAlerts();
setInterval(pollState, 2000);
setInterval(pollAlerts, 1500);
setInterval(() => {
  refreshStats();
  if (S.page === "dash") refreshLatest();
  if (S.page === "evid") loadEvidence();
  if (S.page === "live" && !S.grid) refreshLiveRecent();
}, 10000);
setInterval(refreshSys, 30000);
setInterval(refreshSnaps, 1500);
setInterval(tick, 1000);
window.__dashOk = true;
})();
