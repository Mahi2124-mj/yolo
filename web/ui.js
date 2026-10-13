/* Poketenashi UI kit - icons, API calls, popups (modal / confirm), toasts.
   ES2015 on purpose (no async/await, no ?. or ??): index.html sends a browser
   that cannot parse this to the plain /tv page instead. */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return ESC[c]; }); };
  var icon = function (name, cls) {
    return '<svg class="i' + (cls ? " " + cls : "") + '" aria-hidden="true" focusable="false"><use href="#i-' + name + '"></use></svg>';
  };

  /* ------------------------------------------------------------------ API */
  // Resolves with the parsed JSON on success. Rejects with an Error carrying
  // .status and .data otherwise - a 401 that asks for a login also calls
  // api.onLogin (the app opens its login popup).
  var api = {onLogin: null};
  function request(url, opt) {
    return fetch(url, opt).then(function (r) {
      return r.text().then(function (t) {
        var data = null;
        try { data = t ? JSON.parse(t) : null; } catch (e) { data = null; }
        if (r.ok) return data;
        var err = new Error((data && data.error) || (r.status + " " + (r.statusText || "error")));
        err.status = r.status;
        err.data = data;
        if (r.status === 401 && data && data.login && api.onLogin) api.onLogin(err.message);
        throw err;
      });
    }, function () {
      var e = new Error("The monitor is not reachable - check the network.");
      e.status = 0;
      throw e;
    });
  }
  api.get = function (url) { return request(url, {credentials: "same-origin", cache: "no-store"}); };
  api.post = function (url, body) {
    return request(url, {method: "POST", credentials: "same-origin",
                         headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  };

  /* --------------------------------------------------------------- popups */
  var stack = [];

  function modal(o) {
    var prev = document.activeElement;
    var back = document.createElement("div");
    back.className = "modal-back";
    back.style.zIndex = String(300 + stack.length * 10);
    var id = "m" + Math.random().toString(36).slice(2, 9);
    back.innerHTML =
      '<div class="modal' + (o.size ? " modal-" + o.size : "") + (o.cls ? " " + o.cls : "") +
      '" role="dialog" aria-modal="true" aria-labelledby="' + id + '">' +
      '<header class="modal-head">' +
      (o.icon ? '<span class="modal-ico ' + (o.tone || "") + '">' + icon(o.icon) + "</span>" : "") +
      '<div class="modal-titles"><h2 id="' + id + '">' + esc(o.title) + "</h2>" +
      (o.subtitle ? "<p>" + esc(o.subtitle) + "</p>" : "") + "</div>" +
      '<button type="button" class="btn-icon modal-x" aria-label="Close">' + icon("x") + "</button></header>" +
      '<div class="modal-body"></div><div class="modal-error hide" role="alert"></div>' +
      '<footer class="modal-foot"></footer></div>';
    var box = $(".modal", back), body = $(".modal-body", back), foot = $(".modal-foot", back);
    var errBox = $(".modal-error", back);
    if (typeof o.body === "string") body.innerHTML = o.body;
    else if (o.body) body.appendChild(o.body);

    var m = {el: box, body: body, foot: foot, closed: false, locked: false, back: back};
    m.close = function (result) {
      if (m.closed) return;
      m.closed = true;
      var i = stack.indexOf(m);
      if (i >= 0) stack.splice(i, 1);
      back.classList.remove("in");
      back.classList.add("leaving");
      setTimeout(function () { back.remove(); }, 140);
      if (prev && prev.focus && document.body.contains(prev)) { try { prev.focus(); } catch (e) {} }
      if (o.onClose) o.onClose(result);
    };
    m.error = function (msg) {
      errBox.innerHTML = msg ? icon("alert") + "<span>" + esc(msg) + "</span>" : "";
      errBox.classList.toggle("hide", !msg);
    };
    m.busy = function (on, label) {
      m.locked = !!on;
      $$("button", foot).forEach(function (b) { b.disabled = !!on; });
      var main = $(".btn-primary, .btn-danger", foot);
      if (main) {
        if (on) {
          main.__html = main.innerHTML;
          main.innerHTML = '<span class="spin"></span><span>' + esc(label || "Working…") + "</span>";
        } else if (main.__html) {
          main.innerHTML = main.__html;
        }
      }
    };
    m.button = function (key) { return $('[data-key="' + key + '"]', foot); };
    (o.actions || []).forEach(function (a) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "btn btn-" + (a.kind || "secondary");
      b.innerHTML = (a.icon ? icon(a.icon) : "") + "<span>" + esc(a.label) + "</span>";
      if (a.key) b.setAttribute("data-key", a.key);
      if (a.disabled) b.disabled = true;
      b.onclick = function (e) { e.preventDefault(); if (!b.disabled) a.onClick(m, b); };
      foot.appendChild(b);
    });
    if (!o.actions || !o.actions.length) foot.remove();
    var dismiss = function () { if (o.dismissible !== false && !m.locked) m.close(null); };
    $(".modal-x", back).onclick = dismiss;
    back.addEventListener("mousedown", function (e) { if (e.target === back) dismiss(); });
    m.dismiss = dismiss;
    document.body.appendChild(back);
    stack.push(m);
    requestAnimationFrame(function () { back.classList.add("in"); });
    setTimeout(function () {
      var f = $("[autofocus]", box) || $("input:not([type=hidden]):not([disabled]), select, textarea", body) ||
              $(".btn-primary, .btn-danger", foot) || $(".modal-x", back);
      if (f) f.focus();
    }, 40);
    return m;
  }

  // Keyboard for the top popup: Esc closes, Tab stays inside, Enter in a text
  // field presses the main button.
  document.addEventListener("keydown", function (e) {
    var m = stack[stack.length - 1];
    if (!m) return;
    if (e.key === "Escape") { e.preventDefault(); m.dismiss(); return; }
    if (e.key === "Tab") {
      var f = $$('a[href], button:not([disabled]), input:not([disabled]):not([type=hidden]), select:not([disabled]), textarea, [tabindex]:not([tabindex="-1"])', m.el)
        .filter(function (x) { return x.offsetParent !== null; });
      if (!f.length) return;
      var first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      return;
    }
    if (e.key === "Enter" && e.target && e.target.tagName === "INPUT" &&
        !/^(checkbox|radio|button|submit)$/i.test(e.target.type) && m.el.contains(e.target)) {
      var main = $(".btn-primary:not([disabled]), .btn-danger:not([disabled])", m.foot);
      if (main) { e.preventDefault(); main.click(); }
    }
  }, true);

  /* The confirmation every change goes through: what will change, then
     Cancel / Confirm. With `action` the popup stays open while it runs, shows
     a spinner, and shows the error in place if it fails (nothing is lost).
     Resolves with the action's result (or true), or false if cancelled. */
  function confirm(o) {
    return new Promise(function (resolve) {
      var done = false;
      var finish = function (v) { if (!done) { done = true; resolve(v); } };
      var html = (o.message ? '<p class="confirm-msg">' + o.message + "</p>" : "") + (o.html || "");
      modal({
        title: o.title, subtitle: o.subtitle, size: o.size || "sm",
        icon: o.icon || (o.danger ? "alert" : "check"), tone: o.danger ? "danger" : (o.tone || ""),
        body: html,
        onClose: function () { finish(false); },
        actions: [
          {label: o.cancel || "Cancel", kind: "secondary", onClick: function (m) { m.close(false); }},
          {label: o.confirm || "Confirm", kind: o.danger ? "danger" : "primary", icon: o.confirmIcon,
           onClick: function (m) {
             if (!o.action) { finish(true); m.close(true); return; }
             m.error("");
             m.busy(true, o.busy);
             Promise.resolve().then(o.action).then(function (res) {
               m.busy(false);
               finish(res === undefined ? true : res);
               m.close(true);
             }, function (err) {
               m.busy(false);
               m.error((err && err.message) || String(err));
             });
           }}
        ]
      });
    });
  }

  // A key/value summary table for confirmations: rows of [label, value] or
  // [label, before, after] (shown as "before -> after").
  function kv(rows) {
    return '<table class="kv">' + rows.map(function (r) {
      var v = r.length > 2
        ? '<span class="from">' + esc(r[1]) + "</span>" + icon("arrow", "sm") + " <b>" + esc(r[2]) + "</b>"
        : (r[1] && r[1].__html ? r[1].__html : esc(r[1]));
      return "<tr><td>" + esc(r[0]) + "</td><td>" + v + "</td></tr>";
    }).join("") + "</table>";
  }

  /* --------------------------------------------------------------- toasts */
  function toast(msg, kind, title) {
    kind = kind || "info";
    var box = $("#sys-toasts");
    if (!box) {
      box = document.createElement("div");
      box.id = "sys-toasts";
      box.className = "sys-toasts";
      box.setAttribute("aria-live", "polite");
      document.body.appendChild(box);
    }
    var t = document.createElement("div");
    t.className = "sys-toast " + kind;
    t.setAttribute("role", kind === "error" ? "alert" : "status");
    var ic = {success: "check", error: "alert", warning: "alert", info: "info"}[kind] || "info";
    t.innerHTML = '<span class="st-ico">' + icon(ic) + '</span><div class="st-body">' +
      (title ? "<b>" + esc(title) + "</b>" : "") + "<span>" + esc(msg) + "</span></div>" +
      '<button type="button" class="btn-icon" aria-label="Dismiss">' + icon("x") + "</button>";
    var gone = function () { if (t.parentNode) t.remove(); };
    $("button", t).onclick = gone;
    box.appendChild(t);
    setTimeout(gone, kind === "error" ? 9000 : 5000);
    while (box.children.length > 4) box.firstChild.remove();
  }

  /* ----------------------------------------------------------- formatting */
  var fmt = {
    n: function (v) { return v == null || isNaN(v) ? "—" : Number(v).toLocaleString(); },
    dur: function (s) {
      s = Math.max(0, Math.floor(s || 0));
      var d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
      return d ? d + "d " + h + "h" : (h ? h + "h " + m + "m" : (m ? m + "m" : s + "s"));
    },
    mb: function (mb) { return mb >= 1000 ? (mb / 1000).toFixed(1) + " GB" : Math.round(mb) + " MB"; },
    hms: function (t) { return String(t || "").slice(11, 19); },
    plural: function (n, one, many) { return n + " " + (n === 1 ? one : (many || one + "s")); },
    today: function () {
      var d = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
      return d.toISOString().slice(0, 10);
    }
  };

  // Copy works on plain-http LAN addresses too (navigator.clipboard needs https).
  function copy(text) {
    var ok = false;
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      ok = document.execCommand("copy");
      ta.remove();
    } catch (e) { ok = false; }
    if (!ok && navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text);
      ok = true;
    }
    return ok;
  }

  // A short two-tone chime for new violations (only after the user switched
  // sound on - browsers allow audio after a click).
  var audioCtx = null;
  function beep() {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      [0, 0.18].forEach(function (at, i) {
        var o = audioCtx.createOscillator(), g = audioCtx.createGain();
        o.type = "sine";
        o.frequency.value = i ? 660 : 880;
        g.gain.setValueAtTime(0.0001, audioCtx.currentTime + at);
        g.gain.exponentialRampToValueAtTime(0.18, audioCtx.currentTime + at + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + at + 0.16);
        o.connect(g);
        g.connect(audioCtx.destination);
        o.start(audioCtx.currentTime + at);
        o.stop(audioCtx.currentTime + at + 0.18);
      });
    } catch (e) {}
  }

  var store = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

  window.UI = {$: $, $$: $$, esc: esc, icon: icon, api: api, modal: modal, confirm: confirm, kv: kv,
               toast: toast, fmt: fmt, copy: copy, beep: beep, store: store,
               openPopups: function () { return stack.length; }};
})();
