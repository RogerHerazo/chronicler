// Live status (timer, chunk countdown, level meters), the Settings level
// check, and error toasts.
(function () {
  "use strict";

  const LABELS = { system: "System", mic: "Mic", file: "File" };
  const HOLD_MS = 1500;

  function fmt(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  // -60..0 dBFS onto 0..100 %, matching the green/yellow/red zones in app.css.
  function meterPct(db) {
    return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  }

  // --- meters ------------------------------------------------------------------

  function meterRow(container, label) {
    let row = container.querySelector(`[data-meter="${label}"]`);
    if (!row) {
      row = document.createElement("div");
      row.className = "meter-row";
      row.dataset.meter = label;
      row.innerHTML =
        '<span class="meter-label"></span>' +
        '<span class="vu" role="meter" aria-valuemin="-60" aria-valuemax="0">' +
        '<span class="vu-mask"></span><span class="vu-hold"></span></span>' +
        '<span class="meter-db"></span>';
      row.querySelector(".meter-label").textContent = LABELS[label] || label;
      row.querySelector(".vu").setAttribute("aria-label", `${LABELS[label] || label} level`);
      row._hold = { db: -90, at: 0 };
      container.appendChild(row);
    }
    return row;
  }

  function renderMeters(container, data) {
    const now = performance.now();
    for (const [label, db] of Object.entries(data.levels || {})) {
      const row = meterRow(container, label);
      const error = (data.errors || {})[label];
      const level = error ? -90 : db;
      if (level >= row._hold.db || now - row._hold.at > HOLD_MS) {
        row._hold = { db: level, at: now };
      }
      const pct = meterPct(level);
      row.querySelector(".vu-mask").style.width = 100 - pct + "%";
      const hold = row.querySelector(".vu-hold");
      hold.style.left = meterPct(row._hold.db) + "%";
      hold.classList.toggle("is-visible", row._hold.db > -60);
      row.querySelector(".vu").setAttribute("aria-valuenow", Math.round(level));
      const clipping = Boolean((data.clipping || {})[label]);
      row.querySelector(".meter-db").textContent = error
        ? "error"
        : clipping
          ? "CLIP"
          : level <= -60
            ? "–"
            : `${Math.round(row._hold.db)} dB`;
      row.classList.toggle("is-clipping", clipping);
      row.title = error ? `Capture error: ${error}` : "";
    }
  }

  function renderProblems(banner, data) {
    if (!banner) return;
    const errors = data.errors || {};
    const messages = Object.entries(errors).map(
      ([label, error]) => `${LABELS[label] || label} stopped recording: ${error}`
    );
    for (const [label, clipping] of Object.entries(data.clipping || {})) {
      if (clipping && !errors[label]) {
        messages.push(
          `${LABELS[label] || label} is clipping (distorted). Turn its input volume down in your system sound settings.`
        );
      }
    }
    banner.hidden = messages.length === 0;
    banner.textContent = messages.join(" · ");
  }

  // --- live session ---------------------------------------------------------------

  let liveTimer = null;

  async function pollLive() {
    const strip = document.getElementById("status-strip");
    if (!strip) {
      clearInterval(liveTimer);
      liveTimer = null;
      return;
    }
    let data;
    try {
      data = await (await fetch("/api/status", { cache: "no-store" })).json();
    } catch {
      return;
    }
    if (!data.recording) return;
    strip.querySelector("[data-elapsed]").textContent = fmt(data.elapsed);
    const left = Math.max(0, data.chunk_seconds - data.buffered);
    strip.querySelector("[data-countdown]").textContent = fmt(left);
    strip.querySelector("[data-chunk-progress]").style.width =
      Math.min(100, (data.buffered / data.chunk_seconds) * 100) + "%";
    renderMeters(strip.querySelector("[data-meters]"), data);
    renderProblems(document.querySelector("[data-capture-error]"), data);
  }

  function startLive() {
    if (!liveTimer && document.getElementById("status-strip")) {
      pollLive();
      liveTimer = setInterval(pollLive, 100);
    }
  }

  // --- settings level check ------------------------------------------------------

  let monitorTimer = null;

  function stopMonitorPolling(box, message) {
    clearInterval(monitorTimer);
    monitorTimer = null;
    if (!box) return;
    box.querySelector("[data-monitor-start]").hidden = false;
    box.querySelector("[data-monitor-stop]").hidden = true;
    box.querySelector("[data-monitor-status]").textContent = message || "";
  }

  async function pollMonitor(box) {
    if (!document.body.contains(box)) {
      stopMonitorPolling(null);
      fetch("/settings/monitor/stop", { method: "POST" });
      return;
    }
    let data;
    try {
      data = await (await fetch("/api/monitor", { cache: "no-store" })).json();
    } catch {
      return;
    }
    renderMeters(box.querySelector("[data-meters]"), data);
    renderProblems(box.querySelector("[data-capture-error]"), data);
    if (data.running) {
      box.querySelector("[data-monitor-status]").textContent = `Listening… stops in ${data.remaining}s`;
    } else {
      stopMonitorPolling(box, "Stopped.");
    }
  }

  async function startMonitor(box) {
    const status = box.querySelector("[data-monitor-status]");
    status.textContent = "Opening devices…";
    box.querySelector("[data-meters]").replaceChildren();
    const r = await fetch("/settings/monitor/start", { method: "POST" });
    if (!r.ok) {
      let message = `Could not open the devices (${r.status}).`;
      try {
        message = (await r.json()).detail || message;
      } catch {
        /* not JSON */
      }
      stopMonitorPolling(box, "");
      toast(message);
      return;
    }
    box.querySelector("[data-monitor-start]").hidden = true;
    box.querySelector("[data-monitor-stop]").hidden = false;
    clearInterval(monitorTimer);
    monitorTimer = setInterval(() => pollMonitor(box), 100);
  }

  document.addEventListener("click", (e) => {
    const box = e.target.closest("[data-monitor]");
    if (!box) return;
    if (e.target.closest("[data-monitor-start]")) startMonitor(box);
    if (e.target.closest("[data-monitor-stop]")) {
      fetch("/settings/monitor/stop", { method: "POST" });
      stopMonitorPolling(box, "Stopped.");
    }
  });

  // Free the microphone when leaving the page mid-check.
  window.addEventListener("pagehide", () => {
    if (monitorTimer) navigator.sendBeacon("/settings/monitor/stop");
  });

  // --- toasts ------------------------------------------------------------------------

  function toast(message) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    clearTimeout(el._hide);
    el._hide = setTimeout(() => (el.textContent = ""), 8000);
  }

  document.addEventListener("DOMContentLoaded", startLive);
  document.addEventListener("htmx:afterSettle", startLive);

  // Show server-side errors (409 "fix health checks", device errors, ...).
  document.addEventListener("htmx:responseError", (e) => {
    const xhr = e.detail.xhr;
    let message = `Something went wrong (${xhr.status}).`;
    try {
      const body = JSON.parse(xhr.responseText);
      if (body.detail) message = body.detail;
    } catch {
      /* not JSON */
    }
    toast(message);
  });
})();
