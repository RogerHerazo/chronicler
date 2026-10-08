// Live status (timer, chunk countdown, level meters) and error toasts.
(function () {
  "use strict";

  const LABELS = { system: "System", mic: "Mic", file: "File" };

  function fmt(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function meterPct(db) {
    return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  }

  let timer = null;

  async function poll() {
    const strip = document.getElementById("status-strip");
    if (!strip) {
      stop();
      return;
    }
    let data;
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      data = await r.json();
    } catch {
      return;
    }
    if (!data.recording) return;
    strip.querySelector("[data-elapsed]").textContent = fmt(data.elapsed);
    const left = Math.max(0, data.chunk_seconds - data.buffered);
    strip.querySelector("[data-countdown]").textContent = fmt(left);
    strip.querySelector("[data-chunk-progress]").style.width =
      Math.min(100, (data.buffered / data.chunk_seconds) * 100) + "%";

    const banner = document.querySelector("[data-capture-error]");
    if (banner) {
      const messages = Object.entries(data.errors).map(
        ([label, error]) => `${LABELS[label] || label} stopped recording: ${error}`
      );
      for (const [label, clipping] of Object.entries(data.clipping || {})) {
        if (clipping && !data.errors[label]) {
          messages.push(
            `${LABELS[label] || label} is clipping (distorted). Lower its input volume in your system sound settings.`
          );
        }
      }
      banner.hidden = messages.length === 0;
      banner.textContent = messages.join(" · ");
    }

    const meters = strip.querySelector("[data-meters]");
    for (const [label, db] of Object.entries(data.levels)) {
      let row = meters.querySelector(`[data-meter="${label}"]`);
      if (!row) {
        row = document.createElement("div");
        row.className = "meter-row";
        row.dataset.meter = label;
        row.innerHTML = `<span></span><span class="meter"><span class="meter-fill"></span></span>`;
        row.firstElementChild.textContent = LABELS[label] || label;
        meters.appendChild(row);
      }
      const error = data.errors[label];
      row.title = error ? `Capture error: ${error}` : `${Math.round(db)} dB`;
      const fill = row.querySelector(".meter-fill");
      fill.style.width = (error ? 0 : meterPct(db)) + "%";
      fill.classList.toggle("is-clipping", Boolean(data.clipping && data.clipping[label]));
    }
  }

  function start() {
    if (!timer && document.getElementById("status-strip")) {
      poll();
      timer = setInterval(poll, 250);
    }
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function toast(message) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    clearTimeout(el._hide);
    el._hide = setTimeout(() => (el.textContent = ""), 8000);
  }

  document.addEventListener("DOMContentLoaded", start);
  document.addEventListener("htmx:afterSettle", start);

  // Show server-side errors (409 "fix health checks", 500 device errors, ...).
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
