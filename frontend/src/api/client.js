export const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function jsonFetch(path, opts = {}, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...opts,
      signal: ctrl.signal,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  async health() {
    try {
      await jsonFetch("/api/health", {}, 2500);
      return true;
    } catch {
      return false;
    }
  },

  // Full health payload for truthful UI labels (cpu vs cuda, queue depth).
  // Never throws — returns { up:false } when unreachable.
  async engineInfo() {
    try {
      const h = await jsonFetch("/api/health", {}, 2500);
      const eng = h.engine || {};
      return {
        up: true,
        device: eng.fallback ? "fallback" : eng.device || null,
        model: eng.fallback ? "pseudo-fallback" : eng.model || null,
        modelLoaded: !!eng.model_loaded,
        queued: h.jobs_queued ?? 0,
        maxJobs: h.max_jobs ?? null,
      };
    } catch {
      return { up: false };
    }
  },

  uploadImage(file, gcps = []) {
    const fd = new FormData();

    fd.append("image", file);

    if (gcps.length) {
      fd.append(
        "gcps",
        JSON.stringify(
          gcps.filter((g) => g.lat && g.lon && g.elev)
        )
      );
    }

    return jsonFetch(
      "/api/jobs",
      {
        method: "POST",
        body: fd,
      },
      60000
    );
  },

  getJob(jobId) {
    return jsonFetch(`/api/jobs/${jobId}`);
  },

  retry(jobId) {
    return jsonFetch(
      `/api/jobs/${jobId}/retry`,
      { method: "POST" },
      30000
    );
  },

  async getDsmBinary(jobId, timeoutMs = 120000, token = "") {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(
        `${API_URL}/api/jobs/${jobId}/dsm.bin${token ? `?token=${encodeURIComponent(token)}` : ""}`,
        { signal: ctrl.signal }
      );

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const buf = await res.arrayBuffer();

      return new Float32Array(buf);
    } finally {
      clearTimeout(t);
    }
  },

  async getTerrainBinary(jobId, timeoutMs = 120000, token = "") {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(
        `${API_URL}/api/jobs/${jobId}/terrain.bin${token ? `?token=${encodeURIComponent(token)}` : ""}`,
        { signal: ctrl.signal },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return new Float32Array(await res.arrayBuffer());
    } finally {
      clearTimeout(t);
    }
  },

  validate(jobId, refFile) {
    const fd = new FormData();

    fd.append("reference", refFile);

    return jsonFetch(
      `/api/jobs/${jobId}/validate`,
      {
        method: "POST",
        body: fd,
      },
      120000
    );
  },

  exportUrl(jobId, kind, token = "") {
    const q = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${API_URL}/api/jobs/${jobId}/export/${kind}${q}`;
  },

  // Phase 11: hazard scenarios (plain JSON helper, no FormData).
  async hazardFetch(path, opts = {}, timeout = 30000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const res = await fetch(`${API_URL}${path}`, { ...opts, signal: ctrl.signal });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          if (body?.detail) detail = body.detail;
        } catch { /* keep HTTP status */ }
        throw new Error(detail);
      }
      return res.json();
    } finally {
      clearTimeout(timer);
    }
  },

  // Phase 10: map-driven jobs.
  imageryProviders() {
    return jsonFetch("/api/imagery/providers");
  },

  imagerySearch(aoi, { provider = "auto", maxCloud = null } = {}) {
    const q = new URLSearchParams({
      north: aoi.north, south: aoi.south, east: aoi.east, west: aoi.west,
      provider,
      ...(maxCloud != null ? { max_cloud: maxCloud } : {}),
    });
    return jsonFetch(`/api/imagery/search?${q.toString()}`, {}, 60000);
  },

  createMapJob({ aoi, provider = "auto", itemId = null, maxCloud = null, quality = "medium" }) {
    return jsonFetch("/api/map-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        aoi: { north: aoi.north, south: aoi.south, east: aoi.east, west: aoi.west },
        provider, item_id: itemId, max_cloud: maxCloud, quality,
      }),
    }, 120000);
  },
};

export class JobSocket {
  constructor(jobId, handlers = {}) {
    this.jobId = jobId;
    this.handlers = handlers;
    this.closedByUser = false;
    this.retries = 0;

    this.connect();
  }

  connect() {
    const wsUrl = API_URL.replace(/^http/, "ws");

    const url =
      `${wsUrl}/api/ws/jobs/${this.jobId}`;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this.fail();
      return;
    }

    this.ws.onmessage = (e) => {
      try {
        this.handlers.onEvent?.(JSON.parse(e.data));
      } catch {
        // Ignore malformed frame
      }
    };

    this.ws.onopen = () => {
      this.retries = 0;
      this.handlers.onOpen?.();
    };

    this.ws.onclose = () => {
      if (this.closedByUser) return;

      this.fail();
    };

    this.ws.onerror = () => {
      this.ws.close();
    };
  }

  fail() {
    if (this.retries < 6) {
      this.retries += 1;

      setTimeout(
        () => this.connect(),
        Math.min(500 * 2 ** this.retries, 6000)
      );

      this.handlers.onRetry?.(this.retries);
    } else {
      this.handlers.onDead?.();
    }
  }

  close() {
    this.closedByUser = true;
    this.ws?.close();
  }
}
