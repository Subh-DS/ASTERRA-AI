// Phase 11: hazard helpers (decode canonical grids, format stats, clamp inputs).
import { api as baseApi } from '../api/client'

export const hazardApi = {
  simulate(jobId, type, parameters, scenarioName) {
    return baseApi.hazardFetch('/api/hazards/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, type, parameters, scenario_name: scenarioName }),
    }, 120000)
  },
}

export function b64ToU8(b64) {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export function b64ToF32(b64) {
  return new Float32Array(b64ToU8(b64).buffer)
}

/** Attach decoded typed arrays to a simulation result's grids (cached). */
export function decodeGrids(result) {
  const g = result.grids
  if (!g || g._decoded) return g
  const size = g.grid_h * g.grid_w
  const take = (key, fn) => {
    if (!g[key]) return null
    const a = fn(g[key])
    return a.length === size ? a : null
  }
  g._maskU8 = take('mask_b64', b64ToU8)
  g._depthF32 = take('depth_b64', b64ToF32)
  g._sourceU8 = take('source_b64', b64ToU8)
  g._pathU8 = take('path_mask_b64', b64ToU8)
  g._depU8 = take('deposition_b64', b64ToU8)
  g._decoded = true
  return g
}

export function clampWaterLevel(v, maxElev) {
  const x = Number(v)
  if (!isFinite(x)) return null
  const hi = Math.min(10, Math.max(0.1, maxElev ?? 10))
  return Math.min(hi, Math.max(0.1, Math.round(x * 10) / 10))
}

export const DISCLAIMER_SHORT = 'Scenario visualization — not a disaster forecast or engineering prediction.'
