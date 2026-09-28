// Evacuation routing over the real DSM grid (Phase 36).
// A* with slope + hazard costs. All inputs are measured terrain or backend
// simulation masks; every rendered route is labeled SIMULATED and is a
// planning visualization, never a verified safe path.

function idx(x, y, n) {
  return y * n + x
}

/** Resample a W×H height grid to N×N by nearest source pixel (full coverage,
 *  no invented values — every cell traces to a measured source sample). */
export function downsampleGrid(heights, W, H, N = 64) {
  const h = new Float32Array(N * N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const c = Math.min(W - 1, Math.floor(((x + 0.5) / N) * W))
      const r = Math.min(H - 1, Math.floor(((y + 0.5) / N) * H))
      const v = heights[r * W + c]
      h[y * N + x] = Number.isFinite(v) ? v : NaN
    }
  }
  return { h, pxPerCellX: W / N, pxPerCellY: H / N }
}

/** Slope magnitude per cell in grid-height units (0 = flat). */
export function slopeGrid(h, N) {
  const s = new Float32Array(N * N)
  const at = (x, y) => {
    x = Math.min(N - 1, Math.max(0, x))
    y = Math.min(N - 1, Math.max(0, y))
    return h[idx(x, y, N)]
  }
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const c = at(x, y)
      if (!Number.isFinite(c)) {
        s[idx(x, y, N)] = NaN
        continue
      }
      const xp = at(x + 1, y)
      const xm = at(x - 1, y)
      const yp = at(x, y + 1)
      const ym = at(x, y - 1)
      const dx = ((Number.isFinite(xp) ? xp : c) - (Number.isFinite(xm) ? xm : c)) / 2
      const dy = ((Number.isFinite(yp) ? yp : c) - (Number.isFinite(ym) ? ym : c)) / 2
      s[idx(x, y, N)] = Math.hypot(dx, dy)
    }
  }
  return s
}

/**
 * A* over an N×N grid. cost(field) per step = dist * (1 + slopeK*slope + hzK*hazard).
 * blocked: Uint8Array(1 = impassable). Returns array of [x,y] cells or null.
 */
export function astar(N, start, goal, blocked, costField, { slopeK = 2.5, hzK = 6 } = {}) {
  const [sx, sy] = start
  const [gx, gy] = goal
  const ok = (x, y) => x >= 0 && y >= 0 && x < N && y < N && !blocked[idx(x, y, N)]
  if (!ok(sx, sy) || !ok(gx, gy)) return null
  const g = new Float64Array(N * N).fill(Infinity)
  const came = new Int32Array(N * N).fill(-1)
  const closed = new Uint8Array(N * N)
  const open = [[0, idx(sx, sy, N)]]
  g[idx(sx, sy, N)] = 0
  const hOf = (x, y) => Math.hypot(x - gx, y - gy)
  const DIRS = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]]
  let guard = N * N * 8
  while (open.length && guard-- > 0) {
    let bi = 0
    for (let i = 1; i < open.length; i++) if (open[i][0] < open[bi][0]) bi = i
    const [, cur] = open.splice(bi, 1)[0]
    if (closed[cur]) continue
    closed[cur] = 1
    const cx = cur % N
    const cy = Math.floor(cur / N)
    if (cx === gx && cy === gy) {
      const path = []
      let c = cur
      while (c !== -1) {
        path.push([c % N, Math.floor(c / N)])
        c = came[c]
      }
      return path.reverse()
    }
    for (const [dx, dy, dist] of DIRS) {
      const nx = cx + dx
      const ny = cy + dy
      if (!ok(nx, ny)) continue
      const ni = idx(nx, ny, N)
      if (closed[ni]) continue
      const rawSl = costField ? costField[ni] || 0 : 0
      const sl = Number.isFinite(rawSl) ? rawSl : 0
      const rawHz = costField?.hz ? costField.hz[ni] || 0 : 0
      const hz = Number.isFinite(rawHz) ? rawHz : 0
      const step = dist * (1 + slopeK * Math.min(sl, 3) + hzK * Math.min(hz, 1))
      const ng = g[cur] + step
      if (ng < g[ni]) {
        g[ni] = ng
        came[ni] = cur
        open.push([ng + hOf(nx, ny), ni])
      }
    }
  }
  return null
}

/** Highest, gentlest cells as safe-zone goals (percentile-based, spread out). */
export function pickGoals(h, slope, N, k = 3) {
  const vals = []
  for (let i = 0; i < N * N; i++) {
    if (Number.isFinite(h[i]) && Number.isFinite(slope[i])) vals.push(h[i])
  }
  if (!vals.length) return []
  vals.sort((a, b) => a - b)
  const cutoff = vals[Math.floor(vals.length * 0.85)] ?? vals[vals.length - 1]
  const cands = []
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = idx(x, y, N)
      if (h[i] >= cutoff && slope[i] < 0.6) cands.push({ x, y, h: h[i], s: slope[i] })
    }
  }
  cands.sort((a, b) => b.h - a.h || a.s - b.s)
  const goals = []
  for (const c of cands) {
    if (goals.length >= k) break
    if (goals.every((g) => Math.hypot(g.x - c.x, g.y - c.y) > N / 5)) goals.push(c)
  }
  return goals
}

/** Centroid cell of a mask (Uint8Array), or null when empty. */
export function maskCentroid(mask, N) {
  let sx = 0
  let sy = 0
  let n = 0
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (mask[idx(x, y, N)]) {
        sx += x
        sy += y
        n++
      }
    }
  }
  if (!n) return null
  return [Math.round(sx / n), Math.round(sy / n)]
}

/**
 * Nearest passable cell to (x, y) by BFS ring expansion. Evacuation starts
 * from the closest escapable ground near the affected area — never inside
 * the hazard itself.
 */
export function nearestFree(blocked, N, x, y) {
  x = Math.min(N - 1, Math.max(0, Math.round(x)))
  y = Math.min(N - 1, Math.max(0, Math.round(y)))
  if (!blocked[idx(x, y, N)]) return [x, y]
  const seen = new Uint8Array(N * N)
  seen[idx(x, y, N)] = 1
  let frontier = [[x, y]]
  while (frontier.length) {
    const next = []
    for (const [cx, cy] of frontier) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue
          const nx = cx + dx
          const ny = cy + dy
          if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue
          const ni = idx(nx, ny, N)
          if (seen[ni]) continue
          seen[ni] = 1
          if (!blocked[ni]) return [nx, ny]
          next.push([nx, ny])
        }
      }
    }
    frontier = next
  }
  return null
}
