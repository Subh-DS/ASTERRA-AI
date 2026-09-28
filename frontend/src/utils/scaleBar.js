// Pure helpers for the HUD scale bar (display-only; tested).
export function niceScale(raw) {
  const m = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))))
  const n = raw / m
  return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * m
}

export function formatScale(val, unit) {
  if (unit === 'm') {
    return val >= 1000 ? `${(val / 1000).toFixed(val >= 10000 ? 0 : 1)} km` : `${+val.toFixed(2)} m`
  }
  if (unit === 'demo') return `${+val.toFixed(1)} demo-u`
  return `${+val.toFixed(1)} rel-u`
}
