/**
 * Single source of truth for DSM mode labels (was duplicated across
 * ViewerScreen / AnalysisPanel / ValidationPanel — see audit gap L-1).
 *
 * Modes: REAL metric (DEM/GCP) | REAL relative | OFFLINE demo (never metric).
 */
export function dsmModeInfo(dsm) {
  const meta = dsm?.metadata || null
  const offline = !!dsm?.offline || String(dsm?.id || '').startsWith('OFFLINE')
  const isMetric = !offline && (meta ? !!meta.is_metric : !!dsm?.crs)
  const method = meta?.calibration?.method || null
  const modeLabel = !dsm
    ? null
    : offline
      ? 'Demo Surface — NOT METRIC'
      : isMetric
        ? 'Metric DSM'
        : 'Relative Surface Model'
  const unitLabel = offline ? 'demo units' : isMetric ? 'm' : 'relative'
  const calibrationLabel = offline
    ? 'none — demo surface, not calibrated'
    : method === 'gcp'
      ? 'GCP affine fit (Z = a·d + b)'
      : method === 'dem'
        ? 'DEM-anchored affine fit (Z = a·d + b, RANSAC)'
        : 'none — relative surface (0–100), not metric'
  return {
    offline,
    isMetric,
    method,
    modeLabel,
    unitLabel,
    calibrationLabel,
    verticalRef: meta?.vertical_reference || null,
    crs: dsm?.crs || meta?.crs || null,
  }
}
