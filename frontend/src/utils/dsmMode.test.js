// Stage 7: REAL vs DEMO mode separation — the honesty-critical unit tests.
import { describe, expect, it } from 'vitest'
import { dsmModeInfo } from './dsmMode.js'

const metricDem = {
  id: 'J123', width: 10, height: 10, crs: 'EPSG:32633',
  metadata: {
    mode: 'metric', is_metric: true, unit: 'm', crs: 'EPSG:32633',
    vertical_reference: 'EGM96', calibration: { available: true, method: 'dem' },
  },
}

describe('dsmModeInfo', () => {
  it('labels DEM-calibrated jobs metric', () => {
    const m = dsmModeInfo(metricDem)
    expect(m.isMetric).toBe(true)
    expect(m.offline).toBe(false)
    expect(m.modeLabel).toBe('Metric DSM')
    expect(m.unitLabel).toBe('m')
    expect(m.calibrationLabel).toMatch(/RANSAC/)
  })

  it('labels GCP-calibrated jobs metric', () => {
    const dsm = { ...metricDem, metadata: { ...metricDem.metadata, calibration: { available: true, method: 'gcp' } } }
    const m = dsmModeInfo(dsm)
    expect(m.isMetric).toBe(true)
    expect(m.calibrationLabel).toMatch(/GCP/)
  })

  it('labels relative jobs not-metric', () => {
    const m = dsmModeInfo({ id: 'J1', metadata: { mode: 'relative', is_metric: false, calibration: { available: false, method: 'relative' } } })
    expect(m.isMetric).toBe(false)
    expect(m.modeLabel).toBe('Relative Surface Model')
  })

  it('forces demo jobs non-metric even with a CRS-shaped id/crs', () => {
    const m = dsmModeInfo({ id: 'OFFLINE-1', offline: true, crs: 'EPSG:32633', metadata: { mode: 'demo', is_metric: true } })
    expect(m.offline).toBe(true)
    expect(m.isMetric).toBe(false)
    expect(m.modeLabel).toBe('Demo Surface — NOT METRIC')
    expect(m.unitLabel).toBe('demo units')
  })

  it('detects OFFLINE ids without an explicit flag', () => {
    expect(dsmModeInfo({ id: 'OFFLINE-42' }).offline).toBe(true)
  })

  it('handles null DSM', () => {
    expect(dsmModeInfo(null).modeLabel).toBe(null)
  })
})
