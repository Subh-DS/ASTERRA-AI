import { describe, expect, it } from 'vitest'
import { AOI_LIMITS, aoiMetrics, formatArea, parseLocationInput, validateAoiClient } from './aoi'

describe('validateAoiClient', () => {
  it('rejects missing AOI', () => {
    expect(validateAoiClient(null)).toMatch(/draw/i)
  })
  it('accepts a sane Bhubaneswar rectangle', () => {
    expect(validateAoiClient({ north: 20.3, south: 20.29, east: 85.83, west: 85.82 })).toBeNull()
  })
  it('rejects inverted bounds', () => {
    expect(validateAoiClient({ north: 20.29, south: 20.3, east: 85.83, west: 85.82 })).toMatch(/north/i)
    expect(validateAoiClient({ north: 20.3, south: 20.29, east: 85.82, west: 85.83 })).toMatch(/east/i)
  })
  it('rejects out-of-range coordinates', () => {
    expect(validateAoiClient({ north: 95, south: 20, east: 85, west: 84 })).toMatch(/±90/)
    expect(validateAoiClient({ north: 20, south: 19, east: 200, west: 84 })).toMatch(/±180/)
  })
})

describe('aoiMetrics', () => {
  it('computes a ~1.1km square near Bhubaneswar', () => {
    const m = aoiMetrics(20.3, 20.29, 85.83, 85.82)
    expect(m.widthM).toBeGreaterThan(900)
    expect(m.widthM).toBeLessThan(1200)
    expect(m.areaKm2).toBeGreaterThan(1)
    expect(m.centerLat).toBeCloseTo(20.295, 6)
  })
  it('flags oversize selections against AOI_LIMITS', () => {
    const m = aoiMetrics(20.4, 20.2, 85.9, 85.7) // ~22km x ~21km
    expect(m.areaKm2).toBeGreaterThan(AOI_LIMITS.maxAreaKm2)
  })
})

describe('formatArea', () => {
  it('uses m² for tiny areas', () => {
    expect(formatArea(0.0004)).toMatch(/m²/)
  })
  it('uses km² for large areas', () => {
    expect(formatArea(1.5)).toMatch(/km²/)
  })
})

describe('parseLocationInput', () => {
  it('parses lat,lon pairs', () => {
    expect(parseLocationInput('20.2961, 85.8245')).toEqual({ type: 'coords', lat: 20.2961, lon: 85.8245 })
  })
  it('treats names as queries', () => {
    expect(parseLocationInput('Bhubaneswar')).toEqual({ type: 'query', q: 'Bhubaneswar' })
  })
  it('returns null for empty input', () => {
    expect(parseLocationInput('  ')).toBeNull()
  })
})
