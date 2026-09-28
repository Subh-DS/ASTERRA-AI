// Ingest forensics: probeTIFF must read the VALUE field, never the COUNT.
// Regression: every standard TIFF reported 1x1 (count=1 read as dimension).
import { describe, expect, it } from 'vitest'
import { probeTIFF } from './inspect.js'

function tiffBuffer({ little = true, entries = [], version = 42 }) {
  const n = entries.length
  const buf = new ArrayBuffer(8 + 2 + n * 12 + 4)
  const v = new DataView(buf)
  v.setUint16(0, little ? 0x4949 : 0x4d4d, false)
  v.setUint16(2, version, little)
  v.setUint32(4, 8, little)
  v.setUint16(8, n, little)
  entries.forEach(([tag, type, count, value], i) => {
    const e = 10 + i * 12
    v.setUint16(e, tag, little)
    v.setUint16(e + 2, type, little)
    v.setUint32(e + 4, count, little)
    if (type === 3) v.setUint16(e + 8, value, little)
    else v.setUint32(e + 8, value, little)
  })
  return buf
}

const SHORT = 3
const LONG = 4
const GEO_KEY_DIR = 34735

describe('probeTIFF dimensions', () => {
  it('reads SHORT width/height values, not counts', () => {
    const r = probeTIFF(tiffBuffer({ entries: [[256, SHORT, 1, 1024], [257, SHORT, 1, 768], [GEO_KEY_DIR, SHORT, 1, 1]] }))
    expect(r.width).toBe(1024)
    expect(r.height).toBe(768)
    expect(r.hasGeo).toBe(true)
  })

  it('reads LONG width/height values', () => {
    const r = probeTIFF(tiffBuffer({ entries: [[256, LONG, 1, 256], [257, LONG, 1, 192]] }))
    expect(r.width).toBe(256)
    expect(r.height).toBe(192)
    expect(r.hasGeo).toBe(false)
  })

  it('reads big-endian headers', () => {
    const r = probeTIFF(tiffBuffer({ little: false, entries: [[256, SHORT, 1, 64], [257, SHORT, 1, 48]] }))
    expect(r.width).toBe(64)
    expect(r.height).toBe(48)
  })

  it('reports a genuine 1x1 raster as 1x1', () => {
    const r = probeTIFF(tiffBuffer({ entries: [[256, SHORT, 1, 1], [257, SHORT, 1, 1]] }))
    expect(r.width).toBe(1)
    expect(r.height).toBe(1)
  })

  it('never reports 1x1 for unreadable headers (BigTIFF, garbage)', () => {
    const big = probeTIFF(tiffBuffer({ version: 43, entries: [[256, SHORT, 1, 5]] }))
    expect(big.width).toBeNull()
    expect(big.height).toBeNull()
    expect(probeTIFF(new ArrayBuffer(4)).width).toBeNull()
    expect(probeTIFF(new ArrayBuffer(0)).width).toBeNull()
  })

  it('leaves exotic-type dimensions null instead of guessing', () => {
    const r = probeTIFF(tiffBuffer({ entries: [[256, 16, 1, 7], [257, SHORT, 1, 9]] }))
    expect(r.width).toBeNull()
    expect(r.height).toBe(9)
  })
})
