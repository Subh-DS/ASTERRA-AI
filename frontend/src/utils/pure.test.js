// Stage 7: pure-utility regression tests (node env, no DOM).
import { describe, expect, it } from 'vitest'
import { buildOBJ } from './exporters.js'
import { resolveTier } from '../quality/detectTier.js'
import { fmtBytes, fmtInt, probeTIFF } from './inspect.js'
import { computeStats, fbm2d } from './sampleScene.js'

describe('buildOBJ', () => {
  it('emits vertices + faces for a small grid', () => {
    const obj = buildOBJ(new Float32Array([1, 2, 3, 4]), 2, 2, 1, 1)
    const lines = obj.split('\n')
    expect(lines.filter((l) => l.startsWith('v '))).toHaveLength(4)
    expect(lines.filter((l) => l.startsWith('f '))).toHaveLength(2)
    expect(lines[0]).toMatch(/DepthWizard/)
  })
})

describe('resolveTier', () => {
  it('honors explicit overrides, falls back to auto', () => {
    expect(resolveTier('high', 'low')).toBe('low')
    expect(resolveTier('high', null)).toBe('high')
    expect(resolveTier('bogus', null)).toBe('medium')
  })
})

describe('inspect formatters + TIFF probe', () => {
  it('formats bytes', () => {
    expect(fmtBytes(500)).toMatch(/kB/)
    expect(fmtBytes(2_500_000)).toMatch(/MB/)
    expect(fmtInt(1234567)).toBe('1,234,567')
  })

  it('detects a minimal little-endian TIFF without geo tags', () => {
    const buf = new ArrayBuffer(32)
    const v = new DataView(buf)
    v.setUint16(0, 0x4949, false) // 'II'
    v.setUint16(2, 42, true)
    v.setUint32(4, 8, true)
    v.setUint16(8, 0, true) // zero IFD entries
    expect(probeTIFF(buf)).toMatchObject({ hasGeo: false })
  })

  it('rejects non-TIFF bytes', () => {
    expect(probeTIFF(new ArrayBuffer(32)).hasGeo).toBe(false)
  })
})

describe('sampleScene stats', () => {
  it('computes min/max over the grid', () => {
    const s = computeStats(new Float32Array([3, 1, 2]))
    expect(s.min).toBe(1)
    expect(s.max).toBe(3)
  })

  it('fbm2d is deterministic', () => {
    expect(fbm2d(1.7, 2.3)).toBe(fbm2d(1.7, 2.3))
  })
})
