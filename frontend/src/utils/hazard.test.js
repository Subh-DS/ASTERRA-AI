import { describe, expect, it } from 'vitest'
import { b64ToF32, b64ToU8, clampWaterLevel, decodeGrids, DISCLAIMER_SHORT } from './hazard'

function b64(arr) {
  const bytes = new Uint8Array(arr.buffer || arr)
  let s = ''
  for (const b of bytes) s += String.fromCharCode(b)
  return btoa(s)
}

describe('b64 decoders', () => {
  it('round-trips uint8 grids', () => {
    const enc = b64(new Uint8Array([0, 1, 1, 0]))
    expect([...b64ToU8(enc)]).toEqual([0, 1, 1, 0])
  })
  it('round-trips float32 grids', () => {
    const enc = b64(new Float32Array([0.5, 3.25]))
    expect([...b64ToF32(enc)]).toEqual([0.5, 3.25])
  })
})

describe('decodeGrids', () => {
  it('decodes coastal grids and ignores size mismatch', () => {
    const r = { grids: {
      grid_h: 2, grid_w: 2,
      mask_b64: b64(new Uint8Array([1, 0, 1, 0])),
      depth_b64: b64(new Float32Array([2, 0, 1, 0])),
    } }
    const g = decodeGrids(r)
    expect(g._maskU8.length).toBe(4)
    expect(g._depthF32[0]).toBe(2)
    expect(decodeGrids(r)).toBe(g) // cached
  })
})

describe('clampWaterLevel', () => {
  it('clamps to 0.1–10 m', () => {
    expect(clampWaterLevel(3, 20)).toBe(3)
    expect(clampWaterLevel(0, 20)).toBe(0.1)
    expect(clampWaterLevel(99, 20)).toBe(10)
    expect(clampWaterLevel('x', 20)).toBeNull()
  })
  it('caps at terrain max when lower', () => {
    expect(clampWaterLevel(9, 4.2)).toBe(4.2)
  })
})

describe('disclaimer', () => {
  it('never claims a forecast', () => {
    expect(DISCLAIMER_SHORT).toMatch(/not a disaster forecast/)
    expect(DISCLAIMER_SHORT).not.toMatch(/will be destroyed|exact fatalities/i)
  })
})
