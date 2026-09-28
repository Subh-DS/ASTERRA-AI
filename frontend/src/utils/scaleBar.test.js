import { describe, expect, it } from 'vitest'
import { formatScale, niceScale } from './scaleBar.js'

describe('scale bar helpers', () => {
  it('snaps to 1/2/5 steps', () => {
    expect(niceScale(110)).toBe(100)
    expect(niceScale(230)).toBe(200)
    expect(niceScale(480)).toBe(200)
    expect(niceScale(520)).toBe(500)
    expect(niceScale(0.37)).toBeCloseTo(0.2, 10)
  })
  it('labels metric honestly', () => {
    expect(formatScale(250, 'm')).toBe('250 m')
    expect(formatScale(2500, 'm')).toBe('2.5 km')
    expect(formatScale(3.2, 'rel')).toBe('3.2 rel-u')
    expect(formatScale(3.2, 'demo')).toBe('3.2 demo-u')
  })
})
