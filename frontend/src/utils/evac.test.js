// Tests use synthetic grids to validate ROUTING LOGIC only — never real places.
import { describe, expect, it } from 'vitest'
import { astar, downsampleGrid, maskCentroid, nearestFree, pickGoals, slopeGrid } from './evac'

const N = 8
function openGrid() {
  return {
    blocked: new Uint8Array(N * N),
    cost: new Float32Array(N * N),
  }
}

describe('astar', () => {
  it('finds the diagonal on open ground', () => {
    const { blocked, cost } = openGrid()
    const path = astar(N, [0, 0], [7, 7], blocked, cost)
    expect(path).toBeTruthy()
    expect(path[0]).toEqual([0, 0])
    expect(path[path.length - 1]).toEqual([7, 7])
    expect(path.length).toBeLessThanOrEqual(9)
  })

  it('routes around a wall through the gap', () => {
    const { blocked, cost } = openGrid()
    for (let y = 0; y < N; y++) {
      if (y !== 4) blocked[y * N + 3] = 1
    }
    const path = astar(N, [0, 0], [7, 7], blocked, cost)
    expect(path).toBeTruthy()
    expect(path.some(([x, y]) => x === 3 && y === 4)).toBe(true)
    expect(path.every(([x, y]) => !blocked[y * N + x])).toBe(true)
  })

  it('returns null when the goal is sealed', () => {
    const blocked = new Uint8Array(N * N).fill(1)
    blocked[0] = 0
    const path = astar(N, [0, 0], [7, 7], blocked, new Float32Array(N * N))
    expect(path).toBeNull()
  })

  it('prefers gentle ground over steep ground', () => {
    const { blocked } = openGrid()
    const cost = new Float32Array(N * N)
    for (let y = 0; y < N; y++) cost[y * N + 3] = 2.5 // steep ridge
    const path = astar(N, [0, 3], [7, 3], blocked, cost)
    expect(path).toBeTruthy()
    const onRidge = path.filter(([x]) => x === 3).length
    expect(onRidge).toBeLessThan(path.length)
  })
})

describe('grid helpers', () => {
  it('resamples by nearest source pixel with full coverage', () => {
    const h = new Float32Array([0, 2, NaN, 4])
    const { h: out } = downsampleGrid(h, 2, 2, 2)
    expect([...out].map((v) => (Number.isNaN(v) ? 'nan' : v))).toEqual([0, 2, 'nan', 4])
    // upsampling a coarse grid leaves no holes
    const { h: big } = downsampleGrid(new Float32Array([5, 6, 7, 8]), 2, 2, 6)
    expect(big.every((v) => Number.isFinite(v))).toBe(true)
  })

  it('picks high gentle cells as goals', () => {
    const h = new Float32Array(N * N).fill(1)
    h[63] = 10
    const s = new Float32Array(N * N).fill(0.1)
    const goals = pickGoals(h, s, N, 2)
    expect(goals.length).toBeGreaterThan(0)
    expect(goals[0]).toMatchObject({ x: 7, y: 7 })
  })

  it('centroids masks and reports empty ones', () => {
    const m = new Uint8Array(N * N)
    m[0] = 1
    m[8] = 1
    expect(maskCentroid(m, N)).toEqual([0, 1])
    expect(maskCentroid(new Uint8Array(N * N), N)).toBeNull()
  })

  it('finds nearest passable ground outside the hazard', () => {
    const blocked = new Uint8Array(N * N)
    for (let y = 2; y <= 5; y++) for (let x = 2; x <= 5; x++) blocked[y * N + x] = 1
    // blob spans x,y ∈ [2,5]; (4,4) is inside, nearest free ring is (6,2)-first
    expect(nearestFree(blocked, N, 4, 4)).toEqual([6, 2])
    expect(nearestFree(blocked, N, 0, 0)).toEqual([0, 0])
    expect(nearestFree(new Uint8Array(N * N).fill(1), N, 3, 3)).toBeNull()
  })
})
