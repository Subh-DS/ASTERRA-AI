function hash2(x, y) {
  let h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123
  return h - Math.floor(h)
}

function vnoise(x, y) {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * (3 - 2 * xf)
  const v = yf * yf * (3 - 2 * yf)
  const a = hash2(xi, yi)
  const b = hash2(xi + 1, yi)
  const c = hash2(xi, yi + 1)
  const d = hash2(xi + 1, yi + 1)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

export function fbm2d(x, y, octaves = 5) {
  let f = 0
  let amp = 0.5
  let fx = x
  let fy = y
  for (let i = 0; i < octaves; i++) {
    f += amp * (vnoise(fx, fy) * 2 - 1)
    fx *= 2.03
    fy *= 1.97
    amp *= 0.5
  }
  return f
}

function ridge(x, y) {
  return 1 - Math.abs(fbm2d(x, y))
}

function boxBlur(grid, w, h, radius = 2, passes = 2) {
  let src = Float32Array.from(grid)
  let dst = new Float32Array(src.length)
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0
        let n = 0
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const xx = Math.min(w - 1, Math.max(0, x + dx))
            const yy = Math.min(h - 1, Math.max(0, y + dy))
            sum += src[yy * w + xx]
            n++
          }
        }
        dst[y * w + x] = sum / n
      }
    }
    ;[src, dst] = [dst, src]
  }
  return src
}

function normalize(g) {
  let mn = Infinity
  let mx = -Infinity
  for (const v of g) {
    if (v < mn) mn = v
    if (v > mx) mx = v
  }
  const span = mx - mn || 1
  return g.map((v) => (v - mn) / span)
}

function downsample(g, w, h, nw, nh) {
  const out = new Float32Array(nw * nh)
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.floor((x / nw) * w))
      const sy = Math.min(h - 1, Math.floor((y / nh) * h))
      out[y * nw + x] = g[sy * w + sx]
    }
  }
  return out
}

function pyramids(heights, size) {
  return [48, 96, 192].map((n) => ({ data: downsample(heights, size, size, n, n), w: n, h: n }))
}

export function makeSyntheticCity(size = 512) {
  const heights = new Float32Array(size * size)
  const cells = 9
  const cell = size / cells
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = x / size
      const ny = y / size
      const terrain = fbm2d(nx * 3.2, ny * 3.2) * 0.5 + 0.5
      const hill = ridge(nx * 1.6 + 3.1, ny * 1.6 - 1.4) * 0.35
      const cx = Math.floor(nx * cells)
      const cy = Math.floor(ny * cells)
      const inRoadX = (nx * size) % cell < 6 || (nx * size) % cell > cell - 6
      const inRoadY = (ny * size) % cell < 6 || (ny * size) % cell > cell - 6
      let b = 0
      if (!inRoadX && !inRoadY) {
        const seed = hash2(cx * 13.7, cy * 7.3)
        const fx = ((nx * size) % cell) / cell
        const fy = ((ny * size) % cell) / cell
        const inset = 0.12
        if (fx > inset && fx < 1 - inset && fy > inset && fy < 1 - inset) {
          const taper =
            Math.min(fx, 1 - fx, fy, 1 - fy) > inset + 0.06 ? 1 : Math.min(fx, 1 - fx, fy, 1 - fy) - inset
          b = (8 + seed * 46) * Math.min(1, taper / 0.06)
        }
      }
      heights[y * size + x] = terrain * 26 + hill * 40 + b + fbm2d(nx * 22, ny * 22) * 0.9
    }
  }
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(size, size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4
      const nx = x / size
      const ny = y / size
      const t = (heights[y * size + x] - 0) / 80
      const road =
        (nx * size) % (size / 9) < 6 || (nx * size) % (size / 9) > size / 9 - 6 ||
        (ny * size) % (size / 9) < 6 || (ny * size) % (size / 9) > size / 9 - 6
      let r, g, b
      if (road) {
        r = 52
        g = 54
        b = 50
      } else {
        const veg = fbm2d(nx * 9 + 11, ny * 9 - 7) * 0.5 + 0.5
        r = 58 + veg * 34 + t * 30
        g = 66 + veg * 42 + t * 26
        b = 48 + veg * 20 + t * 18
      }
      img.data[i] = r
      img.data[i + 1] = g
      img.data[i + 2] = b
      img.data[i + 3] = 255
    }
  }
ctx.putImageData(img, 0, 0)

  for (let cy = 0; cy < cells; cy++) {
    for (let cx = 0; cx < cells; cx++) {
      const seed = hash2(cx * 13.7, cy * 7.3)
      const bx = cx * cell
      const by = cy * cell
      const inset = cell * 0.14
      const bw = cell - inset * 2
      const bh = cell - inset * 2
      const shade = 120 + seed * 90
      ctx.fillStyle = `rgb(${Math.round(shade + 18)},${Math.round(shade + 8)},${Math.round(shade - 8)})`
      ctx.fillRect(bx + inset, by + inset, bw, bh)
      ctx.strokeStyle = 'rgba(10,16,12,0.55)'
      ctx.lineWidth = 2
      ctx.strokeRect(bx + inset + 1, by + inset + 1, bw - 2, bh - 2)
      ctx.fillStyle = 'rgba(255,244,214,0.16)'
      ctx.fillRect(bx + inset + 3, by + inset + 3, bw - 6, 3)
      if (seed > 0.55) {
        ctx.fillStyle = 'rgba(30,34,30,0.5)'
        ctx.fillRect(bx + inset + bw * 0.25, by + inset + bh * 0.3, bw * 0.2, bh * 0.2)
      }
    }
  }

  for (let p = 0; p < 5; p++) {
    const px = hash2(p * 3.3, 9.1) * (size - 80)
    const py = hash2(p * 7.7, 2.3) * (size - 80)
    ctx.fillStyle = 'rgba(52,96,58,0.85)'
    ctx.beginPath()
    ctx.ellipse(px + 40, py + 40, 34, 26, hash2(p, p) * 3, 0, Math.PI * 2)
    ctx.fill()
  }

  // Add subtle atmospheric haze overlay
  const haze = ctx.createLinearGradient(0, 0, 0, size)
  haze.addColorStop(0, 'rgba(10,16,12,0.15)')
  haze.addColorStop(1, 'rgba(10,16,12,0.02)')
  ctx.fillStyle = haze
  ctx.fillRect(0, 0, size, size)

  return { heights, width: size, height: size, textureSrc: canvas.toDataURL('image/png'), pyramids: pyramids(heights, size), landscape: 'urban' }
}

export function fromImageBitmap(bmp) {
  const size = 256
  const c = document.createElement('canvas')
  c.width = size
  c.height = size
  const ctx = c.getContext('2d')
  ctx.drawImage(bmp, 0, 0, size, size)
  const px = ctx.getImageData(0, 0, size, size).data
  let lum = new Float32Array(size * size)
  for (let i = 0; i < size * size; i++) {
    lum[i] = (px[i * 4] * 0.299 + px[i * 4 + 1] * 0.587 + px[i * 4 + 2] * 0.114) / 255
  }
  lum = boxBlur(lum, size, size, 3, 2)
  const base = normalize(lum)
  const heights = new Float32Array(size * size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = x / size
      const ny = y / size
      const structural = Math.pow(base[y * size + x], 1.35)
      const macro = (fbm2d(nx * 2.6 + 5, ny * 2.6) * 0.5 + 0.5) * 0.45
      heights[y * size + x] = structural * 62 + macro * 38
    }
  }
  const texC = document.createElement('canvas')
  const maxSide = Math.max(bmp.width, bmp.height)
  const scale = Math.min(1, 2048 / maxSide)
  texC.width = Math.round(bmp.width * scale)
  texC.height = Math.round(bmp.height * scale)
  texC.getContext('2d').drawImage(bmp, 0, 0, texC.width, texC.height)
  return {
    heights,
    width: size,
    height: size,
    textureSrc: texC.toDataURL('image/jpeg', 0.92),
    pyramids: pyramids(heights, size),
    landscape: 'mixed',
  }
}

export async function fileToImageBitmap(file) {
  return createImageBitmap(file)
}

export function computeStats(heights) {
  let mn = Infinity
  let mx = -Infinity
  for (const v of heights) {
    if (v < mn) mn = v
    if (v > mx) mx = v
  }
  return { min: mn, max: mx }
}

// DEMO-ONLY helper. Never display its output as real validation results:
// it invents a reference (heights + hash noise) so charts have something to
// render in offline development. ValidationPanel must NOT call this.
export function synthesizeValidation(heights, w, h) {
  let rmseSum = 0
  let maeSum = 0
  const scatter = []
  const step = Math.max(1, Math.floor(Math.sqrt((w * h) / 400)))
  for (let y = step >> 1; y < h; y += step) {
    for (let x = step >> 1; x < w; x += step) {
      const p = heights[y * w + x]
      const noise = (hash2(x * 3.77, y * 5.19) - 0.5) * 7 + (hash2(x * 0.31, y * 0.57) - 0.5) * 3
      const ref = p + noise
      const e = p - ref
      rmseSum += e * e
      maeSum += Math.abs(e)
      scatter.push([p, ref])
    }
  }
  const n = scatter.length
  const rmse = Math.sqrt(rmseSum / n)
  const mae = maeSum / n
  let sp = 0
  let sr = 0
  let spp = 0
  let srr = 0
  let spr = 0
  for (const [p, r] of scatter) {
    sp += p
    sr += r
    spp += p * p
    srr += r * r
    spr += p * r
  }
  const cov = spr / n - (sp / n) * (sr / n)
  const vp = spp / n - (sp / n) ** 2
  const vr = srr / n - (sr / n) ** 2
  const corr = cov / Math.sqrt(vp * vr || 1)
  const errs = scatter.map(([p, r]) => Math.abs(p - r)).sort((a, b) => a - b)
  const hist = new Array(24).fill(0)
  const maxE = errs[errs.length - 1] || 1
  for (const e of errs) hist[Math.min(23, Math.floor((e / maxE) * 24))]++
  return { rmse, mae, corr, scatter, hist, pass: rmse < 6, note: null }
}
