export function downloadBlob(name, blob) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 4000)
}

export function buildOBJ(heights, w, h, zScale = 1, step = 4) {
  const cols = Math.floor((w - 1) / step) + 1
  const rows = Math.floor((h - 1) / step) + 1
  const out = []
  out.push('# DepthWizard DSM export')
  out.push(`# grid ${w}x${h} sampled every ${step}px -> ${cols * rows} vertices`)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * step
      const y = r * step
      const z = heights[y * w + x] * zScale
      out.push(`v ${x} ${z.toFixed(3)} ${y}`)
    }
  }
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const i = r * cols + c + 1
      const j = i + 1
      const k = (r + 1) * cols + c + 1
      const l = k + 1
      out.push(`f ${i} ${k} ${j}`)
      out.push(`f ${j} ${k} ${l}`)
    }
  }
  return out.join('\n')
}

export function heightmapPNG(heights, w, h) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  const img = ctx.createImageData(w, h)
  let mn = Infinity
  let mx = -Infinity
  for (const v of heights) {
    if (v < mn) mn = v
    if (v > mx) mx = v
  }
  const span = mx - mn || 1
  for (let i = 0; i < heights.length; i++) {
    const v = Math.round(((heights[i] - mn) / span) * 255)
    img.data[i * 4] = v
    img.data[i * 4 + 1] = v
    img.data[i * 4 + 2] = v
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return new Promise((res) => c.toBlob(res, 'image/png'))
}

/**
 * Build a binary GLB of the height grid entirely in the browser (offline/demo
 * path — the backend has no job files for OFFLINE-* DSMs). Uses three's own
 * GLTFExporter, so no new dependency. Invalid cells (NaN / -9999 nodata) are
 * clamped to the grid minimum instead of zero to avoid export-time pits.
 */
export async function buildGLB(heights, w, h, zScale = 1, textureSource = null) {
  const THREE = await import('three')
  const { GLTFExporter } = await import('three/examples/jsm/exporters/GLTFExporter.js')

  const step = Math.max(1, Math.floor(Math.max(w, h) / 220))
  const cols = Math.floor((w - 1) / step) + 1
  const rows = Math.floor((h - 1) / step) + 1

  let mn = Infinity
  for (let i = 0; i < heights.length; i++) {
    const v = heights[i]
    if (Number.isFinite(v) && v !== -9999 && v < mn) mn = v
  }
  if (!Number.isFinite(mn)) mn = 0
  const at = (x, y) => {
    const v = heights[y * w + x]
    return Number.isFinite(v) && v !== -9999 ? v : mn
  }

  const geo = new THREE.PlaneGeometry(w - 1, h - 1, cols - 1, rows - 1)
  geo.rotateX(-Math.PI / 2)
  const pos = geo.attributes.position
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const sx = Math.min(w - 1, c * step)
      const sy = Math.min(h - 1, r * step)
      // PlaneGeometry vertex order is row-major from top-left; map to grid.
      const vi = r * cols + c
      pos.setY(vi, at(sx, sy) * zScale)
      pos.setX(vi, sx - (w - 1) / 2)
      pos.setZ(vi, sy - (h - 1) / 2)
    }
  }
  pos.needsUpdate = true
  geo.computeVertexNormals()

  let map = null
  if (textureSource instanceof HTMLCanvasElement) {
    map = new THREE.CanvasTexture(textureSource)
  } else if (typeof textureSource === 'string' && textureSource) {
    map = await new Promise((resolve) => {
      new THREE.TextureLoader().load(textureSource, resolve, undefined, () => resolve(null))
    })
  }
  if (map) map.colorSpace = THREE.SRGBColorSpace
  const mat = new THREE.MeshStandardMaterial({ map: map || null, roughness: 0.9, metalness: 0 })
  const mesh = new THREE.Mesh(geo, mat)

  const exporter = new GLTFExporter()
  const buffer = await exporter.parseAsync(mesh, { binary: true })
  geo.dispose()
  mat.dispose()
  map?.dispose()
  return new Blob([buffer], { type: 'model/gltf-binary' })
}
