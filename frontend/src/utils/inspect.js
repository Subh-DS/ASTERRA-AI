const GEO_TAGS = new Set([34735, 34736, 34737, 33550, 33922, 34264])

export function probeTIFF(buffer) {
  const view = new DataView(buffer)
  if (buffer.byteLength < 16) return { hasGeo: false, width: null, height: null }
  const bo = view.getUint16(0, false)
  let little
  if (bo === 0x4949) little = true
  else if (bo === 0x4d4d) little = false
  else return { hasGeo: false, width: null, height: null }
  if (view.getUint16(2, little) !== 42) return { hasGeo: false, width: null, height: null }
  const ifdOffset = view.getUint32(4, little)
  if (ifdOffset + 2 > buffer.byteLength) return { hasGeo: false, width: null, height: null }
  const count = view.getUint16(ifdOffset, little)
  let hasGeo = false
  let width = null
  let height = null
  // IFD entry: tag u16 | type u16 | count u32 | value-or-offset u32.
  // Forensic fix: the old code read the COUNT field as the dimension, so
  // every standard TIFF (count=1) reported 1x1. Read the VALUE field.
  const inlineValue = (entry, type, num) => {
    if (num < 1) return null
    if (type === 3 && num <= 2) return view.getUint16(entry + 8, little) // SHORT
    if (type === 4 && num === 1) return view.getUint32(entry + 8, little) // LONG
    return null // offset-stored / exotic types: honestly unknown
  }
  for (let i = 0; i < count; i++) {
    const entry = ifdOffset + 2 + i * 12
    if (entry + 12 > buffer.byteLength) break
    const tag = view.getUint16(entry, little)
    const type = view.getUint16(entry + 2, little)
    const num = view.getUint32(entry + 4, little)
    if (GEO_TAGS.has(tag)) hasGeo = true
    if (tag === 256 && (type === 3 || type === 4)) width = inlineValue(entry, type, num)
    if (tag === 257 && (type === 3 || type === 4)) height = inlineValue(entry, type, num)
  }
  return { hasGeo, width, height }
}

export async function inspectFile(file) {
  const info = {
    name: file.name,
    size: file.size,
    ext: (file.name.split('.').pop() || '').toLowerCase(),
    kind: 'unknown',
    width: null,
    height: null,
    hasCRS: false,
  }
  if (info.ext === 'tif' || info.ext === 'tiff') {
    info.kind = 'TIFF'
    try {
      const buf = await file.slice(0, Math.min(file.size, 512 * 1024)).arrayBuffer()
      const probed = probeTIFF(buf)
      info.hasCRS = !!probed.hasGeo
      info.width = probed.width
      info.height = probed.height
    } catch {
      /* header unreadable */
    }
  } else {
    info.kind = info.ext.toUpperCase() || 'IMAGE'
    try {
      const bmp = await createImageBitmap(file)
      info.width = bmp.width
      info.height = bmp.height
      bmp.close?.()
    } catch {
      /* not decodable client-side */
    }
  }
  return info
}

export function fmtBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  return `${(n / 1e3).toFixed(0)} kB`
}

export function fmtInt(n) {
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 })
}

export function fmtM(n, digits = 2) {
  return `${n.toFixed(digits)} m`
}
