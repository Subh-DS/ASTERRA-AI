// Phase 10: AOI helpers (mirror backend rules for instant feedback;
// the backend re-validates authoritatively).
export const AOI_LIMITS = { minSideM: 50, maxSideM: 2000, maxAreaKm2: 2.0, maxAspect: 8 }

export function validateAoiClient(b) {
  if (!b) return 'Draw a rectangle on the map to select an area.'
  const { north, south, east, west } = b
  for (const v of [north, south, east, west]) {
    if (typeof v !== 'number' || !isFinite(v)) return 'AOI coordinates must be numbers.'
  }
  if (Math.abs(north) > 90 || Math.abs(south) > 90) return 'Latitudes must be within ±90°.'
  if (Math.abs(east) > 180 || Math.abs(west) > 180) return 'Longitudes must be within ±180°.'
  if (north <= south) return 'North must be greater than south.'
  if (east <= west) return 'East must be greater than west.'
  return null
}

export function aoiMetrics(north, south, east, west) {
  const R = 6371000
  const latM = ((north + south) / 2) * (Math.PI / 180)
  const widthM = Math.abs(east - west) * (Math.PI / 180) * R * Math.cos(latM)
  const heightM = Math.abs(north - south) * (Math.PI / 180) * R
  return {
    centerLat: (north + south) / 2,
    centerLon: (east + west) / 2,
    widthM: Math.round(widthM),
    heightM: Math.round(heightM),
    areaKm2: widthM * heightM / 1e6,
  }
}

export function formatArea(km2) {
  if (km2 < 0.01) return `${Math.round(km2 * 1e6)} m²`
  if (km2 < 1) return `${(km2 * 100).toFixed(1)} ha`
  return `${km2.toFixed(2)} km²`
}

/** "20.29, 85.82" → {type:'coords'}; otherwise {type:'query', q}. */
export function parseLocationInput(text) {
  const m = String(text || '')
    .trim()
    .match(/^(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)$/)
  if (m) {
    const lat = parseFloat(m[1])
    const lon = parseFloat(m[2])
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return { type: 'coords', lat, lon }
  }
  const q = String(text || '').trim()
  return q ? { type: 'query', q } : null
}

export async function searchPlaces(q, limit = 5) {
  const res = await fetch(
    `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=${limit}&q=${encodeURIComponent(q)}`,
    { headers: { Accept: 'application/json' } },
  )
  if (!res.ok) throw new Error(`search failed (HTTP ${res.status})`)
  const arr = await res.json()
  return arr.map((r) => ({
    name: r.display_name,
    lat: parseFloat(r.lat),
    lon: parseFloat(r.lon),
    bbox: r.boundingbox ? {
      south: parseFloat(r.boundingbox[0]),
      north: parseFloat(r.boundingbox[1]),
      west: parseFloat(r.boundingbox[2]),
      east: parseFloat(r.boundingbox[3]),
    } : null,
  }))
}
