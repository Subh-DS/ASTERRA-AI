import { useState, useRef, useCallback } from 'react'
import { parseLocationInput, searchPlaces } from '../../utils/aoi'

// Accepts "lat, lon" or "lon, lat" style pairs; used only to give a precise
// validation error instead of sending out-of-range numbers to geocoding.
function invalidCoordPair(text) {
  const m = String(text || '')
    .trim()
    .match(/^(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)$/)
  if (!m) return null
  const a = parseFloat(m[1])
  const b = parseFloat(m[2])
  const latOk = Math.abs(a) <= 90 && Math.abs(b) <= 180
  const lonOk = Math.abs(b) <= 90 && Math.abs(a) <= 180
  if (latOk || lonOk) return null
  return 'Coordinates out of range — latitude ±90°, longitude ±180°.'
}

export default function LocationSearch({ onGo }) {
  const [text, setText] = useState('')
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const inputRef = useRef(null)
  const listRef = useRef(null)

  const focusResult = useCallback((index) => {
    if (!listRef.current) return
    const items = listRef.current.querySelectorAll('button')
    if (items[index]) {
      items[index].focus()
    }
  }, [])

  async function submit(e) {
    e?.preventDefault()
    setError(null)
    if (!String(text || '').trim()) {
      setError('Type a place name, or paste coordinates as lat, lon.')
      return
    }
    const rangeErr = invalidCoordPair(text)
    if (rangeErr) {
      setError(rangeErr)
      return
    }
    const parsed = parseLocationInput(text)
    if (!parsed) {
      setError('Type a place name, or paste coordinates as lat, lon.')
      return
    }
    if (parsed.type === 'coords') {
      setResults([])
      onGo({ lat: parsed.lat, lon: parsed.lon, bbox: null, name: null })
      return
    }
    setBusy(true)
    try {
      const places = await searchPlaces(parsed.q)
      setResults(places)
      if (!places.length) setError(`No matching place found for “${parsed.q}”.`)
    } catch (err) {
      setError(`Search failed — ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="map-search">
      <form onSubmit={submit} className="map-search-row" role="search" aria-label="Navigate map to location">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setResults([])
              setSelectedIndex(-1)
            }
            if (e.key === 'ArrowDown' && results.length) {
              e.preventDefault()
              const next = Math.min(selectedIndex + 1, results.length - 1)
              setSelectedIndex(next)
              focusResult(next)
            }
            if (e.key === 'ArrowUp' && results.length) {
              e.preventDefault()
              const prev = Math.max(selectedIndex - 1, 0)
              setSelectedIndex(prev)
              focusResult(prev)
            }
          }}
          placeholder="Search place or paste lat, lon…"
          aria-label="Search location"
          aria-busy={busy}
          autoComplete="off"
        />
        <button type="submit" disabled={busy}>{busy ? '···' : 'Go'}</button>
      </form>
      {error && <div className="map-error" role="alert">{error}</div>}
      {!!results.length && (
        <ul className="map-search-results" ref={listRef} role="listbox" aria-label="Search results">
          {results.map((r, i) => (
            <li key={i} role="option" aria-selected={i === selectedIndex}>
              <button
                type="button"
                tabIndex={i === selectedIndex ? 0 : -1}
                onClick={() => { onGo({ lat: r.lat, lon: r.lon, bbox: r.bbox, name: r.name }); setResults([]); setSelectedIndex(-1) }}
              >
                <span>{r.name.length > 90 ? `${r.name.slice(0, 90)}…` : r.name}</span>
                <span className="mono tiny">
                  {Number.isFinite(r.lat) && Number.isFinite(r.lon) ? `${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}` : ''}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
