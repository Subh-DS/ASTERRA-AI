import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
// MapLibre v6 ESM ships the GeoJSON/vector-tile worker as a separate chunk:
// without an explicit worker URL it fails its first import and every GeoJSON
// overlay (AOI rectangle, search pin) silently never renders while the raster
// basemap looks fine. `?worker&url` (not plain `?url`) bundles it
// self-contained with its shared chunk (per upstream Vite guide).
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'

maplibregl.setWorkerUrl(workerUrl)
import { useApp } from '../../store/useAppStore'
import { aoiMetrics, formatArea } from '../../utils/aoi'
import { BASEMAPS } from './basemaps'
import LocationSearch from './LocationSearch'
import AOIInfoPanel from './AOIInfoPanel'
import ImagerySelector from './ImagerySelector'
import ReconstructionButton from './ReconstructionButton'

const AOI_SRC = 'dw-aoi'
const PIN_SRC = 'dw-search-pin'
const r6 = (v) => Math.round(v * 1e6) / 1e6

function aoiFeature(aoi) {
  if (!aoi) return { type: 'FeatureCollection', features: [] }
  const { north, south, east, west } = aoi
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]] },
    }],
  }
}

function paintAoiScene(map, locked) {
  try {
    map.setPaintProperty(`${AOI_SRC}-line`, 'line-color', locked ? '#4fd9c4' : '#38bdf8')
    map.setPaintProperty(`${AOI_SRC}-line`, 'line-width', locked ? 3 : 2)
  } catch { /* style mid-swap */ }
}

function pinFeature(pin) {
  if (!pin) return { type: 'FeatureCollection', features: [] }
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {},
      geometry: { type: 'Point', coordinates: [pin.lon, pin.lat] },
    }],
  }
}

function addPinLayers(map) {
  if (map.getSource(PIN_SRC)) return
  map.addSource(PIN_SRC, { type: 'geojson', data: pinFeature(null) })
  map.addLayer({
    id: `${PIN_SRC}-halo`, type: 'circle', source: PIN_SRC,
    paint: { 'circle-radius': 14, 'circle-color': '#4fd9c4', 'circle-opacity': 0.22, 'circle-stroke-width': 0 },
  })
  map.addLayer({
    id: `${PIN_SRC}-dot`, type: 'circle', source: PIN_SRC,
    paint: { 'circle-radius': 5, 'circle-color': '#4fd9c4', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 },
  })
}

function addAoiLayers(map) {
  if (map.getSource(AOI_SRC)) return
  map.addSource(AOI_SRC, { type: 'geojson', data: aoiFeature(null) })
  map.addLayer({ id: `${AOI_SRC}-fill`, type: 'fill', source: AOI_SRC, paint: { 'fill-color': '#38bdf8', 'fill-opacity': 0.18 } })
  map.addLayer({ id: `${AOI_SRC}-line`, type: 'line', source: AOI_SRC, paint: { 'line-color': '#38bdf8', 'line-width': 2 } })
}

export default function MapScreen() {
  const wrapRef = useRef(null)
  const mapRef = useRef(null)
  const drawRef = useRef(null)
  const mapState = useApp((s) => s.map)
  const setMap = useApp((s) => s.setMap)
  const setMapAoi = useApp((s) => s.setMapAoi)
  const setScreen = useApp((s) => s.setScreen)
  const mapJob = useApp((s) => s.mapJob)
  const [drawArmed, setDrawArmed] = useState(false)
  const [ready, setReady] = useState(false)
  const [mapError, setMapError] = useState(null)
  const [pin, setPin] = useState(null)
  const leaving = useApp((s) => s.mapLeaving)
  const imagerySel = useApp((s) => s.map.imagery)
  const aoiRef = useRef(mapState.aoi)
  aoiRef.current = mapState.aoi
  const pinRef = useRef(null)
  pinRef.current = pin
  const leavingRef = useRef(false)
  leavingRef.current = leaving

  // Init once.
  useEffect(() => {
    if (mapRef.current || !wrapRef.current) return
    const params = new URLSearchParams(window.location.search)
    const lat = parseFloat(params.get('lat'))
    const lon = parseFloat(params.get('lon'))
    const zoom = parseFloat(params.get('zoom'))
    const st = useApp.getState().map
    let map
    try {
      map = new maplibregl.Map({
        container: wrapRef.current,
        style: BASEMAPS[st.basemap]?.style || BASEMAPS.satellite.style,
        center: (!isNaN(lon) && !isNaN(lat)) ? [lon, lat] : st.center,
        zoom: !isNaN(zoom) ? zoom : st.zoom,
        attributionControl: { compact: true },
      })
    } catch (err) {
      setMapError(`Map failed to start — ${err.message}`)
      return
    }
    mapRef.current = map
    window.__dwMap = map
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    // Boot is idempotent and listens to both `load` and `idle`: in rare cases
    // (e.g. an instantly-cached style) `load` can fire before a listener is
    // attached, which previously left the map running but the UI stuck.
    let booted = false
    const boot = () => {
      if (booted) return
      booted = true
      try {
        addAoiLayers(map)
        addPinLayers(map)
        const src = map.getSource(AOI_SRC)
        src?.setData(aoiFeature(aoiRef.current))
        paintAoiScene(map, !!(aoiRef.current && useApp.getState().map.imagery?.selectedId))
        map.getSource(PIN_SRC)?.setData(pinFeature(pinRef.current))
      } catch {
        // Overlay setup must never brick map readiness; the sync effects
        // below re-apply layer data once the style settles.
      }
      setReady(true)
      window.__dwMapReady = true
    }
    map.on('load', boot)
    map.on('idle', boot)
    map.on('moveend', () => {
      const c = map.getCenter()
      useApp.getState().setMap({ center: [r6(c.lng), r6(c.lat)], zoom: Math.round(map.getZoom() * 100) / 100 })
    })
    map.on('error', (e) => {
      // Tile/network failures must not kill the workflow; the AOI + backend path stays usable.
      if (!mapRef.current) setMapError(`Map error — ${e?.error?.message || 'failed to load'}`)
    })
    return () => {
      window.__dwMapReady = false
      window.__dwMap = null
      try { map.remove() } catch { /* already gone */ }
      mapRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Basemap switching (re-add AOI + pin overlays after style swap).
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.setStyle(BASEMAPS[mapState.basemap]?.style || BASEMAPS.satellite.style)
    map.once('styledata', () => {
      addAoiLayers(map)
      map.getSource(AOI_SRC)?.setData(aoiFeature(aoiRef.current))
      paintAoiScene(map, !!(aoiRef.current && useApp.getState().map.imagery?.selectedId))
      addPinLayers(map)
      map.getSource(PIN_SRC)?.setData(pinFeature(pinRef.current))
    })
  }, [mapState.basemap, ready])

  // AOI overlay sync. The rectangle turns teal once a reconstruction scene
  // is locked in, visually connecting the selected scene to the map area.
  const hasScene = !!useApp((s) => s.map.imagery?.selectedId)
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.getSource(AOI_SRC)?.setData(aoiFeature(mapState.aoi))
    paintAoiScene(map, hasScene && !!mapState.aoi)
  }, [mapState.aoi, hasScene, ready])

  // Search-pin overlay sync.
  useEffect(() => {
    mapRef.current?.getSource(PIN_SRC)?.setData(pinFeature(pin))
  }, [pin])

  // Departure cinematic: lock the AOI boundary, dive toward the region,
  // and dim the satellite view into a technical reconstruction state while
  // the backend request runs in parallel (see ReconstructionButton).
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !leaving) return
    const aoi = aoiRef.current
    if (!aoi) return
    try {
      map.dragPan.disable()
      map.scrollZoom.disable()
      map.setPaintProperty(`${AOI_SRC}-line`, 'line-color', '#ffffff')
      map.setPaintProperty(`${AOI_SRC}-line`, 'line-width', 4)
      map.setPaintProperty(`${AOI_SRC}-fill`, 'fill-opacity', 0.08)
      map.fitBounds(
        [[aoi.west, aoi.south], [aoi.east, aoi.north]],
        {
          padding: 50,
          duration: useApp.getState().reducedMotion ? 100 : 1600,
          essential: true,
          maxZoom: 17,
        },
      )
    } catch { /* decorative; navigation still commits */ }
  }, [ready, leaving])

  // Leaving is a transient map-screen state: always reset on unmount so a
  // return to the map starts clean.
  useEffect(() => () => {
    useApp.getState().setMapLeaving(false)
  }, [])

  // AOI emphasis: a slow, subtle pulse keeps the selected rectangle as the
  // visual focus without shouting. Disabled under reduced motion. Paused
  // during the departure lock so the two never fight over the boundary.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !mapState.aoi || leaving) return
    if (useApp.getState().reducedMotion) return
    let hi = false
    const id = setInterval(() => {
      hi = !hi
      try {
        map.setPaintProperty(`${AOI_SRC}-fill`, 'fill-opacity', hi ? 0.26 : 0.15)
      } catch { /* style swapped mid-tick */ }
    }, 700)
    return () => clearInterval(id)
  }, [ready, mapState.aoi, leaving])

  // Phase 11: hazard overlay — canonical backend polygons, map-CRS only.
  const hazardResult = useApp((s) => s.hazard.result)
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const HZ = 'dw-hazard'
    const kill = () => {
      if (map.getLayer(`${HZ}-fill`)) map.removeLayer(`${HZ}-fill`)
      if (map.getLayer(`${HZ}-line`)) map.removeLayer(`${HZ}-line`)
      if (map.getSource(HZ)) map.removeSource(HZ)
    }
    kill()
    const feats = hazardResult?.overlays?.polygons
    const crs = hazardResult?.overlay_crs || ''
    const geographic = /4326|WGS.?84/i.test(crs)
    if (!feats?.length || !geographic) return
    const coastal = hazardResult.simulation_type === 'coastal_inundation'
    const color = coastal ? '#3b82f6' : '#f59e0b'
    const add = () => {
      map.addSource(HZ, { type: 'geojson', data: { type: 'FeatureCollection', features: feats } })
      map.addLayer({ id: `${HZ}-fill`, type: 'fill', source: HZ, paint: { 'fill-color': color, 'fill-opacity': 0.35 } })
      map.addLayer({ id: `${HZ}-line`, type: 'line', source: HZ, paint: { 'line-color': color, 'line-width': 1.5 } })
    }
    if (map.isStyleLoaded()) add()
    else map.once('styledata', add)
    return kill
  }, [hazardResult, ready])

  const armedRef = useRef(false)
  armedRef.current = drawArmed

  // Rectangle draw interaction.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const canvas = map.getCanvas()
    const startDraw = (e) => {
      if (leavingRef.current || !armedRef.current || drawRef.current?.active) return
      drawRef.current = { active: false, start: null }
      map.dragPan.disable()
      map.boxZoom.disable()
      const r = canvas.getBoundingClientRect()
      drawRef.current.start = [e.clientX - r.left, e.clientY - r.top]
      drawRef.current.active = true
    }
    const onMove = (e) => {
      const d = drawRef.current
      if (!d?.active) return
      const r = canvas.getBoundingClientRect()
      const cur = [e.clientX - r.left, e.clientY - r.top]
      const p0 = map.unproject(d.start)
      const p1 = map.unproject(cur)
      map.getSource(AOI_SRC)?.setData(aoiFeature({
        north: Math.max(p0.lat, p1.lat), south: Math.min(p0.lat, p1.lat),
        east: Math.max(p0.lng, p1.lng), west: Math.min(p0.lng, p1.lng),
      }))
    }
    const endDraw = (e) => {
      const d = drawRef.current
      if (!d?.active) return
      d.active = false
      map.dragPan.enable()
      map.boxZoom.enable()
      const r = canvas.getBoundingClientRect()
      const cur = [e.clientX - r.left, e.clientY - r.top]
      const p0 = map.unproject(d.start)
      const p1 = map.unproject(cur)
      const px = Math.abs(cur[0] - d.start[0])
      const py = Math.abs(cur[1] - d.start[1])
      if (px < 8 || py < 8) {
        map.getSource(AOI_SRC)?.setData(aoiFeature(aoiRef.current))
        return
      }
      setMapAoi({
        north: r6(Math.max(p0.lat, p1.lat)), south: r6(Math.min(p0.lat, p1.lat)),
        east: r6(Math.max(p0.lng, p1.lng)), west: r6(Math.min(p0.lng, p1.lng)),
      })
      setDrawArmed(false)
      // Reframe gently so the new rectangle becomes the visual focus.
      try {
        map.fitBounds(
          [[Math.min(p0.lng, p1.lng), Math.min(p0.lat, p1.lat)], [Math.max(p0.lng, p1.lng), Math.max(p0.lat, p1.lat)]],
          { padding: 90, duration: 1200, essential: true, maxZoom: 16 },
        )
      } catch { /* reframe is decorative; the AOI itself is already stored */ }
    }
    canvas.addEventListener('mousedown', startDraw)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', endDraw)
    return () => {
      canvas.removeEventListener('mousedown', startDraw)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', endDraw)
    }
  }, [ready, drawArmed, setMapAoi])

  function goTo({ lat, lon, bbox, name }) {
    const map = mapRef.current
    if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) return
    // Smooth cinematic flight everywhere — never an abrupt jump.
    if (bbox && isFinite(bbox.south)) {
      map.fitBounds([[bbox.west, bbox.south], [bbox.east, bbox.north]], {
        padding: 60,
        maxZoom: 15,
        duration: 2200,
        essential: true,
      })
    } else {
      map.flyTo({
        center: [lon, lat],
        zoom: Math.max(map.getZoom(), 13),
        duration: 2200,
        curve: 1.4,
        essential: true,
      })
    }
    setPin({
      lat,
      lon,
      label: name && name.length > 80 ? `${name.slice(0, 80)}…` : name || `${lat.toFixed(4)}, ${lon.toFixed(4)}`,
    })
  }

  return (
    <section className="screen map-screen">
      <header className="map-topbar">
        <button type="button" className="map-ghost" onClick={() => setScreen('hero')}>← ASTERRA</button>
        <LocationSearch onGo={goTo} />
        <div className="map-basemaps" role="group" aria-label="Basemap">
          {Object.entries(BASEMAPS).map(([key, b]) => (
            <button
              key={key}
              type="button"
              className={mapState.basemap === key ? 'on' : ''}
              onClick={() => setMap({ basemap: key })}
            >{b.label}</button>
          ))}
        </div>
      </header>
      {mapError && <div className="map-error map-banner">{mapError}</div>}
      <div className="map-body">
        <div className="map-canvas-wrap">
          <div ref={wrapRef} className="map-canvas" data-testid="map-canvas" />
          <div className="map-tools">
            <button
              type="button"
              className={drawArmed ? 'on' : ''}
              onClick={() => setDrawArmed((v) => !v)}
            >{drawArmed ? 'Drawing… click-drag on map' : 'Draw AOI'}</button>
          </div>
          <div className="map-attr">{BASEMAPS[mapState.basemap]?.attribution}</div>
          {leaving && mapState.aoi && (
            <DepartureOverlay aoi={mapState.aoi} imagery={imagerySel} />
          )}
          {pin && (
            <div className="map-pin-callout" role="status" aria-label={`Located ${pin.label}`}>
              <span className="asterra-dot" aria-hidden="true" />
              <div className="pin-text">
                <div className="pin-coords mono">{pin.lat.toFixed(4)}, {pin.lon.toFixed(4)}</div>
                <div className="pin-label" title={pin.label}>{pin.label}</div>
              </div>
              <button
                type="button"
                className="pin-cta"
                onClick={() => setDrawArmed(true)}
                title="Arm AOI drawing around this location"
              >
                Define AOI here →
              </button>
              <button
                type="button"
                className="pin-x"
                onClick={() => setPin(null)}
                aria-label="Dismiss located pin"
              >
                ✕
              </button>
            </div>
          )}
        </div>
        <aside className={`map-side${mapState.aoi ? ' aoi-on' : ''}`}>
          <AOIInfoPanel />
          <ImagerySelector />
          <ReconstructionButton />
          {mapJob && (
            <div className="map-panel-block">
              <h3>Recent map reconstruction</h3>
              <dl className="map-facts">
                <div><dt>Job</dt><dd>{mapJob.jobId}</dd></div>
                <div><dt>Status</dt><dd>{mapJob.reused ? 'completed (existing)' : mapJob.status}</dd></div>
              </dl>
              <button type="button" className="map-ghost" onClick={() => setScreen('viewer')}>Open 3D</button>
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}

function DepartureOverlay({ aoi, imagery }) {
  const m = aoiMetrics(aoi.north, aoi.south, aoi.east, aoi.west)
  const sel = imagery?.items?.find((i) => i.item_id === imagery.selectedId) || null
  return (
    <div className="map-departure" role="status" aria-live="polite" aria-label="Locking selected area for reconstruction">
      <div className="depart-inner">
        <div className="depart-kicker">AOI locked</div>
        <div className="depart-meta mono">
          {m.centerLat.toFixed(4)}, {m.centerLon.toFixed(4)} · {formatArea(m.areaKm2)}
          {sel ? ` · ${imagery.provider || 'auto'} ${sel.resolution_m ?? imagery.resolution ?? '?'} m` : ''}
        </div>
        <div className="depart-state">Reconstructing — depth → DSM → mesh…</div>
      </div>
    </div>
  )
}
