// Phase 10: basemap sources (display ONLY — never fed to the ML pipeline).
// Public tile endpoints; attribution is mandatory and shown on the map.
export const BASEMAPS = {
  satellite: {
    label: 'Satellite',
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    style: {
      version: 8,
      sources: {
        esri: {
          type: 'raster',
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          tileSize: 256,
          maxzoom: 19,
        },
      },
      layers: [{ id: 'esri', type: 'raster', source: 'esri' }],
    },
  },
  street: {
    label: 'Street',
    attribution: '© OpenStreetMap contributors',
    style: {
      version: 8,
      sources: {
        osm: {
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          maxzoom: 19,
        },
      },
      layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
    },
  },
  terrain: {
    label: 'Terrain',
    attribution: 'Map data © OpenStreetMap contributors, SRTM | style © OpenTopoMap (CC-BY-SA)',
    style: {
      version: 8,
      sources: {
        topo: {
          type: 'raster',
          tiles: ['https://tile.opentopomap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          maxzoom: 17,
        },
      },
      layers: [{ id: 'topo', type: 'raster', source: 'topo' }],
    },
  },
}
