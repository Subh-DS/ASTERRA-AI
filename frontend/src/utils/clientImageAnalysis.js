/**
 * Client-side image analysis for 3D feature extraction
 * Works entirely in browser - no backend needed
 */

export class ClientImageAnalyzer {
  constructor(options = {}) {
    this.options = {
      downscale: options.downscale || 256,     // max dimension for analysis
      edgeThreshold: options.edgeThreshold || 30,
      buildingBoost: options.buildingBoost || 1.5,
      roadSuppression: options.roadSuppression || 0.3,
      vegetationFactor: options.vegetationFactor || 0.7,
      waterLevel: options.waterLevel || 0.1,
      detailScale: options.detailScale || 1.0,
      ...options
    }
  }

  /**
   * Main entry: analyze image and return height map + feature masks
   */
  async analyze(imageBitmap) {
    const { width, height } = this.getAnalysisSize(imageBitmap)
    const canvas = this.createCanvas(width, height)
    const ctx = canvas.getContext('2d')
    
    ctx.drawImage(imageBitmap, 0, 0, width, height)
    const imageData = ctx.getImageData(0, 0, width, height)
    const { data, width: w, height: h } = imageData

    // Extract channels
    const channels = this.extractChannels(data, w, h)
    
    // Feature extraction
    const luminance = this.computeLuminance(channels)
    const edges = this.detectEdges(luminance, w, h)
    const colorFeatures = this.analyzeColors(channels, w, h)
    const saturation = this.computeSaturation(channels, w, h)
    const hue = this.computeHue(channels, w, h)
    
    // Advanced features for better discrimination
    const texture = this.computeTexture(luminance, w, h)
    const localStats = this.computeLocalStats(luminance, w, h)
    const frequencyFeatures = this.computeFrequencyFeatures(luminance, w, h)

    // Generate feature masks
    const masks = this.generateMasks({
      luminance, edges, saturation, hue, colorFeatures, texture, localStats, frequencyFeatures, w, h
    })

    // Generate height map from features
    const heightMap = this.generateHeightMap({
      luminance, edges, masks, texture, localStats, frequencyFeatures, w, h
    })

    // Generate pyramids for progressive rendering
    const pyramids = this.buildPyramids(heightMap, w, h)

    return {
      width: w,
      height: h,
      heightMap,
      pyramids,
      masks,
      textureCanvas: canvas,
      features: {
        luminance, edges, saturation, hue, texture, localStats, frequencyFeatures, masks
      }
    }
  }

  getAnalysisSize(imageBitmap) {
    const maxDim = this.options.downscale
    let { width, height } = imageBitmap
    if (Math.max(width, height) > maxDim) {
      const scale = maxDim / Math.max(width, height)
      width = Math.round(width * scale)
      height = Math.round(height * scale)
    }
    return { width, height }
  }

  createCanvas(width, height) {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }

  extractChannels(data, w, h) {
    const size = w * h
    const r = new Float32Array(size)
    const g = new Float32Array(size)
    const b = new Float32Array(size)
    
    for (let i = 0, j = 0; i < size; i++, j += 4) {
      r[i] = data[j] / 255
      g[i] = data[j + 1] / 255
      b[i] = data[j + 2] / 255
    }
    return { r, g, b, w, h }
  }

  computeLuminance({ r, g, b }, w, h) {
    const size = w * h
    const lum = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      lum[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]
    }
    return lum
  }

  computeSaturation({ r, g, b }, w, h) {
    const size = w * h
    const sat = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      const max = Math.max(r[i], g[i], b[i])
      const min = Math.min(r[i], g[i], b[i])
      sat[i] = max === 0 ? 0 : (max - min) / max
    }
    return sat
  }

  computeHue({ r, g, b }, w, h) {
    const size = w * h
    const hue = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      const max = Math.max(r[i], g[i], b[i])
      const min = Math.min(r[i], g[i], b[i])
      const delta = max - min
      if (delta === 0) {
        hue[i] = 0
      } else if (max === r[i]) {
        hue[i] = ((g[i] - b[i]) / delta) % 6
      } else if (max === g[i]) {
        hue[i] = (b[i] - r[i]) / delta + 2
      } else {
        hue[i] = (r[i] - g[i]) / delta + 4
      }
      hue[i] /= 6
      if (hue[i] < 0) hue[i] += 1
    }
    return hue
  }

  analyzeColors({ r, g, b }, w, h) {
    const size = w * h
    const features = {
      isVegetation: new Uint8Array(w * h),
      isWater: new Uint8Array(w * h),
      isBuiltUp: new Uint8Array(w * h),
      isRoad: new Uint8Array(w * h),
      isBareSoil: new Uint8Array(w * h),
      isRoof: new Uint8Array(w * h),
      isShadow: new Uint8Array(w * h)
    }

    for (let i = 0; i < w * h; i++) {
      const rr = r[i], gg = g[i], bb = b[i]
      const max = Math.max(rr, gg, bb)
      const min = Math.min(rr, gg, bb)
      const delta = max - min
      
      // Vegetation: high green, low red/blue
      if (gg > rr * 1.15 && gg > bb * 1.15 && gg > 0.25) {
        features.isVegetation[i] = 1
      }
      
      // Water: low saturation, blue dominant, dark
      const sat = delta / (max + 1e-6)
      if (sat < 0.15 && bb > rr && bb > gg && max < 0.5) {
        features.isWater[i] = 1
      }
      
      // Built-up: medium brightness, low saturation, reddish/brownish
      if (sat < 0.35 && max > 0.25 && max < 0.85 && rr > gg && rr > bb && rr > 0.2) {
        features.isBuiltUp[i] = 1
      }
      
      // Roofs: reddish/brown, distinct from roads
      if (sat < 0.4 && max > 0.3 && max < 0.8 && rr > gg * 1.1 && rr > bb * 1.1) {
        features.isRoof[i] = 1
      }
      
      // Roads: dark, low saturation, linear
      if (sat < 0.2 && max < 0.3) {
        features.isRoad[i] = 1
      }
      
      // Bare soil: medium brightness, yellowish/brownish
      if (sat > 0.1 && sat < 0.45 && rr > 0.35 && gg > 0.25 && bb < 0.3) {
        features.isBareSoil[i] = 1
      }
      
      // Shadows: very dark, low saturation
      if (max < 0.15 && sat < 0.2) {
        features.isShadow[i] = 1
      }
    }
    return features
  }

  detectEdges(luminance, w, h) {
    const edges = new Float32Array(w * h)
    const threshold = this.options.edgeThreshold / 255
    
    // Sobel operator
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x
        const gx = 
          -luminance[(y-1)*w + (x-1)] + luminance[(y-1)*w + (x+1)]
          -2*luminance[y*w + (x-1)] + 2*luminance[y*w + (x+1)]
          -luminance[(y+1)*w + (x-1)] + luminance[(y+1)*w + (x+1)]
        const gy = 
          -luminance[(y-1)*w + (x-1)] - 2*luminance[(y-1)*w + x] - luminance[(y-1)*w + (x+1)]
          +luminance[(y+1)*w + (x-1)] + 2*luminance[(y+1)*w + x] + luminance[(y+1)*w + (x+1)]
        
        const mag = Math.sqrt(gx*gx + gy*gy)
        edges[i] = mag > threshold ? Math.min(1, mag * 4) : 0
      }
    }
    return edges
  }

  // Compute local texture (variance in neighborhood)
  computeTexture(luminance, w, h) {
    const texture = new Float32Array(w * h)
    const radius = 2
    for (let y = radius; y < h - radius; y++) {
      for (let x = radius; x < w - radius; x++) {
        let sum = 0, sumSq = 0, count = 0
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const v = luminance[(y + dy) * w + (x + dx)]
            sum += v
            sumSq += v * v
            count++
          }
        }
        const mean = sum / count
        const variance = sumSq / count - mean * mean
        texture[y * w + x] = Math.sqrt(Math.max(0, variance)) * 4
      }
    }
    // Fill borders
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (y < 2 || y >= h - 2 || x < 2 || x >= w - 2) {
          texture[y * w + x] = texture[Math.min(h-3, Math.max(2, y)) * w + Math.min(w-3, Math.max(2, x))]
        }
      }
    }
    return texture
  }

  // Compute local statistics (mean, std, min, max in neighborhood)
  computeLocalStats(luminance, w, h) {
    const radius = 4
    const localMean = new Float32Array(w * h)
    const localStd = new Float32Array(w * h)
    const localMin = new Float32Array(w * h)
    const localMax = new Float32Array(w * h)
    
    for (let y = radius; y < h - radius; y++) {
      for (let x = radius; x < w - radius; x++) {
        let sum = 0, sumSq = 0, count = 0
        let min = 1, max = 0
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const v = luminance[(y + dy) * w + (x + dx)]
            sum += v
            sumSq += v * v
            min = Math.min(min, v)
            max = Math.max(max, v)
            count++
          }
        }
        const mean = sum / count
        localMean[y * w + x] = mean
        localStd[y * w + x] = Math.sqrt(Math.max(0, sumSq / count - mean * mean))
        localMin[y * w + x] = min
        localMax[y * w + x] = max
      }
    }
    // Fill borders with nearest values
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (y < radius || y >= h - radius || x < radius || x >= w - radius) {
          const ny = Math.min(h - radius - 1, Math.max(radius, y))
          const nx = Math.min(w - radius - 1, Math.max(radius, x))
          const idx = ny * w + nx
          localMean[y * w + x] = localMean[idx]
          localStd[y * w + x] = localStd[idx]
          localMin[y * w + x] = localMin[idx]
          localMax[y * w + x] = localMax[idx]
        }
      }
    }
    return { localMean, localStd, localMin, localMax }
  }

  // Frequency domain features (simple DCT-like)
  computeFrequencyFeatures(luminance, w, h) {
    const freqLow = new Float32Array(w * h)
    const freqHigh = new Float32Array(w * h)
    
    // Simple 3x3 high-pass and low-pass
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x
        const center = luminance[i]
        const neighbors = [
          luminance[(y-1)*w + (x-1)], luminance[(y-1)*w + x], luminance[(y-1)*w + (x+1)],
          luminance[y*w + (x-1)], luminance[y*w + (x+1)],
          luminance[(y+1)*w + (x-1)], luminance[(y+1)*w + x], luminance[(y+1)*w + (x+1)]
        ]
        const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / 8
        freqLow[i] = (center + avgNeighbor) / 2
        freqHigh[i] = Math.abs(center - avgNeighbor) * 2
      }
    }
    return { freqLow, freqHigh }
  }

  generateMasks({ luminance, edges, saturation, hue, colorFeatures, texture, localStats, frequencyFeatures, w, h }) {
    const masks = {
      building: new Float32Array(w * h),
      road: new Float32Array(w * h),
      vegetation: new Float32Array(w * h),
      water: new Float32Array(w * h),
      terrain: new Float32Array(w * h),
      roof: new Float32Array(w * h),
      shadow: new Float32Array(w * h)
    }

    for (let i = 0; i < w * h; i++) {
      const lum = luminance[i]
      const edg = edges[i]
      const sat = saturation[i]
      const tex = texture[i] || 0
      const freqHigh = frequencyFeatures.freqHigh[i] || 0
      const localStd = localStats.localStd[i] || 0
      
      // Building mask: edges + built-up color + roof color + medium-high luminance + high local variance
      masks.building[i] = Math.min(1, 
        colorFeatures.isBuiltUp[i] * 0.35 + 
        colorFeatures.isRoof[i] * 0.25 +
        (edg * (luminance[i] > 0.25 ? 1 : 0.4)) * 0.25 +
        (localStd > 0.05 ? 0.15 : 0) +
        (freqHigh > 0.1 ? 0.1 : 0)
      )

      // Roof mask: distinct from roads
      masks.roof[i] = Math.min(1,
        colorFeatures.isRoof[i] * 0.6 +
        (edg * (luminance[i] > 0.3 ? 1 : 0.3)) * 0.2
      )

      // Road mask: road color + strong edges + dark + low texture
      masks.road[i] = Math.min(1,
        colorFeatures.isRoad[i] * 0.4 +
        (edg * (1 - luminance[i]) * (1 - texture[i] || 0)) * 0.6
      )

      // Vegetation mask: vegetation color + medium texture
      masks.vegetation[i] = Math.min(1,
        colorFeatures.isVegetation[i] * 0.8 +
        (texture[i] * 0.2)
      )

      // Water mask: water color + very low texture + low saturation
      masks.water[i] = Math.min(1,
        colorFeatures.isWater[i] * 0.9 +
        ((1 - texture[i]) * 0.1)
      )

      // Shadow mask
      masks.shadow[i] = colorFeatures.isShadow[i] ? 0.8 : 0

      // Terrain (catch-all for ground)
      masks.terrain[i] = 1 - Math.max(
        masks.building[i], masks.road[i], masks.vegetation[i], masks.water[i], masks.shadow[i]
      )
    }

    // Smooth masks slightly
    return this.smoothMasks(masks, 2, w, h)
  }

  smoothMasks(masks, radius = 2, w, h) {
    const smoothed = {}
    for (const key of Object.keys(masks)) {
      const input = masks[key]
      let output = new Float32Array(w * h)
      let current = input
      
      for (let pass = 0; pass < 2; pass++) {
        output = new Float32Array(w * h)
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            let sum = 0, count = 0
            for (let dy = -radius; dy <= radius; dy++) {
              for (let dx = -radius; dx <= radius; dx++) {
                const ny = y + dy, nx = x + dx
                if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                  sum += current[ny * w + nx]
                  count++
                }
              }
            }
            output[y * w + x] = sum / count
          }
        }
        current = output
      }
      smoothed[key] = current
    }
    return smoothed
  }

  generateHeightMap({ luminance, edges, masks, texture, localStats, frequencyFeatures, w, h }) {
    const heightMap = new Float32Array(w * h)
    const opts = this.options

    for (let i = 0; i < w * h; i++) {
      let height = 0
      const lum = isNaN(luminance[i]) ? 0 : luminance[i]
      const edg = isNaN(edges[i]) ? 0 : edges[i]
      const bld = isNaN(masks.building[i]) ? 0 : masks.building[i]
      const roof = isNaN(masks.roof[i]) ? 0 : masks.roof[i]
      const rd = isNaN(masks.road[i]) ? 0 : masks.road[i]
      const veg = isNaN(masks.vegetation[i]) ? 0 : masks.vegetation[i]
      const wat = isNaN(masks.water[i]) ? 0 : masks.water[i]
      const ter = isNaN(masks.terrain[i]) ? 0 : masks.terrain[i]
      const tex = isNaN(texture[i]) ? 0 : texture[i]
      const freqHigh = isNaN(frequencyFeatures.freqHigh[i]) ? 0 : frequencyFeatures.freqHigh[i]
      const localStd = isNaN(localStats.localStd[i]) ? 0 : localStats.localStd[i]
      const localMean = isNaN(localStats.localMean[i]) ? 0 : localStats.localMean[i]
      const shad = isNaN(masks.shadow[i]) ? 0 : masks.shadow[i]

      // Deterministic noise based on position and image content for variation
      const noiseSeed = (i * 1234567) % 1000000 / 1000000
      const noise = Math.sin(noiseSeed * 1000) * 0.5 + 0.5

      // Water = baseline (0)
      if (wat > 0.5) {
        height = 0
      }
      // Roads: slight depression, flat
      else if (rd > 0.5) {
        height = 0.3
      }
      // Water bodies (partial)
      else if (wat > 0.3) {
        height = 0.5
      }
      // Shadows: depressions
      else if (shad > 0.5) {
        height = 1 + lum * 5
      }
      // Vegetation: height from luminance + texture variation
      else if (veg > 0.5) {
        height = 3 + lum * 20 + tex * 8 + noise * 3
      }
      // Buildings: tall structures with roof detail
      else if (bld > 0.3 || roof > 0.4) {
        const baseHeight = 8 + lum * 35
        const roofDetail = roof * 5
        const edgeDetail = edg * 8
        const textureVar = tex * 10
        height = baseHeight + roofDetail + edgeDetail + textureVar + noise * 2
      }
      // Roads: flat, slight texture
      else if (rd > 0.3) {
        height = 0.5 + tex * 2
      }
      // Terrain: base elevation from luminance + local variation
      else {
        const base = 1 + lum * 25
        const variation = localStd * 20 + (isNaN(frequencyFeatures.freqHigh[i]) ? 0 : frequencyFeatures.freqHigh[i]) * 15 + noise * 2
        height = base + variation
      }

      // Add edge detail everywhere (building edges, terrain breaks)
      height += edg * 4

      // Deterministic noise based on image content for unique variation per image
      const contentNoise = Math.sin(i * 0.01 + lum * 100) * 0.5 + 0.5
      height += contentNoise * 0.5

      // Clamp and store - ensure no NaN
      heightMap[i] = isNaN(height) || !isFinite(height) ? 0 : Math.max(0, height)
    }

    // Apply slight smoothing for natural look
    return this.smoothHeightMap(heightMap, w, h)
  }

  smoothHeightMap(heightMap, w, h) {
    const smoothed = new Float32Array(w * h)
    const radius = 1
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0, count = 0
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const ny = y + dy, nx = x + dx
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
              const v = heightMap[ny * w + nx]
              if (!isNaN(v) && isFinite(v)) {
                sum += v
                count++
              }
            }
          }
        }
        smoothed[y * w + x] = count > 0 ? sum / count : 0
      }
    }
    return smoothed
  }

  buildPyramids(heightMap, w, h) {
    const levels = [64, 128, 256].filter(s => s <= Math.max(w, h))
    return levels.map(size => {
      const scale = size / Math.max(w, h)
      const nw = Math.round(w * scale)
      const nh = Math.round(h * scale)
      const data = new Float32Array(nw * nh)
      for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) {
          const srcX = Math.min(w - 1, Math.floor(x / scale))
          const srcY = Math.min(h - 1, Math.floor(y / scale))
          const v = heightMap[srcY * w + srcX]
          data[y * nw + x] = isNaN(v) || !isFinite(v) ? 0 : v
        }
      }
      return { data, w: nw, h: nh }
    })
  }
}

/**
 * Quick analysis for preview - returns simplified result
 */
export async function quickAnalyze(imageBitmap) {
  const analyzer = new ClientImageAnalyzer({ downscale: 128 })
  return analyzer.analyze(imageBitmap)
}

export default ClientImageAnalyzer