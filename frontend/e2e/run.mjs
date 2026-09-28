// DepthWizard E2E: production build + live backend, driven through system Chrome.
// Run: npm run test:e2e   (requires backend :8000 and `vite preview` :4173)
// Flow matches the current screen state machine (hero → workflow → upload /
// map → progress → viewer). For map-mock coverage the backend must run with
// DW_MAP_MOCK=true (dev/CI only); otherwise section 4 is skipped, not failed.
import { chromium } from 'playwright-core'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const BASE = process.env.E2E_BASE || 'http://localhost:4173'
const API = process.env.E2E_API || 'http://127.0.0.1:8000'
const HERE = dirname(fileURLToPath(import.meta.url))
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'

let failures = 0
let skips = 0
const check = (name, cond, extra = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`)
  if (!cond) failures++
}
const skip = (name, reason) => {
  skips++
  console.log(`SKIP  ${name} — ${reason}`)
}

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--no-sandbox'],
})
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const pageErrors = []
page.on('pageerror', (e) => pageErrors.push(String(e)))

async function gotoUploadScreen() {
  await page.goto(BASE, { waitUntil: 'networkidle' })
  await page.getByRole('button', { name: /choose your workflow/i }).click()
  await page.waitForSelector('.wf-grid', { timeout: 15000 })
  await page.getByRole('button', { name: /image.*3d reconstruction/i }).click()
  await page.waitForSelector('section.screen input[type="file"]', { timeout: 15000 })
}

async function uploadViaScreen(fixture) {
  await gotoUploadScreen()
  await page.setInputFiles('section.screen input[type="file"]', fixture)
  await page.waitForSelector('.meta-list', { timeout: 15000 })
}

async function beginProcessing() {
  // Non-georeferenced fixtures must explicitly choose Relative Surface
  // before Begin enables (GCP gate).
  const relBtn = page.getByRole('button', { name: /generate relative surface/i })
  if (await relBtn.count()) await relBtn.first().click()
  await page.getByRole('button', { name: /begin processing/i }).click()
}

// ---- 1. entry flow: hero → workflow cards → back ----
await page.goto(BASE, { waitUntil: 'networkidle' })
await page.waitForSelector('.asterra-hero', { timeout: 30000 })
check('entry renders hero', await page.locator('.asterra-hero').count() === 1)
await page.getByRole('button', { name: /choose your workflow/i }).click()
await page.waitForSelector('.wf-grid', { timeout: 15000 })
const wfText = await page.locator('.wf-grid').textContent()
check('workflow offers map path', /geospatial workspace/i.test(wfText || ''))
check('workflow offers upload path', /image.*3d reconstruction/i.test(wfText || ''))
await page.getByRole('button', { name: /entry/i }).click()
await page.waitForSelector('.asterra-hero', { timeout: 15000 })
check('workflow back returns to entry', await page.locator('.asterra-hero').count() === 1)

// ---- 2. real backend upload (PNG): must read REL, never metric ----
await uploadViaScreen(join(HERE, 'fixtures', 'small.png'))
await beginProcessing()
await page.waitForSelector('.viewer-root', { timeout: 300000 })
await page.waitForSelector('.job-chip', { timeout: 30000 })
const relChip = await page.textContent('.job-chip')
check('real PNG job reads REL (rDSM)', /REL \(rDSM\)/.test(relChip || ''), relChip)
check('real PNG job is not labeled metric', !/Metric DSM/.test(relChip || ''))
const relId = (/JOB (\S+)/.exec(relChip || '') || [])[1]
// Topbar must fit without horizontal scrolling: a long backend model path
// once pushed Validation/Export off-screen (Windows backslash bug).
const topbarFits = await page.locator('.viewer-root .topbar').evaluate(
  (el) => el.scrollWidth <= el.clientWidth + 1,
)
check('viewer topbar fits without overflow', topbarFits)
const modelChip = await page.locator('.viewer-root .model-chip').textContent().catch(() => '')
check('model chip shows basename, not a path', !/\\/.test(modelChip || '') && !/weights/.test(modelChip || ''), modelChip)

// ---- 3. nodata GeoTIFF: viewer must survive masked cells ----
await uploadViaScreen(join(HERE, 'fixtures', 'nodata.tif'))
await beginProcessing()
await page.waitForSelector('.viewer-root', { timeout: 300000 })
await page.waitForSelector('.job-chip', { timeout: 30000 })
const nodChip = await page.textContent('.job-chip')
const nodId = (/JOB (\S+)/.exec(nodChip || '') || [])[1]
check('nodata job reaches viewer', !!nodId, nodChip)
// backend grid must actually contain masked cells (proves the mask path ran)
const bin = await (await fetch(`${API}/api/jobs/${nodId}/dsm.bin`)).arrayBuffer()
const vals = new Float32Array(bin)
let masked = 0
for (let i = 0; i < vals.length; i++) if (vals[i] === -9999) masked++
check('nodata cells preserved as -9999 in DSM', masked > 100, `${masked} masked cells`)
// validation failure must NOT fabricate metrics (API level)
const badRef = new Blob(['not a dem'], { type: 'text/plain' })
const fd = new FormData()
fd.append('reference', badRef, 'bad.txt')
const vres = await fetch(`${API}/api/jobs/${relId}/validate`, { method: 'POST', body: fd })
check('garbage reference rejected (no fake metrics)', vres.status === 400, `HTTP ${vres.status}`)

// ---- 4. map mode (mock imagery): AOI → Reconstruct → viewer → Show on map ----
const providers = await (await fetch(`${API}/api/imagery/providers`)).json()
const mockAvailable = (providers?.providers || []).some((p) => p.name === 'mock')
if (!mockAvailable) {
  skip('map mock flow', 'backend not running with DW_MAP_MOCK=true (dev/CI only)')
} else {
  await page.goto(BASE, { waitUntil: 'networkidle' })
  await page.getByRole('button', { name: /choose your workflow/i }).click()
  await page.getByRole('button', { name: /open map/i }).click()
  await page.waitForFunction(() => window.__dwMapReady === true, null, { timeout: 60000 })
  await page.waitForSelector('[data-testid="map-canvas"] canvas', { timeout: 30000 })
  // Search by raw coords (no geocoding dependency), then draw at map center.
  await page.getByLabel('Search location').fill('20.2954, 85.8046')
  await page.getByRole('button', { name: /^go$/i }).click()
  await page.waitForSelector('.map-pin-callout', { timeout: 15000 })
  await page.locator('.pin-x').click()
  const box = await page.locator('[data-testid="map-canvas"]').boundingBox()
  const cx = box.x + box.width / 2
  const cy = box.y + box.height / 2
  await page.mouse.click(cx, cy)
  await page.keyboard.press('+')
  await page.waitForTimeout(300)
  await page.keyboard.press('+')
  await page.waitForTimeout(300)
  await page.getByRole('button', { name: /^draw aoi$/i }).click()
  await page.mouse.move(cx - 35, cy - 28)
  await page.mouse.down()
  await page.mouse.move(cx + 35, cy + 28, { steps: 12 })
  await page.mouse.up()
  await page.waitForSelector('.aoi-selected', { timeout: 15000 })
  const aoiText = await page.locator('.map-side').textContent()
  check('AOI panel shows center/width/area', /Center/.test(aoiText || '') && /Width/.test(aoiText || '') && /Area/.test(aoiText || ''))
  await page.locator('.map-side .map-field select').first().selectOption('mock')
  await page.waitForFunction(
    () => /test pattern/.test(document.querySelector('.map-side')?.textContent || ''),
    null, { timeout: 60000 },
  )
  check('mock imagery scene auto-selected', true)
  await page.getByRole('button', { name: /use this scene/i }).click()
  await page.waitForSelector('.viewer-root', { timeout: 300000 })
  const mapChip = await page.textContent('.job-chip')
  const mapId = (/JOB (\S+)/.exec(mapChip || '') || [])[1]
  check('map job reaches viewer', !!mapId, mapChip)
  const mapStatus = await (await fetch(`${API}/api/jobs/${mapId}`)).json()
  check('result source is map + mock', mapStatus?.result?.source?.type === 'map' && mapStatus?.result?.source?.imagery?.provider === 'mock')
  check('georeference block present', !!mapStatus?.result?.georeference?.bounds && !!mapStatus?.result?.georeference?.origin)
  check('AOI bounds match request', Math.abs((mapStatus?.result?.source?.aoi?.north || 0) - (mapStatus?.result?.georeference?.bounds?.north || 1)) < 1e-4)
  // Analysis panel content renders on demand — open it explicitly, then check.
  await page.getByRole('button', { name: /^analysis$/i }).click()
  await page.waitForSelector('text=/Map · mock/', { timeout: 15000 })
  const analysisText = await page.textContent('body')
  check('analysis panel shows map source', /Map · mock \(test pattern\)/.test(analysisText || ''))
  await page.getByRole('button', { name: /show on map/i }).click()
  await page.waitForSelector('[data-testid="map-canvas"]', { timeout: 30000 })
  const backText = await page.locator('.map-side').textContent()
  check('AOI retained after Show on map', /Width/.test(backText || '') && / m/.test(backText || ''))
}

// ---- 5. hazard simulation: coastal + landslide on an upload job ----
await uploadViaScreen(join(HERE, 'fixtures', 'small.png'))
await beginProcessing()
await page.waitForSelector('.viewer-root', { timeout: 300000 })
await page.getByRole('button', { name: /^hazards$/i }).click()
await page.waitForSelector('.hz-panel', { timeout: 15000 })
// coastal (default +3 m scenario visualization)
await page.locator('.hz-panel select').first().selectOption('coastal_inundation')
await page.getByRole('button', { name: /run simulation/i }).click()
await page.waitForSelector('text=/Inundated area/', { timeout: 120000 })
const hzCoast = await page.locator('.hz-panel').textContent()
check('coastal stats appear', /Max depth/.test(hzCoast || '') && /Potentially affected buildings/.test(hzCoast || ''))
check('coastal disclaimer visible', /not a disaster forecast/i.test(hzCoast || ''))
check('water legend shown', /Water depth/.test(hzCoast || ''))
check('timeline + layers present', await page.getByRole('button', { name: /play mission/i }).count() === 1)
// before/simulated toggle keeps the base scene intact
await page.getByRole('button', { name: /^before$/i }).click()
await page.getByRole('button', { name: /^simulated$/i }).click()
// water-level preset re-simulates (canonical backend result)
await page.getByRole('button', { name: /^\+5m$/ }).click()
await page.waitForFunction(
  () => /\+5\.0 m/.test(document.querySelector('.hz-panel')?.textContent || ''),
  null, { timeout: 120000 },
)
check('water level update re-simulates', true)
// landslide medium
await page.locator('.hz-panel select').first().selectOption('landslide')
await page.getByRole('button', { name: /medium/i }).click()
await page.getByRole('button', { name: /run landslide/i }).click()
await page.waitForSelector('text=/Modeled affected area/', { timeout: 120000 })
const hzSlide = await page.locator('.hz-panel').textContent()
check('landslide stats appear', /Deposition area/.test(hzSlide || '') && /Source slope/.test(hzSlide || ''))
check('landslide disclaimer visible', /not a geotechnical prediction/i.test(hzSlide || ''))
// reset restores the base scene
await page.getByRole('button', { name: /reset simulation/i }).click()
await page.waitForFunction(
  () => !/Inundated area|Modeled affected area/.test(document.querySelector('.hz-panel')?.textContent || ''),
  null, { timeout: 15000 },
)
check('hazard reset clears overlays + stats', true)

check('zero uncaught page errors', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '))
await browser.close()
console.log(failures === 0 ? (skips ? `E2E: ALL PASS (${skips} skipped)` : 'E2E: ALL PASS') : `E2E: ${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
