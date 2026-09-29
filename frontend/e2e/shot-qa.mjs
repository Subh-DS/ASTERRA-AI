import { chromium } from 'playwright-core'

const BASE = process.env.E2E_BASE || 'http://localhost:4173'
const b = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', args: ['--no-sandbox'] })
const results = []
const check = (name, cond, extra = '') => {
  results.push(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`)
}

for (const vp of [{ w: 1440, h: 900 }, { w: 1280, h: 800 }]) {
  const pg = await b.newPage({ viewport: { width: vp.w, height: vp.h } })
  const errs = []
  pg.on('pageerror', (e) => errs.push(String(e)))
  const tag = `${vp.w}x${vp.h}`

  // ENTRY -> WORKFLOW
  await pg.goto(BASE, { waitUntil: 'networkidle' })
  await pg.waitForTimeout(800)
  check(`${tag} entry renders hero`, await pg.locator('.asterra-hero').count() === 1)
  check(`${tag} no h-overflow on entry`, await pg.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
  await pg.getByRole('button', { name: /choose your workflow/i }).click()
  await pg.waitForSelector('.wf-grid')

  // WORKFLOW -> MAP
  await pg.getByRole('button', { name: /open map/i }).click()
  await pg.waitForFunction(() => window.__dwMapReady === true, null, { timeout: 60000 })
  check(`${tag} slim internal bar`, await pg.locator('.asterra-topbar.internal').count() === 1)
  check(`${tag} mission bar present`, await pg.locator('.mission-bar').count() === 1)

  // SEARCH (coords path, no network dependency)
  await pg.getByLabel('Search location').fill('20.2961, 85.8245')
  await pg.getByRole('button', { name: /^go$/i }).click()
  await pg.waitForSelector('.map-pin-callout', { timeout: 15000 })
  check(`${tag} search pin + callout`, true)
  await pg.locator('.pin-x').click()

  // AOI
  await pg.waitForSelector('[data-testid="map-canvas"] canvas', { timeout: 30000 })
  const box = await pg.locator('[data-testid="map-canvas"]').boundingBox()
  const cx = box.x + box.width / 2
  const cy = box.y + box.height / 2
  await pg.mouse.click(cx, cy)
  await pg.keyboard.press('+')
  await pg.waitForTimeout(300)
  await pg.keyboard.press('+')
  await pg.waitForTimeout(300)
  await pg.getByRole('button', { name: /^draw aoi$/i }).click()
  await pg.mouse.move(cx - 35, cy - 28)
  await pg.mouse.down()
  await pg.mouse.move(cx + 35, cy + 28, { steps: 12 })
  await pg.mouse.up()
  await pg.waitForSelector('.aoi-selected', { timeout: 15000 })
  // The rectangle must actually paint on the map (a missing worker URL once
  // left all GeoJSON overlays silently invisible while the panel looked fine).
  await pg.waitForFunction(() => {
    const m = window.__dwMap
    if (!m) return false
    try { return m.queryRenderedFeatures({ layers: ['dw-aoi-fill'] }).length > 0 } catch { return false }
  }, null, { timeout: 30000 })
  check(`${tag} AOI rectangle renders on map`, true)
  await pg.waitForSelector('.scene-list', { timeout: 180000 })
  check(`${tag} AOI + scenes + GSD`, (await pg.locator('.map-side').textContent()).includes('Source GSD'))

  // RECONSTRUCT -> PIPELINE -> TWIN
  await pg.getByRole('button', { name: /use this scene/i }).click()
  await pg.waitForSelector('.map-departure', { timeout: 15000 })
  check(`${tag} departure cinematic`, true)
  await pg.waitForSelector('.viewer-root.twin-born', { timeout: 420000 })
  await pg.waitForTimeout(4500)
  check(`${tag} twin birth completes`, await pg.locator('.viewer-root.twin-born').count() === 1)
  check(`${tag} camera dock, no legacy clusters`,
    (await pg.locator('.cameradock').count()) === 1 &&
    (await pg.locator('.modeswitch').count()) === 0 &&
    (await pg.locator('.tourbar').count()) === 0)
  check(`${tag} meta HUD real values`, ((await pg.locator('.meta-hud').textContent()) || '').includes('EPSG'))
  check(`${tag} no h-overflow in twin`, await pg.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
  const structureTab = pg.getByRole('tab', { name: /structures/i })
  check(`${tag} structures scene available`, await structureTab.count() === 1 && await structureTab.isEnabled())
  const shelf = pg.locator('.layerbox')
  check(`${tag} metric geometry shelf visible`, await shelf.count() === 1)
  if (await shelf.count()) {
    const shelfLayout = await pg.evaluate(() => {
      const panel = document.querySelector('.layerbox')
      const rail = document.querySelector('.leftrail')
      if (!panel || !rail) return false
      const p = panel.getBoundingClientRect()
      const r = rail.getBoundingClientRect()
      return p.left >= r.right + 4 && p.top < window.innerHeight
    })
    check(`${tag} metric geometry clears left rail`, shelfLayout)
    const toggle = pg.getByRole('button', { name: /metric geometry/i })
    check(`${tag} metric geometry toggle exposed`, await toggle.count() === 1 && await toggle.getAttribute('aria-expanded') === 'true')
    await toggle.click()
    check(`${tag} metric geometry closes`, await toggle.getAttribute('aria-expanded') === 'false' && await shelf.locator('.layer-row').count() === 0)
    await toggle.click()
    check(`${tag} metric geometry reopens`, await toggle.getAttribute('aria-expanded') === 'true' && await shelf.locator('.layer-row').count() > 0)
  }
  check(`${tag} segmentation diagnostics absent`, !/segmentation diagnostics|segformer model/i.test((await pg.locator('.viewer-root').textContent()) || ''))

  // ANALYSIS + HAZARDS + STORM + LIGHT THEME
  await pg.getByRole('button', { name: /^analysis$/i }).click()
  await pg.waitForTimeout(500)
  check(`${tag} analysis opens`, await pg.locator('.panel-host.open').count() === 1)
  await pg.keyboard.press('Escape')
  await pg.getByRole('button', { name: /^hazards$/i }).click()
  await pg.waitForSelector('.hz-panel', { timeout: 15000 })
  await pg.getByLabel('Toggle storm visualization').check()
  await pg.waitForTimeout(1200)
  check(`${tag} storm engages`, await pg.getByLabel('Toggle storm visualization').isChecked())
  // light theme across the twin
  await pg.locator('.viewer-root .topbar').getByRole('button', { name: /theme/i }).first().click()
  await pg.waitForTimeout(1000)
  check(`${tag} light theme applies`, await pg.evaluate(() => document.documentElement.dataset.theme) === 'light')
  await pg.screenshot({ path: `C:/Users/subha/AppData/Local/Temp/opencode/qa-${vp.w}.png` })
  // back to dark for consistency
  await pg.locator('.viewer-root .topbar').getByRole('button', { name: /theme/i }).first().click()
  await pg.waitForTimeout(500)
  check(`${tag} zero page errors`, errs.length === 0, errs.slice(0, 2).join(' | '))
  await pg.close()
}
await b.close()
console.log(results.join('\n'))
console.log(results.some((r) => r.startsWith('FAIL')) ? 'QA: FAILURES PRESENT' : 'QA: ALL PASS')
