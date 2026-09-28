import { Euler, Vector3 } from 'three'

const KEYMAP = {
  KeyW: 'fwd',
  KeyS: 'back',
  KeyA: 'left',
  KeyD: 'right',
  KeyQ: 'down',
  KeyE: 'up',
}

export class FlyControls {
  constructor(camera, dom) {
    this.camera = camera
    this.dom = dom
    this.enabled = false
    this.keys = {}
    this.yaw = 0
    this.pitch = 0
    this.velocity = new Vector3()
    this.speed = 24
    this.locked = false
    this.onLockChange = null

    this.euler = new Euler(0, 0, 0, 'YXZ')

    document.addEventListener('keydown', this.onKeyDown)
    document.addEventListener('keyup', this.onKeyUp)
    dom.ownerDocument.addEventListener('pointerlockchange', this.handleLock)
    this.onMouseMove = this.onMouseMove.bind(this)
  }

  handleLock = () => {
    this.locked = document.pointerLockElement === this.dom
    if (this.locked) document.addEventListener('mousemove', this.onMouseMove)
    else document.removeEventListener('mousemove', this.onMouseMove)
    this.onLockChange?.(this.locked)
  }

  requestLock() {
    this.dom.requestPointerLock?.()
  }

  releaseLock() {
    document.exitPointerLock?.()
  }

  onMouseMove(e) {
    const s = 0.0022
    this.yaw -= e.movementX * s
    this.pitch -= e.movementY * s
    const lim = Math.PI / 2 - 0.02
    this.pitch = Math.max(-lim, Math.min(lim, this.pitch))
  }

  onKeyDown = (e) => {
    if (!this.enabled) return
    if (KEYMAP[e.code]) {
      this.keys[KEYMAP[e.code]] = true
      e.preventDefault()
    }
    if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') this.sprint = true
  }

  onKeyUp = (e) => {
    if (KEYMAP[e.code]) this.keys[KEYMAP[e.code]] = false
    if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') this.sprint = false
  }

  setFromCamera() {
    this.euler.setFromQuaternion(this.camera.quaternion, 'YXZ')
    this.yaw = this.euler.y
    this.pitch = this.euler.x
  }

  update(dt) {
    if (!this.enabled) return
    const k = this.keys
    const dir = new Vector3()
    const fwd = new Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw))
    const right = new Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw))
    if (k.fwd) dir.add(fwd)
    if (k.back) dir.sub(fwd)
    if (k.right) dir.add(right)
    if (k.left) dir.sub(right)
    if (k.up) dir.y += 1
    if (k.down) dir.y -= 1
    if (dir.lengthSq() > 0) dir.normalize()

    const target = this.sprint ? this.speed * 3 : this.speed
    const desired = dir.multiplyScalar(target)
    this.velocity.lerp(desired, Math.min(1, dt * 6))
    this.camera.position.addScaledVector(this.velocity, dt)

    this.euler.set(this.pitch, this.yaw, 0, 'YXZ')
    this.camera.quaternion.setFromEuler(this.euler)
  }

  getHeading() {
    let deg = (-this.yaw * 180) / Math.PI
    deg = ((deg % 360) + 360) % 360
    return deg
  }

  getSpeed() {
    return this.velocity.length()
  }

  dispose() {
    document.removeEventListener('keydown', this.onKeyDown)
    document.removeEventListener('keyup', this.onKeyUp)
    document.removeEventListener('mousemove', this.onMouseMove)
    this.dom.ownerDocument.removeEventListener('pointerlockchange', this.handleLock)
  }
}
