import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'
import { getAssistantReply } from '../api/assistant'

const SUGGESTIONS = {
  hero: ['What is a DSM?', 'Explain the pipeline', 'What is the difference between DSM and DTM?'],
  upload: ['What are ground control points?', 'What is a GeoTIFF?'],
  progress: ['Which stage is running?', 'How does calibration work?'],
  viewer: ['Set exaggeration to 2.5', 'Switch to fly mode', 'Toggle isolines', 'What is the RMSE?'],
}

function stamp() {
  const d = new Date()
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':')
}

function sanitizeHtml(text) {
  const div = document.createElement('div')
  div.textContent = String(text ?? '')
  return div.innerHTML
}

function formatMessage(text) {
  const escaped = sanitizeHtml(text)
  return escaped
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
}

export default function AssistantDock() {
  const open = useApp((s) => s.assistantOpen)
  const setOpen = useApp((s) => s.setAssistantOpen)
  const transcript = useApp((s) => s.transcript)
  const pushMessage = useApp((s) => s.pushMessage)
  const updateLastMessage = useApp((s) => s.updateLastMessage)
  const clearTranscript = useApp((s) => s.clearTranscript)
  const backendUp = useApp((s) => s.backendUp)
  const screen = useApp((s) => s.screen)
  const reduced = useApp((s) => s.reducedMotion)

  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const listRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 280)
  }, [open])

  useEffect(() => {
    if (!open) return
    const fn = (e) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', fn)
    return () => window.removeEventListener('keydown', fn)
  }, [open])

  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [transcript, busy])

  const send = async (raw) => {
    const text = (raw ?? draft).trim()
    if (!text || busy) return
    setDraft('')
    pushMessage({ role: 'you', text, t: stamp() })
    setBusy(true)
    let reply
    try {
      reply = await getAssistantReply(text)
    } catch {
      reply = 'That query hit a snag locally — try rephrasing.'
    }
    pushMessage({ role: 'dw', text: '', t: stamp() })
    if (reduced) {
      updateLastMessage(reply)
    } else {
      const step = Math.max(2, Math.round(reply.length / 90))
      for (let i = step; i <= reply.length; i += step) {
        updateLastMessage(reply.slice(0, i))
        await new Promise((r) => setTimeout(r, 12))
      }
      updateLastMessage(reply)
    }
    setBusy(false)
  }

  const suggestions = SUGGESTIONS[screen] || []
  const showChips = suggestions.length > 0 && transcript.length < 3

  return (
    <>
      <button
        className={`assistant-handle${open ? ' hidden' : ''}`}
        onClick={() => setOpen(true)}
        aria-label="Open field assistant"
      >
        Ask ◂
      </button>

      <aside className={`assistant-dock${open ? ' open' : ''}`} aria-hidden={!open} aria-label="Field assistant">
        <div className="panel-inner" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div className="panel-head">
            <h2 style={{ fontSize: 'var(--t-20)' }}>Field Assistant</h2>
            <span className={`assistant-status mono${backendUp ? '' : ' local'}`}>
              {backendUp === null ? '…' : backendUp ? 'GEOSPATIAL AI' : 'LOCAL'}
            </span>
            <button className="btn ghost" onClick={clearTranscript} aria-label="Clear conversation" title="Clear">
              ⟲
            </button>
            <button className="panel-close" onClick={() => setOpen(false)} aria-label="Close assistant">
              ✕
            </button>
          </div>

          <div className="assistant-log" ref={listRef} aria-live="polite">
            {transcript.length === 0 && (
              <div className="assistant-seed">
                <span className="eyebrow">ASTERRA Field Assistant</span>
                <p>
                  I'm your geospatial copilot — I answer questions about elevation models, remote sensing,
                  terrain analysis, photogrammetry, GIS, and the ASTERRA pipeline. I can also drive the
                  viewer with commands like <span className="mono">set exaggeration to 3</span>.
                </p>
              </div>
            )}
            {transcript.map((m, i) => (
              <div key={i} className={`assistant-msg ${m.role}`}>
                <div className="msg-meta mono">
                  <span className="who">{m.role === 'you' ? 'YOU' : 'DW'}</span>
                  <span className="t">{m.t}</span>
                </div>
                <p dangerouslySetInnerHTML={{ __html: formatMessage(m.text) }} />
              </div>
            ))}
            {busy && (
              <div className="assistant-msg dw busy-msg">
                <div className="msg-meta mono">
                  <span className="who">DW</span>
                  <span className="t">{stamp()}</span>
                </div>
                <div className="busy-content">
                  <span className="thinking-text">Thinking</span>
                  <span className="typing-dots" aria-hidden="true">
                    <span></span><span></span><span></span>
                  </span>
                </div>
              </div>
            )}
          </div>

          {showChips && (
            <div className="assistant-chips">
              {suggestions.map((sug) => (
                <button key={sug} className="chip" onClick={() => send(sug)}>
                  {sug}
                </button>
              ))}
            </div>
          )}

          <form
            className="assistant-input"
            onSubmit={(e) => {
              e.preventDefault()
              send()
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              placeholder={screen === 'viewer' ? 'Command the terrain…' : 'Ask about the scene…'}
              onChange={(e) => {
                setDraft(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = `${Math.min(96, e.target.scrollHeight)}px`
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              aria-label="Message the field assistant"
            />
            <button type="submit" className="send" disabled={!draft.trim() || busy} aria-label="Send message">
              ↑
            </button>
          </form>
        </div>
      </aside>
    </>
  )
}
