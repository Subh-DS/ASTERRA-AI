import { Component } from 'react'

// Last-resort guard: without this, any render-time throw (e.g. a flaky
// WebGL/driver hiccup) unmounts the whole tree into a blank page.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error) {
    console.warn('[error-boundary] caught:', error?.message || error)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="size-notice">
          <div className="box">
            <span className="eyebrow">DepthWizard</span>
            <h2>Something failed to render</h2>
            <p>
              {this.state.error?.message || 'Unexpected rendering error.'}
              {' '}Your data and the backend are unaffected.
            </p>
            <p>
              <button type="button" className="btn primary" onClick={() => window.location.reload()}>
                Reload
              </button>
            </p>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
