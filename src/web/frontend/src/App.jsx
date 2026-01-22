import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarkCanvasRef = useRef(null)
  const wsRef = useRef(null)

  const [isConnected, setIsConnected] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [confidence, setConfidence] = useState(0)
  const [currentText, setCurrentText] = useState('')
  const [history, setHistory] = useState([])
  const [landmarks, setLandmarks] = useState(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [isRightHand, setIsRightHand] = useState(true)
  const [showReference, setShowReference] = useState(false)
  const [activeModal, setActiveModal] = useState(null) // 'contribute', 'contact', or null

  // WebSocket Connection
  useEffect(() => {
    const connectWebSocket = () => {
      // Dynamic WebSocket URL for Deployment
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host // "localhost:8000" or "myapp.render.com"
      const wsUrl = `${protocol}//${host}/ws`

      // Fallback for local development if running on port 5173 (Vite default)
      // but wanting to connect to backend on 8000
      const finalUrl = host.includes('5173') ? 'ws://localhost:8000/ws' : wsUrl

      console.log('Connecting to WebSocket:', finalUrl)
      const ws = new WebSocket(finalUrl)

      ws.onopen = () => setIsConnected(true)

      ws.onmessage = (event) => {
        const response = JSON.parse(event.data)

        if (response.type === 'result') {
          const { prediction, confidence, current_sentence, history, landmarks, status_message, is_right_hand } = response.data
          setPrediction(prediction)
          setConfidence(confidence || 0)
          setCurrentText(current_sentence || '')
          setHistory(history || [])
          setStatusMessage(status_message || '')
          setLandmarks(landmarks || null)
          setIsRightHand(is_right_hand !== undefined ? is_right_hand : true)
        } else if (response.type === 'state') {
          setCurrentText(response.data.current_sentence || '')
          setHistory(response.data.history || [])
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        setTimeout(connectWebSocket, 3000)
      }

      wsRef.current = ws
    }

    connectWebSocket()
    return () => wsRef.current?.close()
  }, [])

  // Camera Setup
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }
    })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      })
      .catch(err => console.error('Camera error:', err))
  }, [])

  // Frame Streaming
  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current?.readyState === 4) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0, 640, 480)

        canvas.toBlob((blob) => {
          if (blob) {
            blob.arrayBuffer().then(buffer => {
              wsRef.current.send(buffer)
            })
          }
        }, 'image/jpeg', 0.85)
      }
    }, 100)

    return () => clearInterval(interval)
  }, [])

  // Sync canvas size with video visible area
  useEffect(() => {
    const syncCanvasSize = () => {
      if (videoRef.current && landmarkCanvasRef.current) {
        const video = videoRef.current
        const canvas = landmarkCanvasRef.current

        // Set canvas internal resolution to match video stream dimensions
        // CSS (width: 100%, height: 100%) will handle display scaling
        if (video.videoWidth && video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
      }
    }

    // Sync on load and resize
    if (videoRef.current) {
      videoRef.current.addEventListener('loadedmetadata', syncCanvasSize)
      setTimeout(syncCanvasSize, 100)
    }
    window.addEventListener('resize', syncCanvasSize)

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', syncCanvasSize)
      }
      window.removeEventListener('resize', syncCanvasSize)
    }
  }, [])

  // Draw Hand Landmarks and Bounding Box
  useEffect(() => {
    const canvas = landmarkCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!landmarks || landmarks.length === 0) return

    // CRITICAL: Calculate crop offset for object-fit: cover
    // The video is zoomed and cropped to fill the container
    const video = videoRef.current
    if (!video || !video.videoWidth || !video.videoHeight) return

    // Get CONTAINER (displayed) dimensions
    const containerRect = canvas.getBoundingClientRect()
    const containerAspect = containerRect.width / containerRect.height
    const videoAspect = video.videoWidth / video.videoHeight

    let renderWidth, renderHeight, offsetX, offsetY

    // Logic for object-fit: COVER (cropping)
    if (containerAspect > videoAspect) {
      // Container is wider than video -> Video is scaled to fit width, top/bottom cropped
      renderWidth = containerRect.width
      renderHeight = renderWidth / videoAspect
      offsetX = 0
      offsetY = (containerRect.height - renderHeight) / 2
    } else {
      // Container is taller than video -> Video is scaled to fit height, left/right cropped
      renderHeight = containerRect.height
      renderWidth = renderHeight * videoAspect
      offsetX = (containerRect.width - renderWidth) / 2
      offsetY = 0
    }

    // Scale from rendered pixels (virtual) to canvas internal resolution
    // Note: renderWidth/Height is the size the video WOULD be if fully visible
    const scale = canvas.width / containerRect.width // We assume canvas width matches container width in standard layout

    // Since we're mapping to internal canvas resolution which matches video resolution...
    // Let's do it simpler: Map normalized (0-1) to the "rendered" rectangle, then offset.

    // Wait, simpler approach for cover:
    // 1. Calculate the 'visible' portion of the video in normalized coordinates
    // 2. Map 0-1 from visible portion to 0-1 of canvas

    // Better approach matching previous reliable logic but inverted for crop:
    const points = landmarks.map(p => {
      // 1. Project normalized point to the full rendered video dimensions
      let x_rendered = (1 - p[0]) * renderWidth
      let y_rendered = p[1] * renderHeight

      // 2. Apply the crop offset (offsetY is negative if top is cropped, etc.)
      // Actually with the logic above:
      // If container wider (landscape), renderHeight > containerHeight. offsetY should be roughly -(renderHeight - containerHeight)/2

      // Let's use the calculated offsets from above
      let x_final = x_rendered + offsetX
      let y_final = y_rendered + offsetY

      // 3. Scale to canvas internal resolution
      // The canvas internal resolution is set to video.videoWidth/Height.
      // But purely visually, we want to draw on the canvas.
      // If we set canvas dimensions to containerRect dimensions, it's easier.
      // BUT current setup sets canvas.width = video.videoWidth.

      // Let's remap back to canvas internal resolution
      const canvasScaleX = canvas.width / containerRect.width
      const canvasScaleY = canvas.height / containerRect.height

      return {
        x: x_final * canvasScaleX,
        y: y_final * canvasScaleY
      }
    })

    // Calculate bounding box
    const xs = points.map(p => p.x)
    const ys = points.map(p => p.y)
    const minX = Math.min(...xs) - 20
    const maxX = Math.max(...xs) + 20
    const minY = Math.min(...ys) - 20
    const maxY = Math.max(...ys) + 20

    // Draw bounding box with Unified Cyber Purple or Red for error
    if (isRightHand) {
      const gradient = ctx.createLinearGradient(minX, minY, maxX, maxY)
      gradient.addColorStop(0, '#a855f7')  // Purple
      gradient.addColorStop(1, '#d946ef')  // Magenta
      ctx.strokeStyle = gradient
      ctx.shadowColor = '#d946ef'
    } else {
      ctx.strokeStyle = '#ff0000'  // Red for wrong hand
      ctx.shadowColor = '#ff0000'
    }

    ctx.lineWidth = 3

    // Add shadow blur for neon pulse effect
    ctx.shadowBlur = 15

    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY)

    // Reset shadow for other elements
    ctx.shadowBlur = 0

    // Draw hand connections
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],  // Index
      [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [0, 13], [13, 14], [14, 15], [15, 16],  // Ring
      [0, 17], [17, 18], [18, 19], [19, 20],  // Pinky
      [5, 9], [9, 13], [13, 17]  // Palm
    ]

    ctx.strokeStyle = isRightHand ? '#d946ef' : '#ff0000' // Cyber Purple vs Red
    ctx.lineWidth = 2
    connections.forEach(([start, end]) => {
      if (start < points.length && end < points.length) {
        ctx.beginPath()
        ctx.moveTo(points[start].x, points[start].y)
        ctx.lineTo(points[end].x, points[end].y)
        ctx.stroke()
      }
    })

    // Draw landmark points
    points.forEach(point => {
      ctx.strokeStyle = isRightHand ? '#a855f7' : '#ef4444' // Light Purple vs Red
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI) // Slightly larger dots
      ctx.fillStyle = isRightHand ? 'rgba(217, 70, 239, 0.3)' : 'rgba(239, 68, 68, 0.3)' // Semi-transparent fill
      ctx.fill()
      ctx.stroke()
    })
  }, [landmarks, isRightHand])

  const sendCommand = (action) => {
    console.log('Sending command:', action)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action }))
      console.log('Command sent successfully')
    } else {
      console.log('WebSocket not open, state:', wsRef.current?.readyState)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e) => {
      // Prevent default for all our shortcuts
      if (e.code === 'Space') {
        e.preventDefault();
        sendCommand('SPACE')
      }
      if (e.code === 'Backspace') {
        e.preventDefault();
        sendCommand('BACKSPACE')
      }
      if (e.code === 'Enter') {
        e.preventDefault();
        sendCommand('ENTER')
      }
      if (e.code === 'KeyC' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        sendCommand('CLEAR')
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

  const handDetected = landmarks && landmarks.length > 0

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo">
              <svg className="logo-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
            </div>
            <div>
              <h1 className="title">Silent Voice Bridge 💬</h1>
              <p className="subtitle">
                ASL Fingerspelling to English <span className="beta-tag">(Beta: Right Hand Only)</span>
              </p>
            </div>
          </div>

          <div className="header-right">
            <nav className="nav-menu">
              <button
                className={`nav-btn ghost ${activeModal === null ? 'active' : ''}`}
                onClick={() => setActiveModal(null)}
              >
                Home
              </button>
              <button
                className={`nav-btn ghost ${activeModal === 'contribute' ? 'active' : ''}`}
                onClick={() => setActiveModal('contribute')}
              >
                Contribute
              </button>
              <button
                className={`nav-btn primary-pill ${activeModal === 'contact' ? 'active' : ''}`}
                onClick={() => setActiveModal('contact')}
              >
                Contact Me
              </button>
            </nav>
            <div className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
              <div className="status-dot"></div>
              {isConnected ? 'Live' : 'Offline'}
            </div>
          </div>
        </div>
      </header>

      {/* MODALS */}
      {activeModal === 'contribute' && (
        <div className="modal-overlay">
          <div className="modal-content large">
            <div className="modal-header">
              <h2>Join Our Efforts</h2>
              <button className="close-btn" onClick={() => setActiveModal(null)}>×</button>
            </div>
            <p className="modal-desc">
              Our sign language translation app is open source and always improving, but we need your help!
              By contributing, you make the app more accessible to everyone.
            </p>

            <div className="contribute-grid">
              <div className="contribute-card">
                <h3>{'<>'} Develop</h3>
                <p>
                  Fix bugs. Improve code quality. <br />
                  <strong>We need more data!</strong> Help us train the model to be more responsive and correct across different hand structures.
                </p>
                <button className="action-btn" onClick={() => window.open('https://github.com/aramishf/silent-voice-bridge', '_blank')}>
                  View on GitHub
                </button>
              </div>
              <div className="contribute-card">
                <h3>🌍 Translate</h3>
                <p>Help make the app accessible in more languages.</p>
                <button className="action-btn" onClick={() => window.open('https://github.com/aramishf/silent-voice-bridge', '_blank')}>
                  View on GitHub
                </button>
              </div>
              <div className="contribute-card">
                <h3>💬 Feedback</h3>
                <p>Let us know how to make the app better!</p>
                <button className="action-btn" onClick={() => window.open('https://github.com/aramishf/silent-voice-bridge/issues', '_blank')}>
                  Give Feedback
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeModal === 'contact' && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>About & Future Work</h2>
              <button className="close-btn" onClick={() => setActiveModal(null)}>×</button>
            </div>

            <div className="about-section">
              <div className="hero-text">
                <h3>Glimpse into the future of accessibility</h3>
                <p>
                  This project works towards translating sign language to English text in real-time.
                  Currently a <strong>prototype</strong> to start somewhere, focusing on fingerspelling.
                </p>
                <p>
                  <strong>Future Goal:</strong> Full sentence translation and two-hand support.
                </p>
              </div>

              <div className="contact-box">
                <p>Have ideas or want to collaborate?</p>
                <div className="contact-buttons">
                  <a href="mailto:farooqaramish@gmail.com" className="email-btn">
                    📧 Email Me
                  </a>
                  <a href="https://www.linkedin.com/in/aramishfarooq/" target="_blank" rel="noopener noreferrer" className="linkedin-btn">
                    👔 LinkedIn
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Grid Layout */}
      <main className="main-grid">

        {/* LEFT: Camera Feed */}
        <div className="camera-panel">
          <div className="card">
            <div className="card-header">
              <h2>Camera Feed</h2>
              {/* Detection badge hidden to prevent clutter */}
            </div>

            <div className="video-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="video-feed"
              />

              {/* Hidden canvas for frame capture */}
              <canvas ref={canvasRef} width="640" height="480" className="hidden-canvas" />

              {/* Visible overlay canvas for landmarks */}
              <canvas
                ref={landmarkCanvasRef}
                width="640"
                height="480"
                className="landmark-canvas"
              />

              {/* Status Badge - TOP LEFT */}
              <div className="status-hud">
                <div className="status-hud-dot"></div>
                <span className="status-hud-text">Live</span>
              </div>

              {/* Confidence HUD - TOP RIGHT */}
              {prediction && confidence > 0.5 && (
                <div className="confidence-hud">
                  <div className="confidence-hud-label">Detecting</div>
                  <div className="confidence-hud-content">
                    <span className="confidence-letter">{prediction}</span>
                    <span className={`confidence-percent ${confidence > 0.8 ? 'high' : confidence > 0.5 ? 'medium' : 'low'
                      }`}>
                      {Math.round(confidence * 100)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Status/Instruction Message (e.g. Release Hand) */}
              {statusMessage && (
                <div className="instruction-hud">
                  <span className="instruction-icon">⚠️</span>
                  <span className="instruction-text">{statusMessage}</span>
                </div>
              )}

              {/* Subtitle Bar - BOTTOM */}
              <div className="subtitle-bar">
                <div className={`subtitle-text ${!currentText ? 'empty' : ''}`}>
                  {currentText || 'Start signing to see translation...'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: Translation Console */}
        <div className="console-panel">
          <div className="card">
            <div className="card-header">
              <h2>English Translation</h2>
            </div>

            <div className="console-content">
              {/* Translation Text Area */}
              <div className="text-display">
                {currentText || <span className="placeholder">Start signing...</span>}
                {currentText && <span className="cursor"></span>}
              </div>

              {/* Toolbar */}
              <div className="toolbar">
                <button onClick={() => sendCommand('SPACE')} className="btn btn-outline" title="Add space">
                  <span className="btn-label">SPACE</span>
                  <span className="btn-icon">␣</span>
                </button>
                <button onClick={() => sendCommand('BACKSPACE')} className="btn btn-outline" title="Backspace">
                  <span className="btn-label">BACKSPACE</span>
                  <span className="btn-icon">⌫</span>
                </button>
                <button onClick={() => sendCommand('CLEAR')} className="btn btn-outline" title="Clear All">
                  <span className="btn-label">CLEAR</span>
                  <span className="btn-icon">🗑️</span>
                </button>
                <button onClick={() => sendCommand('ENTER')} className="btn btn-primary" title="Save Sentence">
                  <span className="btn-label">SAVE</span>
                  <span className="btn-icon">💾</span>
                </button>
              </div>

              {/* History */}
              {history.length > 0 && (
                <div className="history">
                  <div className="history-header">📝 Saved Sentences ({history.length})</div>
                  <div className="history-list">
                    {history.slice().reverse().map((item, i) => (
                      <div key={i} className="history-item">
                        <span className="history-time">{item[0]}</span>
                        <span className="history-text">{item[1]}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Bottom: ASL Reference (Collapsible) */}
      {/* Bottom Fixed Dock (Guide + Footer) */}
      <div className="bottom-dock">
        {/* Conditionally Rendered Guide Content */}
        {showReference && (
          <div className="guide-content">
            <img
              src="/asl-reference.png"
              alt="ASL Alphabet Chart (A-Z, 0-9)"
              className="guide-image"
            />
          </div>
        )}

        {/* Trigger Handle */}
        <button
          className="guide-trigger"
          onClick={() => setShowReference(!showReference)}
          title="Toggle ASL Guide"
        >
          <span className="trigger-text">ASL Alphabet and 0-9 Guide</span>
          <span className="trigger-icon">{showReference ? '▼' : '▲'}</span>
        </button>

        {/* Footer Credits */}
        <div className="footer-credits">
          <p>Made with ❤️ for the Deaf and Hard of Hearing community</p>
        </div>
      </div>
    </div>
  )
}

export default App
