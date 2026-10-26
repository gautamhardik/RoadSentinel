---
name: Tactical Swiss Autonomous Perception
colors:
  canvas: '#090A0D'
  surface: '#111318'
  surface-elevated: '#171A21'
  border-hairline: '#1E222B'
  border-subtle: '#2A303C'
  text-primary: '#F3F4F6'
  text-secondary: '#8A93A6'
  text-muted: '#4B5565'
  tactical-emerald: '#10B981'
  signal-amber: '#F59E0B'
  alert-coral: '#F43F5E'
  cool-ice-blue: '#38BDF8'
typography:
  font-primary: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
  font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace
  tabular-nums: true
  sizes:
    label-micro:
      fontSize: 10px
      lineHeight: 12px
      letterSpacing: 0.08em
      textTransform: uppercase
    label-mono:
      fontSize: 11px
      lineHeight: 14px
      letterSpacing: 0.06em
      textTransform: uppercase
    body-sm:
      fontSize: 12px
      lineHeight: 16px
    body-md:
      fontSize: 13px
      lineHeight: 18px
    body-lg:
      fontSize: 14px
      lineHeight: 20px
    heading-sm:
      fontSize: 16px
      lineHeight: 22px
      fontWeight: 600
    heading-md:
      fontSize: 20px
      lineHeight: 26px
      fontWeight: 600
    heading-lg:
      fontSize: 28px
      lineHeight: 34px
      fontWeight: 700
rounded:
  tight: 4px
  pill: 9999px
---

# RoadSentinel Cloud // Tactical Swiss Autonomous Perception Design System

## 1. Design Ethos: "Tactical Swiss Engineering" (Zero AI Slop)

RoadSentinel Cloud is designed for autonomous fleet safety directors, AV perception architects, and mission-critical safety engineers (in the aesthetic lineage of Anduril Lattice, Foxglove Studio, Linear, and Teenage Engineering).

### Core Aesthetic Rules:
1. **Zero Neon Gradients:** Strictly no purple-to-cyan glowing gradients, fuzzy drop-shadow soup, or AI sparkle badges.
2. **Matte Obsidian & Hairline Structural Grid:** Deep matte canvas (`#090A0D`), elevated slate panels (`#111318`), and crisp `1px solid #1E222B` structural grid lines (`divide-x divide-y divide-[#1E222B]`).
3. **Typography & Tabular Numerals:** Headers and structural chrome use `Plus Jakarta Sans`. All coordinates, confidences, latencies, hashes, and telemetry use `JetBrains Mono` with mandatory `font-variant-numeric: tabular-nums` to eliminate jitter during real-time video playback and streaming calculations.
4. **Semantic Tactical Color Palette:**
   - **Tactical Emerald (`#10B981`)**: Confirmed locked true positive, recovered VRU, WBF consensus agreement.
   - **Signal Amber (`#F59E0B`)**: Epistemic uncertainty, active MCP tool invocation, active foveation beam, counterfactual secondary crop pass.
   - **Alert Coral (`#F43F5E`)**: Baseline single-pass false negative (missed VRU), safety escalation required.
   - **Precision Ice Blue (`#38BDF8`)**: AWS Graviton Arm64 & COOL hardware acceleration flags, SIMD NEON/KleidiCV indicators.
   - **Muted Steels (`#4B5565` - `#8A93A6`)**: Grid guides, 3D ground coordinates, secondary metadata.

---

## 2. Minimalist Feature Architecture: 3 Studio Workspaces

The application executes within a single-viewport (100vh) studio shell with a 48px top tactical command bar and an instant 3-mode segmented switcher:

### Workspace 01 // PERCEPTION STUDIO ($1,000 Agentic Vision Award)
- **Left Pane (7 Cols):** High-definition BDD100K road camera viewport with SVG corner-bracket HUD reticles (no thick cartoon boxes), 2.5x circular optical magnifier loupe (reporting instantaneous Luma and Laplacian variance under the crosshair), live drag-and-drop image upload, and an interactive 12-scenario golden filmstrip.
- **Top Right (5 Cols):** Interactive split-curtain $640\times640$ crop comparator (dragging the vertical hairline divider smoothly reveals Raw Unenhanced vs. OpenCV 5 CLAHE/Bilateral Enhanced) + counterfactual confidence bar + Three.js 3D frustum mini-viewport.
- **Bottom Right (5 Cols):** 5-Stage MCP execution trace with real-time millisecond latency badges and live operator safety action buttons (`[Approve Autonomous Lock]`, `[Escalate to Safety Engineer]`).

### Workspace 02 // CONTINUOUS VIDEO & 3D CORRIDOR ($5,000 Grand Prize)
- **Left Pane (8 Cols):** Synchronized dual-stream H.264 video comparison player (Standard Single-Pass YOLO11m vs. RoadSentinel Active Foveation + Farneback Optical Flow + ByteTrack) with frame-stepping controls (`-1f`, `Play/Pause`, `+1f`), 0.5x/1.0x playback speed, and live VRU recovery counter.
- **Right Pane (4 Cols):** Expanded interactive Three.js 3D spatial corridor with perspective ground grid, distance rings (10m, 25m, 50m), vehicle frustum wireframe, 3D VRU tracking pillars, dynamic foveation laser beam, and live Farneback optical flow velocity vectors.

### Workspace 03 // AWS GRAVITON + COOL & EVIDENCE LEDGER ($1,000 COOL Award & Grand Prize)
- **Top Left:** Interactive AWS Fleet FinOps Calculator slider (1M to 500M frames/month) with mechanical rolling odometer numbers comparing Intel x86 `c6i.xlarge` vs. Arm64 Baseline vs. AWS Graviton3/4 `c7g/c8g` + COOL, plus a `[Run Live OpenCV 5 Server Benchmark]` button executing 25 live iterations of CLAHE, Resize, Bilateral, Laplacian, and Optical Flow on the host.
- **Top Right:** Cryptographic model registry with live SHA-256 verification badges (`session4_best.pt`, `best.pt`, `yolo11m_p2`) and host SIMD flags.
- **Bottom Full-Width Swiss Table:** Audited scientific evaluation ledger across Phase 3.4 (10,000 images, +10.44 pp small-VRU recall), Phase 6, Phase 11 (2,000-image blind test with $B=2,000$ bootstrap CIs), and Phase 12 (YOLO11m-P2 stride-4 study).

---

## 3. Specialized Interactive Components

1. **SVG Corner-Bracket Reticles:**
   ```html
   <svg class="reticle-box" viewBox="0 0 100 100">
     <!-- 4 sharp corner ticks with 1.5px stroke -->
     <path d="M 0,15 L 0,0 L 15,0" stroke="#10B981" stroke-width="2" fill="none"/>
     <path d="M 85,0 L 100,0 L 100,15" stroke="#10B981" stroke-width="2" fill="none"/>
     <path d="M 100,85 L 100,100 L 85,100" stroke="#10B981" stroke-width="2" fill="none"/>
     <path d="M 15,100 L 0,100 L 0,85" stroke="#10B981" stroke-width="2" fill="none"/>
   </svg>
   ```
2. **Split-Curtain Slider:** Vertical 1px hairline slider handle with double chevron indicator (`⟨|⟩`) allowing smooth mouse/touch wiping between the raw crop and OpenCV-enhanced crop.
3. **Three.js 3D Frustum:** Rendered using WebGL on an HTML5 canvas with zero external UI bloat. Dark gray grid with circular distance markers (`10m`, `25m`, `50m`), an ego-vehicle marker at `(0, 0, 0)`, a wireframe camera pyramid extending along the positive Z-axis, and dynamic vertical bounding pillars for active road targets.
4. **Mechanical Rolling Odometer:** CSS/JS tabular-num transitions where numbers roll vertically when values update.
5. **Instant `⌘K` / `Ctrl+K` Command Palette:** Modal overlay with instant fuzzy keyboard search across all 12 scenarios, benchmark triggers, and developer API routes.
