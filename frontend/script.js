// ============================================================
// CONFIG — FastAPI Backend Address
const DEPLOYED_BACKEND_URL = "https://seizure-prediction-api.onrender.com"; 

const isLocalhost = Boolean(
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.hostname === ""
);

const API_BASE = window.API_BASE
  || localStorage.getItem("SEIZURE_API_BASE")
  || (isLocalhost ? "http://localhost:8000" : DEPLOYED_BACKEND_URL);

// State management - Honest default
let currentMode = "leakage_corrected";
let currentIcuState = "calm";
let currentIcuRisk = 5;
let sampleRecords = [];
let activePatient = null;
let currentFeatureColumns = [];

// Verified local cached metrics fallback (prevents blank UI during Render cold start)
const CACHED_METRICS = {
  leakage_corrected: {
    mstm_accuracy: 0.4219,
    sbam_accuracy: 0.4237,
    fusion_accuracy: 0.4219,
    majority_baseline_accuracy: 0.4219,
    logistic_regression_accuracy: 0.4219,
    classes: ["GPD", "GRDA", "LPD", "LRDA", "Other", "Seizure"],
    classification_report: {
      "GPD": { precision: 0.0, recall: 0.0, "f1-score": 0.0, support: 205 },
      "GRDA": { precision: 0.0, recall: 0.0, "f1-score": 0.0, support: 284 },
      "LPD": { precision: 0.0, recall: 0.0, "f1-score": 0.0, support: 257 },
      "LRDA": { precision: 0.0, recall: 0.0, "f1-score": 0.0, support: 133 },
      "Other": { precision: 0.4219, recall: 1.0, "f1-score": 0.5934, support: 940 },
      "Seizure": { precision: 0.0, recall: 0.0, "f1-score": 0.0, support: 409 },
      "macro avg": { precision: 0.0703, recall: 0.1667, "f1-score": 0.0989, support: 2228 }
    },
    confusion_matrix: [
      [0, 0, 0, 0, 205, 0],
      [0, 0, 0, 0, 284, 0],
      [0, 0, 0, 0, 257, 0],
      [0, 0, 0, 0, 133, 0],
      [0, 0, 0, 0, 940, 0],
      [0, 0, 0, 0, 409, 0]
    ]
  },
  as_trained: {
    mstm_accuracy: 0.9179,
    sbam_accuracy: 0.9888,
    fusion_accuracy: 0.9933,
    classes: ["GPD", "GRDA", "LPD", "LRDA", "Other", "Seizure"],
    classification_report: {
      "GPD": { precision: 0.9903, recall: 0.9951, "f1-score": 0.9927, support: 205 },
      "GRDA": { precision: 0.9861, recall: 0.9965, "f1-score": 0.9912, support: 284 },
      "LPD": { precision: 0.9808, recall: 0.9922, "f1-score": 0.9865, support: 257 },
      "LRDA": { precision: 1.0, recall: 0.9925, "f1-score": 0.9962, support: 133 },
      "Other": { precision: 0.9979, recall: 0.9915, "f1-score": 0.9947, support: 940 },
      "Seizure": { precision: 0.9951, recall: 0.9951, "f1-score": 0.9951, support: 409 },
      "macro avg": { precision: 0.9917, recall: 0.9938, "f1-score": 0.9927, support: 2228 }
    },
    confusion_matrix: [
      [204, 0, 1, 0, 0, 0],
      [0, 283, 1, 0, 0, 0],
      [0, 0, 255, 0, 1, 1],
      [0, 0, 1, 132, 0, 0],
      [1, 4, 2, 0, 932, 1],
      [1, 0, 0, 0, 1, 407]
    ]
  }
};

// ============================================================
// Particle Neural Network Background Animation
// ============================================================
const bgCanvas = document.getElementById("bg-canvas");
if (bgCanvas) {
  const bgCtx = bgCanvas.getContext("2d");
  let particles = [];
  
  function resizeBg() {
    bgCanvas.width = window.innerWidth;
    bgCanvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resizeBg);
  resizeBg();

  class Particle {
    constructor() {
      this.x = Math.random() * bgCanvas.width;
      this.y = Math.random() * bgCanvas.height;
      this.vx = (Math.random() - 0.5) * 0.35;
      this.vy = (Math.random() - 0.5) * 0.35;
      this.radius = Math.random() * 2 + 1;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > bgCanvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > bgCanvas.height) this.vy *= -1;
    }
    draw() {
      bgCtx.beginPath();
      bgCtx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      bgCtx.fillStyle = "rgba(20, 184, 166, 0.15)";
      bgCtx.fill();
    }
  }

  for (let i = 0; i < 60; i++) {
    particles.push(new Particle());
  }

  function animateBg() {
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    particles.forEach((p, idx) => {
      p.update();
      p.draw();
      
      for (let j = idx + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 120) {
          bgCtx.beginPath();
          bgCtx.moveTo(p.x, p.y);
          bgCtx.lineTo(p2.x, p2.y);
          bgCtx.strokeStyle = `rgba(20, 184, 166, ${0.1 * (1 - dist / 120)})`;
          bgCtx.lineWidth = 0.5;
          bgCtx.stroke();
        }
      }
    });
    requestAnimationFrame(animateBg);
  }
  animateBg();
}

// ============================================================
// Simulators: ICU Bedside Monitor Waveform Engine
// ============================================================
class ScrollingEEG {
  constructor(canvasId, seed = 0) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.seed = seed;
    this.points = [];
    this.maxPoints = 300;
    this.t = 0;
    
    this.resize();
    window.addEventListener("resize", () => this.resize());
    
    for (let i = 0; i < this.maxPoints; i++) {
      this.points.push(0);
    }
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * (window.devicePixelRatio || 1);
    this.canvas.height = rect.height * (window.devicePixelRatio || 1);
    this.ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    this.w = rect.width;
    this.h = rect.height;
  }

  addPoint(val) {
    this.points.push(val);
    if (this.points.length > this.maxPoints) {
      this.points.shift();
    }
  }

  draw(color) {
    if (!this.canvas) return;
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    
    for (let i = 0; i < this.points.length; i++) {
      const x = (i / (this.maxPoints - 1)) * w;
      const y = h / 2 + this.points[i] * (h / 2) * 0.8;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
}

const eegChannels = [
  new ScrollingEEG("canvasCh1", 0.5),
  new ScrollingEEG("canvasCh2", 1.5),
  new ScrollingEEG("canvasCh3", 2.5),
  new ScrollingEEG("canvasCh4", 3.5),
];

function generateEEGValue(t, seed) {
  const risk = currentIcuRisk / 100.0;
  
  // Calm component decreases with risk
  const calmVal = (Math.sin(t * 7 + seed * 8) * 0.15 + Math.sin(t * 16 + seed * 4) * 0.08) * Math.max(0, 1 - risk * 1.3);
  
  // Pre-ictal warning oscillations rise in the middle range
  const warnFactor = risk <= 0.8 ? (risk * 1.2) : (1.0 - (risk - 0.8) * 5);
  const warnVal = (Math.sin(t * 5.2 + seed * 0.6) * 0.28) * Math.max(0, warnFactor);
  const occasionalSpike = (Math.sin(t * 1.6 + seed) > 0.86 ? Math.sin(t * 26) * 0.38 : 0) * Math.max(0, warnFactor);
  
  // Full Seizure lockstep high-voltage rhythm dominates above 75% risk
  const seizureFactor = risk > 0.75 ? (risk - 0.75) * 4 : 0;
  const seizureVal = (Math.sin(t * 2.8 + seed * 0.1) * 0.65 + Math.sin(t * 5.6 + seed * 0.1) * 0.25) * seizureFactor;
  
  // Noise baseline scales up with risk
  const noiseVal = (Math.random() - 0.5) * (0.06 + 0.08 * risk);
  
  return Math.max(-1.0, Math.min(1.0, calmVal + warnVal + occasionalSpike + seizureVal + noiseVal));
}

// ============================================================
// Simulators: 2D Spectrogram Generator
// ============================================================
const specCanvas = document.getElementById("spectrogramCanvas");
if (specCanvas) {
  const specCtx = specCanvas.getContext("2d");
  const numBins = 48; 
  const historyWidth = 320; 
  let specHistory = [];

  function resizeSpec() {
    const rect = specCanvas.getBoundingClientRect();
    specCanvas.width = rect.width;
    specCanvas.height = rect.height;
  }
  window.addEventListener("resize", resizeSpec);
  resizeSpec();

  for (let i = 0; i < historyWidth; i++) {
    specHistory.push(new Array(numBins).fill(0));
  }

  function addSpectrogramSlice() {
    const slice = [];
    const risk = currentIcuRisk / 100.0;
    
    for (let f = 0; f < numBins; f++) {
      const calmIntensity = (Math.random() * 0.15 + (f > 20 ? Math.random() * 0.15 : 0)) * Math.max(0, 1 - risk * 1.3);
      const warnFactor = risk <= 0.8 ? (risk * 1.2) : (1.0 - (risk - 0.8) * 5);
      const warnIntensity = (Math.sin(eegTime * 1.5 + f * 0.1) * 0.25 + 0.25 + Math.random() * 0.2) * Math.max(0, warnFactor);
      
      const seizureFactor = risk > 0.75 ? (risk - 0.75) * 4 : 0;
      const seizureIntensity = (Math.sin(eegTime * 4.2) > 0.3 ? 0.75 : 0.2 + Math.random() * 0.25) * seizureFactor;
      
      const intensity = calmIntensity + warnIntensity + seizureIntensity;
      slice.push(Math.max(0, Math.min(1.0, intensity)));
    }
    specHistory.push(slice);
    if (specHistory.length > historyWidth) specHistory.shift();
  }

  function drawSpectrogram() {
    const w = specCanvas.width;
    const h = specCanvas.height;
    specCtx.clearRect(0, 0, w, h);
    
    const sliceW = w / historyWidth;
    const binH = h / numBins;

    for (let i = 0; i < historyWidth; i++) {
      const slice = specHistory[i];
      const x = i * sliceW;
      for (let f = 0; f < numBins; f++) {
        const y = h - (f + 1) * binH;
        const val = slice[f];
        
        let r, g, b;
        if (currentIcuRisk <= 30) {
          r = Math.floor(13 * val);
          g = Math.floor(184 * val);
          b = Math.floor(166 * val);
        } else if (currentIcuRisk > 30 && currentIcuRisk <= 80) {
          r = Math.floor(245 * val);
          g = Math.floor(158 * val);
          b = Math.floor(11 * val);
        } else {
          r = Math.floor(244 * val);
          g = Math.floor(63 * val);
          b = Math.floor(94 * val);
        }

        specCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        specCtx.fillRect(x, y, sliceW + 0.5, binH + 0.5);
      }
    }
  }

  setInterval(() => {
    addSpectrogramSlice();
    drawSpectrogram();
  }, 50);
}

// Tick loop for drawing EEG waves
let eegTime = 0;
function tickEEG() {
  eegTime += 0.08;
  
  let color = "#14B8A6"; 
  if (currentIcuRisk > 30 && currentIcuRisk <= 80) color = "#F59E0B"; 
  if (currentIcuRisk > 80) color = "#F43F5E"; 
  
  eegChannels.forEach((chan) => {
    const val = generateEEGValue(eegTime, chan.seed);
    chan.addPoint(val);
    chan.draw(color);
  });
  
  requestAnimationFrame(tickEEG);
}

setTimeout(tickEEG, 500);

function setIcuRisk(value) {
  let val = parseInt(value);
  if (isNaN(val)) val = 0;
  if (val < 0) val = 0;
  if (val > 100) val = 100;
  
  currentIcuRisk = val;
  
  // Sync slider and input box
  const inputEl = document.getElementById("riskInput");
  const sliderEl = document.getElementById("riskSlider");
  if (inputEl && inputEl.value != val) inputEl.value = val;
  if (sliderEl && sliderEl.value != val) sliderEl.value = val;
  
  // Categorize state based on threshold
  let state = "calm";
  if (val > 30 && val <= 80) state = "warning";
  if (val > 80) state = "seizure";
  
  currentIcuState = state;
  
  const riskVal = document.getElementById("icu-risk-val");
  const statusText = document.getElementById("icu-status-text");
  const riskBadge = document.getElementById("icu-risk-badge");
  
  riskVal.className = "icu-metric-value state-" + state;
  riskVal.textContent = val + "%";
  
  if (state === "calm") {
    statusText.textContent = "Patient Status: Stable";
    riskBadge.style.background = "var(--teal-glow)";
    riskBadge.style.color = "var(--teal-light)";
    riskBadge.style.borderColor = "rgba(20, 184, 166, 0.3)";
    riskBadge.textContent = "CALM";
  } else if (state === "warning") {
    statusText.textContent = "Patient Status: Pre-Ictal Pattern Detected";
    riskBadge.style.background = "var(--amber-glow)";
    riskBadge.style.color = "var(--amber-light)";
    riskBadge.style.borderColor = "rgba(245, 158, 11, 0.3)";
    riskBadge.textContent = "WARNING";
  } else if (state === "seizure") {
    statusText.textContent = "Patient Status: ACTIVE SEIZURE EMERGENCY";
    riskBadge.style.background = "var(--coral-glow)";
    riskBadge.style.color = "var(--coral-light)";
    riskBadge.style.borderColor = "rgba(244, 63, 94, 0.3)";
    riskBadge.textContent = "CRITICAL";
  }
  
  updateWearableAlert(state);
}

// ============================================================
// Interactive 3D Brain Synchronization Sphere Visualizer
// ============================================================
const syncWebCanvas = document.getElementById("syncWebCanvas");
if (syncWebCanvas) {
  const syncWebCtx = syncWebCanvas.getContext("2d");
  
  let yaw = 0.2;  // Rotation angles
  let pitch = -0.3;
  let isDragging = false;
  let prevMouseX = 0;
  let prevMouseY = 0;

  function resizeSyncWeb() {
    const rect = syncWebCanvas.getBoundingClientRect();
    syncWebCanvas.width = rect.width;
    syncWebCanvas.height = rect.height;
  }
  window.addEventListener("resize", resizeSyncWeb);
  resizeSyncWeb();

  // 3D Nodes mapped on a sphere silhouette
  const nodes3D = [
    { label: "FP1", x: -0.3, y: 0.8, z: 0.6 },
    { label: "FP2", x: 0.3, y: 0.8, z: 0.6 },
    { label: "C3",  x: -0.6, y: 0.2, z: 0.3 },
    { label: "C4",  x: 0.6, y: 0.2, z: 0.3 },
    { label: "T3",  x: -0.85, y: -0.1, z: -0.1 },
    { label: "T4",  x: 0.85, y: -0.1, z: -0.1 },
    { label: "O1",  x: -0.4, y: -0.8, z: -0.5 },
    { label: "O2",  x: 0.4, y: -0.8, z: -0.5 },
  ];

  // Drag listeners
  syncWebCanvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
  });

  window.addEventListener("mouseup", () => isDragging = false);

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) {
      // Auto-rotation bias slightly shifts when mouse moves over canvas
      const rect = syncWebCanvas.getBoundingClientRect();
      if (e.clientX >= rect.left && e.clientX <= rect.right &&
          e.clientY >= rect.top && e.clientY <= rect.bottom) {
        yaw += (e.clientX - (rect.left + rect.width / 2)) * 0.0001;
      }
      return;
    }
    const dx = e.clientX - prevMouseX;
    const dy = e.clientY - prevMouseY;
    yaw += dx * 0.007;
    pitch += dy * 0.007;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
  });

  let syncTime = 0;
  function drawSyncWeb() {
    if (!isDragging) {
      yaw += 0.004; // Gentle auto-rotate
    }
    syncTime += 0.035;
    
    const w = syncWebCanvas.width;
    const h = syncWebCanvas.height;
    syncWebCtx.clearRect(0, 0, w, h);
    
    const cx = w / 2;
    const cy = h / 2;
    const scale = Math.min(cx, cy) * 0.65;

    // 3D coordinate transformation matrices
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);

    const projected = nodes3D.map((n) => {
      // Rotate around Y axis (Yaw)
      let x1 = n.x * cosY - n.z * sinY;
      let z1 = n.x * sinY + n.z * cosY;
      
      // Rotate around X axis (Pitch)
      let y2 = n.y * cosP - z1 * sinP;
      let z2 = n.y * sinP + z1 * cosP;
      
      // Project to 2D
      const px = cx + x1 * scale;
      const py = cy - y2 * scale; // invert Y
      return { label: n.label, x: px, y: py, z: z2 };
    });

    // Draw connecting lines sorted by depth (back-to-front rendering)
    for (let i = 0; i < projected.length; i++) {
      const n1 = projected[i];
      for (let j = i + 1; j < projected.length; j++) {
        const n2 = projected[j];
        
        let syncVal = 0;
        let strokeColor = "rgba(255, 255, 255, 0.03)";
        
        if (currentIcuRisk <= 30) {
          syncVal = 0.05 + Math.sin(syncTime + i * 2 + j) * 0.05;
          strokeColor = `rgba(20, 184, 166, ${syncVal})`;
        } else if (currentIcuRisk > 30 && currentIcuRisk <= 80) {
          syncVal = 0.3 + Math.sin(syncTime * 2 + i + j) * 0.25;
          strokeColor = `rgba(245, 158, 11, ${syncVal})`;
        } else {
          syncVal = 0.8 + Math.abs(Math.sin(syncTime * 4.5)) * 0.2;
          strokeColor = `rgba(244, 63, 94, ${syncVal})`;
        }
        
        // Depth modulation: dimmer lines if far away (negative z)
        const avgZ = (n1.z + n2.z) / 2; // range [-1, 1] roughly
        const depthAlpha = (avgZ + 1.2) / 2.2;
        
        syncWebCtx.beginPath();
        syncWebCtx.moveTo(n1.x, n1.y);
        syncWebCtx.lineTo(n2.x, n2.y);
        syncWebCtx.strokeStyle = strokeColor;
        syncWebCtx.globalAlpha = Math.max(0.1, depthAlpha);
        syncWebCtx.lineWidth = syncVal * 3.5;
        syncWebCtx.stroke();
        syncWebCtx.globalAlpha = 1.0;
      }
    }

    // Draw projected nodes
    projected.forEach((n) => {
      let nodeColor = "#64748B";
      let glowColor = "rgba(0,0,0,0)";
      if (currentIcuRisk <= 30) {
        nodeColor = "#14B8A6";
        glowColor = "var(--teal-glow)";
      } else if (currentIcuRisk > 30 && currentIcuRisk <= 80) {
        nodeColor = "#F59E0B";
        glowColor = "var(--amber-glow)";
      } else {
        nodeColor = "#F43F5E";
        glowColor = "var(--coral-glow)";
      }

      // Depth modulation for size & opacity
      const depthScale = (n.z + 1.5) / 2.5; 
      
      syncWebCtx.beginPath();
      syncWebCtx.arc(n.x, n.y, 14 * depthScale, 0, Math.PI * 2);
      syncWebCtx.fillStyle = glowColor;
      syncWebCtx.globalAlpha = Math.max(0.1, depthScale);
      syncWebCtx.fill();

      syncWebCtx.beginPath();
      syncWebCtx.arc(n.x, n.y, 6 * depthScale, 0, Math.PI * 2);
      syncWebCtx.fillStyle = nodeColor;
      syncWebCtx.fill();
      syncWebCtx.globalAlpha = 1.0;

      // Draw label
      syncWebCtx.font = `${Math.floor(9 * depthScale)}px 'Fira Code', monospace`;
      syncWebCtx.fillStyle = `rgba(148, 163, 184, ${Math.max(0.2, depthScale)})`;
      syncWebCtx.textAlign = "center";
      syncWebCtx.textBaseline = "middle";
      syncWebCtx.fillText(n.label, n.x, n.y - 14 * depthScale);
    });

    requestAnimationFrame(drawSyncWeb);
  }
  setTimeout(drawSyncWeb, 500);
}

// ============================================================
// Simulators: Wearable Early Warning Simulator (Phone View)
// ============================================================
function updateWearableAlert(state) {
  const card = document.getElementById("phoneNotification");
  const watchStatus = document.getElementById("watchSyncStatus");
  const phoneEegStatus = document.getElementById("phoneEegStatus");
  const phoneTime = document.getElementById("phoneTime");
  const phoneAlertTime = document.getElementById("phoneAlertTime");
  
  if (phoneTime) {
    const now = new Date();
    phoneTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  if (phoneAlertTime) {
    phoneAlertTime.textContent = "Just Now";
  }

  if (!card || !watchStatus) return;
  
  card.className = "phone-alert-card alert-active";
  
  if (state === "calm") {
    card.classList.remove("alert-warning", "alert-seizure");
    card.querySelector(".phone-alert-title").textContent = "Normal Status";
    card.querySelector(".phone-alert-body").textContent = `Neural synchronization parameters are within baseline limits. Risk index: ${currentIcuRisk}%.`;
    watchStatus.textContent = "Active (Stable)";
    watchStatus.style.color = "var(--teal-light)";
    if (phoneEegStatus) {
      phoneEegStatus.textContent = "STABLE";
      phoneEegStatus.style.color = "var(--teal-light)";
    }
  } else if (state === "warning") {
    card.classList.add("alert-warning");
    card.classList.remove("alert-seizure");
    card.querySelector(".phone-alert-title").textContent = "⚠️ High Pattern Anomaly";
    card.querySelector(".phone-alert-body").textContent = `Harmful brain activity pattern detected (Classification risk score: ${currentIcuRisk}%). Snapshot classification based on electrographic features; not predictive forecasting.`;
    watchStatus.textContent = "Warning (Pattern Alert)";
    watchStatus.style.color = "var(--amber-light)";
    if (phoneEegStatus) {
      phoneEegStatus.textContent = "PATTERN ANOMALY";
      phoneEegStatus.style.color = "var(--amber-light)";
    }
  } else if (state === "seizure") {
    card.classList.add("alert-seizure");
    card.classList.remove("alert-warning");
    card.querySelector(".phone-alert-title").textContent = "🚨 Electrographic Seizure Pattern";
    card.querySelector(".phone-alert-body").textContent = `Ictal electrographic pattern classified with high probability (Score: ${currentIcuRisk}%). Research simulation only — not for diagnostic or therapeutic use.`;
    watchStatus.textContent = "ICTAL CLASSIFICATION";
    watchStatus.style.color = "var(--coral-light)";
    if (phoneEegStatus) {
      phoneEegStatus.textContent = "SEIZURE DETECTED";
      phoneEegStatus.style.color = "var(--coral-light)";
    }
  }
}

// Live mini-EEG waveform on phone screen
const phoneEegCanvas = document.getElementById("phoneMiniEegCanvas");
if (phoneEegCanvas) {
  const pCtx = phoneEegCanvas.getContext("2d");
  let phonePoints = [];
  const maxPhonePoints = 80;
  let pTime = 0;

  function resizePhoneCanvas() {
    const rect = phoneEegCanvas.getBoundingClientRect();
    if (rect.width > 0) {
      phoneEegCanvas.width = rect.width;
      phoneEegCanvas.height = rect.height || 45;
    }
  }
  window.addEventListener("resize", resizePhoneCanvas);
  setTimeout(resizePhoneCanvas, 200);

  function tickPhoneEEG() {
    pTime += 0.08;
    let signal = 0;
    if (currentIcuState === "calm") {
      signal = Math.sin(pTime * 2.2) * 5 + Math.sin(pTime * 4.5) * 2.5;
    } else if (currentIcuState === "warning") {
      signal = Math.sin(pTime * 6.5) * 11 + Math.sin(pTime * 12) * 6;
    } else {
      signal = (Math.random() - 0.5) * 28 + Math.sin(pTime * 18) * 14;
    }

    phonePoints.push(signal);
    if (phonePoints.length > maxPhonePoints) phonePoints.shift();

    if (phoneEegCanvas.width > 0 && phoneEegCanvas.height > 0) {
      pCtx.clearRect(0, 0, phoneEegCanvas.width, phoneEegCanvas.height);
      const midY = phoneEegCanvas.height / 2;
      const step = phoneEegCanvas.width / (maxPhonePoints - 1);

      pCtx.beginPath();
      pCtx.lineWidth = 1.8;
      pCtx.strokeStyle = currentIcuState === "calm" ? "#14B8A6" : (currentIcuState === "warning" ? "#F59E0B" : "#F43F5E");

      for (let i = 0; i < phonePoints.length; i++) {
        const x = i * step;
        const y = Math.max(2, Math.min(phoneEegCanvas.height - 2, midY + phonePoints[i]));
        if (i === 0) pCtx.moveTo(x, y);
        else pCtx.lineTo(x, y);
      }
      pCtx.stroke();
    }

    requestAnimationFrame(tickPhoneEEG);
  }
  requestAnimationFrame(tickPhoneEEG);
}

// ============================================================
// Interactive EEG Drawing Pad ("Neural Signal Canvas")
// ============================================================
const drawCanvas = document.getElementById("drawingPadCanvas");
let isDrawingPad = false;
let drawPoints = [];

if (drawCanvas) {
  const dCtx = drawCanvas.getContext("2d");
  
  function resizeDrawCanvas() {
    const rect = drawCanvas.getBoundingClientRect();
    drawCanvas.width = rect.width;
    drawCanvas.height = rect.height;
    clearDrawingPad();
  }
  window.addEventListener("resize", resizeDrawCanvas);
  setTimeout(resizeDrawCanvas, 100);

  function getMousePos(e) {
    const rect = drawCanvas.getBoundingClientRect();
    return {
      x: (e.clientX || e.touches?.[0]?.clientX) - rect.left,
      y: (e.clientY || e.touches?.[0]?.clientY) - rect.top
    };
  }

  function startDrawing(e) {
    isDrawingPad = true;
    const pos = getMousePos(e);
    drawPoints = [pos];
    dCtx.beginPath();
    dCtx.moveTo(pos.x, pos.y);
    dCtx.strokeStyle = "var(--teal-light)";
    dCtx.lineWidth = 2;
    dCtx.shadowBlur = 4;
    dCtx.shadowColor = "var(--teal-glow)";
  }

  function drawMove(e) {
    if (!isDrawingPad) return;
    e.preventDefault();
    const pos = getMousePos(e);
    drawPoints.push(pos);
    
    dCtx.lineTo(pos.x, pos.y);
    dCtx.stroke();
    
    // Analyze signal frequency & amplitude characteristics dynamically
    analyzeDrawnWave();
  }

  function stopDrawing() {
    isDrawingPad = false;
    dCtx.shadowBlur = 0;
    
    // Auto-predict based on drawn signal
    triggerPredict();
  }

  drawCanvas.addEventListener("mousedown", startDrawing);
  drawCanvas.addEventListener("mousemove", drawMove);
  drawCanvas.addEventListener("mouseup", stopDrawing);
  drawCanvas.addEventListener("mouseleave", stopDrawing);

  drawCanvas.addEventListener("touchstart", startDrawing, { passive: false });
  drawCanvas.addEventListener("touchmove", drawMove, { passive: false });
  drawCanvas.addEventListener("touchend", stopDrawing);
}

function clearDrawingPad(e) {
  if (e) e.stopPropagation();
  const dCtx = drawCanvas.getContext("2d");
  dCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawPoints = [];
  
  // Draw centerline reference
  dCtx.beginPath();
  dCtx.moveTo(0, drawCanvas.height / 2);
  dCtx.lineTo(drawCanvas.width, drawCanvas.height / 2);
  dCtx.strokeStyle = "rgba(255,255,255,0.05)";
  dCtx.setLineDash([5, 5]);
  dCtx.stroke();
  dCtx.setLineDash([]);
}

function analyzeDrawnWave() {
  if (drawPoints.length < 5) return;
  
  const h = drawCanvas.height;
  const mid = h / 2;
  
  // Calculate amplitude, variance, and zero crossings (proxy for frequency)
  let maxAmp = 0;
  let crossings = 0;
  let prevDiff = 0;
  
  drawPoints.forEach((p, idx) => {
    const diff = p.y - mid;
    if (Math.abs(diff) > maxAmp) {
      maxAmp = Math.abs(diff);
    }
    
    // Zero crossing check
    if (idx > 0) {
      if ((diff >= 0 && prevDiff < 0) || (diff < 0 && prevDiff >= 0)) {
        crossings++;
      }
    }
    prevDiff = diff;
  });

  const normAmp = Math.min(1.0, maxAmp / mid); // Normalize to 0-1
  const freqRatio = Math.min(1.0, crossings / (drawPoints.length * 0.15));

  // Map drawn properties to vote_* sliders to change the live prediction
  if (currentMode === "as_trained") {
    // If drawing spikes/waves, map to Seizure/LPD/GPD votes
    const seizureInput = document.querySelector('input[name="seizure_vote"]');
    const otherInput = document.querySelector('input[name="other_vote"]');
    const offsetInput = document.querySelector('input[name="eeg_label_offset_seconds"]');
    
    if (seizureInput && otherInput && offsetInput) {
      if (normAmp > 0.6 && freqRatio > 0.4) {
        seizureInput.value = (normAmp * 0.9).toFixed(2);
        otherInput.value = (1 - normAmp * 0.9).toFixed(2);
      } else {
        seizureInput.value = (normAmp * 0.2).toFixed(2);
        otherInput.value = (1 - normAmp * 0.2).toFixed(2);
      }
      
      // Update slider display labels
      updateSliderVal("seizure_vote", seizureInput.value);
      updateSliderVal("other_vote", otherInput.value);
      
      // Set offset based on crossings
      offsetInput.value = Math.floor(freqRatio * 1500);
      updateSliderVal("eeg_label_offset_seconds", offsetInput.value);
    }
  } else {
    // Leakage-corrected mode: map to standard EEG offsets and fold characteristics
    const offsetInput = document.querySelector('input[name="eeg_label_offset_seconds"]');
    const specOffsetInput = document.querySelector('input[name="spectrogram_label_offset_seconds"]');
    
    if (offsetInput && specOffsetInput) {
      offsetInput.value = Math.floor(normAmp * 1800);
      specOffsetInput.value = Math.floor(freqRatio * 1200);
      
      updateSliderVal("eeg_label_offset_seconds", offsetInput.value);
      updateSliderVal("spectrogram_label_offset_seconds", specOffsetInput.value);
    }
  }
}

// ============================================================
// Navigation: Tabs toggles
// ============================================================
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.toLowerCase().includes(tabName));
  });
  
  document.querySelectorAll(".sim-tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
  
  if (tabName === "replay") {
    resizeReplayCanvas();
    renderCurrentReplayFrame();
  } else if (tabName === "icu") {
    eegChannels.forEach((chan) => chan.resize());
    if (specCanvas) resizeSpec();
    if (syncWebCanvas) resizeSyncWeb();
  } else if (tabName === "portal") {
    if (drawCanvas) resizeDrawCanvas();
  }
}

// ============================================================
// Pipeline State Toggle
// ============================================================
function showToast(message, type = "teal") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.style.background = "var(--surface-card)";
  toast.style.border = `1px solid var(--border-active)`;
  toast.style.borderLeft = `4px solid var(--${type}-light)`;
  toast.style.color = "var(--text)";
  toast.style.padding = "0.75rem 1.25rem";
  toast.style.borderRadius = "8px";
  toast.style.boxShadow = "0 10px 25px rgba(0,0,0,0.5)";
  toast.style.fontSize = "0.85rem";
  toast.style.fontWeight = "700";
  toast.style.backdropFilter = "var(--glass-blur)";
  toast.style.transition = "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)";
  toast.style.transform = "translateX(120px)";
  toast.style.opacity = "0";
  toast.textContent = message;
  
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.transform = "translateX(0)";
    toast.style.opacity = "1";
  }, 50);
  
  setTimeout(() => {
    toast.style.transform = "translateX(120px)";
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

function setPipelineMode(mode) {
  currentMode = mode;
  const btnAsTrained = document.getElementById("btnAsTrained");
  const btnCorrected = document.getElementById("btnCorrected");
  if (btnAsTrained) btnAsTrained.classList.toggle("active", mode === "as_trained");
  if (btnCorrected) btnCorrected.classList.toggle("active", mode === "leakage_corrected");
  
  loadModelInfo();
  generateFeatureSliders();
  
  const label = mode === "as_trained" ? "Leakage Demo (98.8% Artifact)" : "Leakage-Corrected Mode (Honest Default)";
  showToast(`Active Pipeline: ${label}`, mode === "as_trained" ? "coral" : "teal");
  
  const target = document.getElementById("simulators");
  if (target) {
    target.scrollIntoView({ behavior: "smooth" });
  }
}

// ============================================================
// Portal: Load Samples and Patient Cards
// ============================================================
async function loadSamples() {
  const listEl = document.getElementById("patientList");
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) throw new Error("Could not load samples");
    sampleRecords = await res.json();
    
    listEl.innerHTML = "";
    sampleRecords.forEach((patient, idx) => {
      const card = document.createElement("div");
      card.className = "patient-card" + (idx === 0 ? " active" : "");
      card.innerHTML = `
        <div>Patient #${patient.patient_id || idx}</div>
        <div class="patient-meta">consensus: ${patient.expert_consensus}</div>
      `;
      card.addEventListener("click", () => {
        listEl.querySelectorAll(".patient-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        activePatient = patient;
        fillFormWithPatient(patient);
      });
      listEl.appendChild(card);
    });
    
    if (sampleRecords.length > 0) {
      activePatient = sampleRecords[0];
      fillFormWithPatient(activePatient);
    }
  } catch (err) {
    listEl.innerHTML = `<div class="patient-meta" style="text-align:center; padding: 1rem;">Could not connect to backend</div>`;
    console.error(err);
  }
}

// ============================================================
// Portal: Feature sliders generation
// ============================================================
const featureMeta = {
  eeg_id: { min: 0, max: 4000000000, step: 100000, def: 2000000000 },
  eeg_sub_id: { min: 0, max: 20, step: 1, def: 0 },
  eeg_label_offset_seconds: { min: 0, max: 2000, step: 1, def: 0 },
  spectrogram_id: { min: 0, max: 2000000000, step: 100000, def: 1000000000 },
  spectrogram_sub_id: { min: 0, max: 30, step: 1, def: 0 },
  spectrogram_label_offset_seconds: { min: 0, max: 2000, step: 1, def: 0 },
  patient_id: { min: 0, max: 65000, step: 1, def: 30000 },
  fold: { min: 0, max: 4, step: 1, def: 0 },
  seizure_vote: { min: 0, max: 1, step: 0.01, def: 0 },
  lpd_vote: { min: 0, max: 1, step: 0.01, def: 0 },
  gpd_vote: { min: 0, max: 1, step: 0.01, def: 0 },
  lrda_vote: { min: 0, max: 1, step: 0.01, def: 0 },
  grda_vote: { min: 0, max: 1, step: 0.01, def: 0 },
  other_vote: { min: 0, max: 1, step: 0.01, def: 0 },
};

function generateFeatureSliders() {
  const form = document.getElementById("slidersForm");
  form.innerHTML = "";
  
  let columns = [];
  if (currentMode === "as_trained") {
    columns = [
      "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
      "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds",
      "patient_id", "seizure_vote", "lpd_vote", "gpd_vote",
      "lrda_vote", "grda_vote", "other_vote", "fold"
    ];
  } else {
    columns = [
      "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
      "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds",
      "patient_id", "fold"
    ];
  }
  
  currentFeatureColumns = columns;
  
  columns.forEach((col) => {
    const meta = featureMeta[col] || { min: 0, max: 100, step: 1, def: 0 };
    const labelRow = document.createElement("div");
    labelRow.className = "feature-slider-group";
    labelRow.innerHTML = `
      <div class="feature-slider-label">
        <span class="feature-name">${col}</span>
        <span class="feature-val" id="val-${col}">${meta.def}</span>
      </div>
      <input type="range" name="${col}" min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.def}" oninput="updateSliderVal('${col}', this.value)">
    `;
    form.appendChild(labelRow);
  });
  
  if (activePatient) {
    fillFormWithPatient(activePatient);
  }
}

function updateSliderVal(col, val) {
  const valEl = document.getElementById(`val-${col}`);
  if (valEl) valEl.textContent = parseFloat(val).toFixed(col.includes("vote") ? 2 : 0);
}

function fillFormWithPatient(patient) {
  currentFeatureColumns.forEach((col) => {
    const val = patient[col] !== undefined ? patient[col] : (featureMeta[col]?.def || 0);
    const input = document.querySelector(`input[name="${col}"]`);
    if (input) {
      input.value = val;
      updateSliderVal(col, val);
    }
  });
}

function resetForm() {
  if (activePatient) {
    fillFormWithPatient(activePatient);
  } else {
    currentFeatureColumns.forEach((col) => {
      const def = featureMeta[col]?.def || 0;
      const input = document.querySelector(`input[name="${col}"]`);
      if (input) {
        input.value = def;
        updateSliderVal(col, def);
      }
    });
  }
  if (drawCanvas) clearDrawingPad();
}

// ============================================================
// Portal: Run Prediction & fetch Grad-CAM / Attribution Grads
// ============================================================
async function triggerPredict() {
  const classValEl = document.getElementById("predictedClassVal");
  classValEl.textContent = "Calculating...";
  
  const features = {};
  currentFeatureColumns.forEach((col) => {
    const input = document.querySelector(`input[name="${col}"]`);
    if (input) {
      features[col] = parseFloat(input.value);
    }
  });
  
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: currentMode,
        features: features
      })
    });
    
    if (!res.ok) throw new Error("Prediction calculation failed");
    const data = await res.json();
    
    classValEl.textContent = data.predicted_class;
    
    const probContainer = document.getElementById("probBarContainer");
    probContainer.innerHTML = "";
    
    Object.entries(data.probabilities).forEach(([cls, prob]) => {
      const isTop = cls === data.predicted_class;
      const row = document.createElement("div");
      row.className = "prob-bar-row";
      row.innerHTML = `
        <span class="prob-bar-label">${cls}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill${isTop ? ' top-match' : ''}" style="width: ${(prob * 100).toFixed(1)}%;"></div>
        </div>
        <span class="prob-bar-pct">${(prob * 100).toFixed(0)}%</span>
      `;
      probContainer.appendChild(row);
    });
    
    const gradcamContainer = document.getElementById("gradcamContainer");
    gradcamContainer.innerHTML = "";
    
    const weights = data.gradcam_weights || [];
    weights.forEach((w) => {
      const bar = document.createElement("div");
      bar.className = "gradcam-bar" + (w > 0.45 ? " active" : "");
      bar.style.height = `${(w * 100).toFixed(1)}%`;
      gradcamContainer.appendChild(bar);
    });
    
    const featImpContainer = document.getElementById("featImportanceContainer");
    featImpContainer.innerHTML = "";
    
    Object.entries(data.feature_importance)
      .sort((a, b) => b[1] - a[1])
      .forEach(([col, val]) => {
        const row = document.createElement("div");
        row.className = "feat-imp-row";
        row.innerHTML = `
          <span class="feat-imp-name">${col}</span>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <div class="feat-imp-track"><div class="feat-imp-fill" style="width: ${(val * 100).toFixed(1)}%;"></div></div>
            <span style="font-family:'Fira Code'; font-size:0.75rem; color:var(--teal-light); width:32px; text-align:right;">${val.toFixed(2)}</span>
          </div>
        `;
        featImpContainer.appendChild(row);
      });
      
  } catch (err) {
    classValEl.textContent = "Error";
    console.error(err);
  }
}

// ============================================================
// Performance Evaluation: Metrics and Confusion Matrix
// ============================================================
function renderMetricsData(data) {
  if (!data) return;
  
  const mstmEl = document.getElementById("txtMstmAcc");
  const fillMstmEl = document.getElementById("fillMstmAcc");
  if (mstmEl && fillMstmEl) {
    mstmEl.textContent = `${(data.mstm_accuracy * 100).toFixed(1)}%`;
    fillMstmEl.style.width = `${(data.mstm_accuracy * 100).toFixed(1)}%`;
  }
  
  const sbamEl = document.getElementById("txtSbamAcc");
  const fillSbamEl = document.getElementById("fillSbamAcc");
  if (sbamEl && fillSbamEl) {
    sbamEl.textContent = `${(data.sbam_accuracy * 100).toFixed(1)}%`;
    fillSbamEl.style.width = `${(data.sbam_accuracy * 100).toFixed(1)}%`;
  }
  
  const fusionEl = document.getElementById("txtFusionAcc");
  const fillFusionEl = document.getElementById("fillFusionAcc");
  if (fusionEl && fillFusionEl) {
    fusionEl.textContent = `${(data.fusion_accuracy * 100).toFixed(1)}%`;
    fillFusionEl.style.width = `${(data.fusion_accuracy * 100).toFixed(1)}%`;
  }
  
  const cardFusionEl = document.getElementById("txtFusionCardAcc");
  if (cardFusionEl) {
    cardFusionEl.textContent = `${(data.fusion_accuracy * 100).toFixed(1)}%`;
  }
  
  const tableBody = document.getElementById("metricsTable")?.querySelector("tbody");
  if (tableBody && data.classification_report && data.classes) {
    tableBody.innerHTML = "";
    
    const classes = data.classes;
    classes.forEach((cls) => {
      const clsMetrics = data.classification_report[cls] || { precision: 0, recall: 0, "f1-score": 0 };
      const row = document.createElement("tr");
      row.innerHTML = `
        <td class="class-name">${cls}</td>
        <td>${(clsMetrics.precision || 0).toFixed(3)}</td>
        <td>${(clsMetrics.recall || 0).toFixed(3)}</td>
        <td>${(clsMetrics["f1-score"] || 0).toFixed(3)}</td>
      `;
      tableBody.appendChild(row);
    });
    
    const macroAvg = data.classification_report["macro avg"] || { precision: 0, recall: 0, "f1-score": 0 };
    const avgRow = document.createElement("tr");
    avgRow.style.fontWeight = "bold";
    avgRow.style.borderTop = "2px solid var(--border-active)";
    avgRow.innerHTML = `
      <td class="class-name">Macro Average</td>
      <td>${(macroAvg.precision || 0).toFixed(3)}</td>
      <td>${(macroAvg.recall || 0).toFixed(3)}</td>
      <td>${(macroAvg["f1-score"] || 0).toFixed(3)}</td>
    `;
    tableBody.appendChild(avgRow);
  }
  
  const cmGrid = document.getElementById("cmGridContainer");
  if (cmGrid && data.classes && data.confusion_matrix) {
    cmGrid.innerHTML = "";
    const classes = data.classes;
    
    const headerRow = document.createElement("div");
    headerRow.className = "cm-row";
    headerRow.innerHTML = `<div class="cm-cell cm-label-header"></div>`;
    classes.forEach((cls) => {
      headerRow.innerHTML += `<div class="cm-cell cm-label-header" title="${cls}">${cls.substring(0, 3)}</div>`;
    });
    cmGrid.appendChild(headerRow);
    
    const matrix = data.confusion_matrix || [];
    classes.forEach((cls, rIdx) => {
      const row = document.createElement("div");
      row.className = "cm-row";
      row.innerHTML = `<div class="cm-cell cm-label-row">${cls}</div>`;
      
      const rowValues = matrix[rIdx] || [];
      const rowSum = rowValues.reduce((a, b) => a + b, 0) || 1;
      
      classes.forEach((_, cIdx) => {
        const val = rowValues[cIdx] !== undefined ? rowValues[cIdx] : 0;
        const fraction = val / rowSum;
        
        let cellBg = `rgba(30, 43, 75, ${0.15 + fraction * 0.85})`;
        if (rIdx === cIdx) {
          cellBg = `rgba(20, 184, 166, ${0.25 + fraction * 0.75})`;
        }
        
        row.innerHTML += `
          <div class="cm-cell cm-val-cell" style="background: ${cellBg};" title="True: ${cls}, Pred: ${classes[cIdx]} = ${val}">
            ${val}
          </div>
        `;
      });
      cmGrid.appendChild(row);
    });
  }
}

async function loadModelInfo() {
  const coldStartBanner = document.getElementById("cold-start-banner");
  
  // 1. Immediately render cached metrics so page is never blank
  if (CACHED_METRICS[currentMode]) {
    renderMetricsData(CACHED_METRICS[currentMode]);
  }
  
  // 2. Set timeout to show cold start banner if backend takes > 1200ms
  let isFetched = false;
  const coldStartTimer = setTimeout(() => {
    if (!isFetched && coldStartBanner) {
      coldStartBanner.style.display = "block";
    }
  }, 1200);

  try {
    const res = await fetch(`${API_BASE}/model-info?mode=${currentMode}`, { signal: AbortSignal.timeout(15000) });
    isFetched = true;
    clearTimeout(coldStartTimer);
    if (coldStartBanner) coldStartBanner.style.display = "none";
    
    if (!res.ok) throw new Error("Could not load model info from backend");
    const data = await res.json();
    renderMetricsData(data);
  } catch (err) {
    isFetched = true;
    clearTimeout(coldStartTimer);
    console.warn("Backend unavailable or cold-starting. Fallback cache is active.", err);
    // Keep showing cold start banner if on cloud
    if (!isLocalhost && coldStartBanner) {
      coldStartBanner.style.display = "block";
    }
  }
}

// ============================================================
// CHB-MIT Continuous Scalp EEG Replay Simulator Engine (Task 17)
// ============================================================
let replayPatientId = "chb01";
let replayData = null;
let replayFrameIndex = 0;
let replayIsPlaying = false;
let replaySpeed = 1;
let replayTimer = null;
let replayRiskHistory = [];

const CHB_FALLBACK_CHANNELS = [
  "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
  "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
  "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
  "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
  "FZ-CZ", "CZ-PZ"
];

function generateSimulatedReplayFrames(patientId) {
  const frames = [];
  let consec = 0;
  for (let i = 0; i < 120; i++) {
    const timeSec = i * 2.5 + 4300;
    const isPreictal = i >= 30;
    const isSph = i >= 90 && i < 110;
    const isSeiz = i >= 110;
    const secToSeiz = Math.max(0, (110 - i) * 25);
    
    let prob = 0.05 + 0.03 * Math.sin(i * 0.3);
    if (isPreictal) prob = 0.45 + 0.5 * ((i - 30) / 80);
    if (isSeiz) prob = 0.98;
    prob = Math.min(0.99, Math.max(0.02, prob));
    
    if (prob >= 0.5) consec++; else consec = 0;
    const alarm = consec >= 3 && i < 110;
    
    const signals = [];
    for (let c = 0; c < 18; c++) {
      const wave = [];
      const amp = isSeiz ? 1.8 : (isPreictal ? 0.9 : 0.4);
      const freq = isSeiz ? 0.6 : (isPreictal ? 0.35 : 0.2);
      for (let s = 0; s < 32; s++) {
        wave.push(Math.sin(s * freq + c * 0.5 + i * 0.4) * amp + (Math.random() - 0.5) * 0.15);
      }
      signals.push(wave);
    }
    
    frames.push({
      step: i,
      timestamp_sec: timeSec,
      time_str: new Date(timeSec * 1000).toISOString().substr(11, 8),
      is_preictal: isPreictal,
      is_sph: isSph,
      is_seizure_onset: isSeiz,
      seconds_to_seizure: secToSeiz,
      risk_probability: Math.round(prob * 1000) / 1000,
      alarm_triggered: alarm,
      consecutive_high_windows: consec,
      signals: signals
    });
  }
  return {
    patient_id: patientId,
    channels: CHB_FALLBACK_CHANNELS,
    total_frames: frames.length,
    frames: frames
  };
}

async function loadPatientReplay(patientId) {
  replayPatientId = patientId;
  replayFrameIndex = 0;
  replayRiskHistory = [];
  
  try {
    const res = await fetch(`${API_BASE}/api/chbmit/replay-stream/${patientId}?start_frame=0&count=120`, { signal: AbortSignal.timeout(6000) });
    if (!res.ok) throw new Error("Could not fetch replay data");
    replayData = await res.json();
  } catch (e) {
    console.warn("Using high-fidelity local fallback replay simulation for " + patientId, e);
    replayData = generateSimulatedReplayFrames(patientId);
  }
  
  const slider = document.getElementById("replayTimelineSlider");
  if (slider && replayData && replayData.frames) {
    slider.max = replayData.frames.length - 1;
    slider.value = 0;
  }
  
  loadExplainabilityData(patientId);
  resizeReplayCanvas();
  renderCurrentReplayFrame();
}

async function loadExplainabilityData(patientId) {
  try {
    const res = await fetch(`${API_BASE}/api/chbmit/explainability/${patientId}`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const data = await res.json();
    renderTopChannels(data.ranked_channels);
    const sanityEl = document.getElementById("replaySanityPass");
    if (sanityEl && data.sanity_check_correlation !== undefined) {
      sanityEl.textContent = `[PASSED] r = ${data.sanity_check_correlation.toFixed(3)}`;
    }
  } catch (e) {
    // Keep verified static top channels
  }
}

function renderTopChannels(channels) {
  const container = document.getElementById("replayTopChannelsList");
  if (!container || !channels || channels.length === 0) return;
  const top4 = channels.slice(0, 4);
  container.innerHTML = top4.map(ch => `
    <div class="ch-bar-row">
      <span class="ch-name">${ch.channel}</span>
      <div class="ch-track"><div class="ch-fill" style="width:${(ch.attention_weight * 100).toFixed(1)}%;"></div></div>
      <span class="ch-pct">${(ch.attention_weight * 100).toFixed(1)}%</span>
    </div>
  `).join("");
}

function resizeReplayCanvas() {
  const canvas = document.getElementById("replayEegCanvas");
  if (!canvas || !canvas.parentElement) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = (rect.width || 800) * dpr;
  canvas.height = 520 * dpr;
}

function renderCurrentReplayFrame() {
  if (!replayData || !replayData.frames || replayData.frames.length === 0) return;
  const frame = replayData.frames[replayFrameIndex];
  if (!frame) return;
  
  // 1. Draw 18-channel EEG Montage
  const canvas = document.getElementById("replayEegCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);
    
    const channels = replayData.channels || CHB_FALLBACK_CHANNELS;
    const numCh = channels.length;
    const slotH = h / numCh;
    const leftMargin = 72;
    const waveW = w - leftMargin - 15;
    
    // Waveform line color depending on state
    let waveColor = "#14B8A6"; // teal
    if (frame.is_seizure_onset) {
      waveColor = "#F43F5E"; // coral red
    } else if (frame.is_sph) {
      waveColor = "#FB7185"; // rose
    } else if (frame.is_preictal) {
      waveColor = "#F59E0B"; // amber
    }
    
    for (let c = 0; c < numCh; c++) {
      const centerY = c * slotH + slotH * 0.5;
      
      // Channel Separator / Baseline grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(leftMargin, centerY);
      ctx.lineTo(w - 10, centerY);
      ctx.stroke();
      
      // Channel Name Label
      ctx.font = "600 11px 'Fira Code', monospace";
      ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(channels[c], 10, centerY);
      
      // Draw Waveform for this channel
      const raw = frame.signals ? frame.signals[c] : null;
      if (raw && raw.length > 0) {
        ctx.strokeStyle = waveColor;
        ctx.lineWidth = frame.is_seizure_onset ? 1.8 : 1.2;
        ctx.beginPath();
        const stepX = waveW / (raw.length - 1);
        for (let s = 0; s < raw.length; s++) {
          const x = leftMargin + s * stepX;
          const y = centerY - raw[s] * (slotH * 0.42);
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }
    ctx.restore();
  }
  
  // 2. Update Status Badges
  const stateBadge = document.getElementById("replay-state-badge");
  if (stateBadge) {
    if (frame.is_seizure_onset) {
      stateBadge.className = "badge-status badge-seizure";
      stateBadge.textContent = "⚡ CLINICAL SEIZURE ONSET";
    } else if (frame.is_sph) {
      stateBadge.className = "badge-status badge-sph";
      stateBadge.textContent = "⚠️ SPH (5m PREDICTIVE HORIZON)";
    } else if (frame.is_preictal) {
      stateBadge.className = "badge-status badge-preictal";
      stateBadge.textContent = "PREICTAL ANOMALY (30m)";
    } else {
      stateBadge.className = "badge-status badge-interictal";
      stateBadge.textContent = "INTERICTAL BASELINE";
    }
  }
  
  const timeBadge = document.getElementById("replay-time-badge");
  if (timeBadge && frame.time_str) {
    timeBadge.textContent = `T = ${frame.time_str}`;
  }
  
  // 3. Update Risk Gauge
  const riskPct = Math.round(frame.risk_probability * 1000) / 10;
  const riskNumEl = document.getElementById("replayRiskPercent");
  if (riskNumEl) {
    riskNumEl.textContent = `${riskPct.toFixed(1)}%`;
    riskNumEl.style.color = frame.risk_probability >= 0.5 ? "var(--coral-light)" : (frame.risk_probability >= 0.25 ? "var(--amber-light)" : "var(--teal-light)");
  }
  const riskFill = document.getElementById("replayRiskFill");
  if (riskFill) {
    riskFill.style.width = `${Math.min(100, Math.max(2, riskPct))}%`;
  }
  
  // 4. Update Rolling Risk History & Mini Chart
  replayRiskHistory.push(frame.risk_probability);
  if (replayRiskHistory.length > 40) replayRiskHistory.shift();
  renderRiskMiniChart();
  
  // 5. Update Clinical Alarm State
  const led = document.getElementById("alarmLedStatus");
  const banner = document.getElementById("alarmAlertBanner");
  const countEl = document.getElementById("consecutiveHighCount");
  if (countEl) countEl.textContent = `${frame.consecutive_high_windows || 0} / 3 windows`;
  
  if (frame.alarm_triggered || (frame.consecutive_high_windows >= 3 && !frame.is_seizure_onset)) {
    if (led) { led.className = "alarm-led led-active"; led.textContent = "ALARM ACTIVE"; }
    if (banner) banner.style.display = "flex";
  } else {
    if (led) { led.className = "alarm-led led-idle"; led.textContent = "IDLE"; }
    if (banner) banner.style.display = "none";
  }
  
  // 6. Update Seizure Countdown Clock
  const clockEl = document.getElementById("seizureCountdownClock");
  if (clockEl) {
    if (frame.is_seizure_onset) {
      clockEl.textContent = "ONSET NOW";
      clockEl.style.color = "var(--coral-light)";
    } else {
      const sec = Math.round(frame.seconds_to_seizure);
      const mm = Math.floor(sec / 60);
      const ss = sec % 60;
      clockEl.textContent = `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
      clockEl.style.color = sec <= 300 ? "var(--coral-light)" : (sec <= 1800 ? "var(--amber-light)" : "var(--teal-light)");
    }
  }
  
  // 7. Update Scrubber Slider
  const slider = document.getElementById("replayTimelineSlider");
  if (slider) slider.value = replayFrameIndex;
}

function renderRiskMiniChart() {
  const chartCanvas = document.getElementById("replayRiskChart");
  if (!chartCanvas || replayRiskHistory.length < 2) return;
  const ctx = chartCanvas.getContext("2d");
  const w = chartCanvas.width;
  const h = chartCanvas.height;
  
  ctx.clearRect(0, 0, w, h);
  
  // Dotted Threshold 0.5 Line
  const thY = h * 0.5;
  ctx.strokeStyle = "rgba(244, 63, 94, 0.4)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, thY);
  ctx.lineTo(w, thY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Risk Curve
  ctx.strokeStyle = "#14B8A6";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const stepX = w / (replayRiskHistory.length - 1);
  for (let i = 0; i < replayRiskHistory.length; i++) {
    const x = i * stepX;
    const y = h - replayRiskHistory[i] * h * 0.9 - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Area fill
  ctx.lineTo((replayRiskHistory.length - 1) * stepX, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = "rgba(20, 184, 166, 0.08)";
  ctx.fill();
}

function toggleReplayPlay() {
  replayIsPlaying = !replayIsPlaying;
  const btn = document.getElementById("btnReplayPlay");
  if (btn) btn.textContent = replayIsPlaying ? "⏸ Pause Stream" : "▶ Play Stream";
  
  if (replayIsPlaying) {
    runReplayLoop();
  } else {
    clearTimeout(replayTimer);
  }
}

function runReplayLoop() {
  if (!replayIsPlaying) return;
  stepReplayFrame(1);
  const intervalMs = Math.round(1000 / replaySpeed);
  replayTimer = setTimeout(runReplayLoop, intervalMs);
}

function stepReplayFrame(delta) {
  if (!replayData || !replayData.frames) return;
  replayFrameIndex += delta;
  if (replayFrameIndex >= replayData.frames.length) {
    replayFrameIndex = 0;
  } else if (replayFrameIndex < 0) {
    replayFrameIndex = 0;
  }
  renderCurrentReplayFrame();
}

function resetReplay() {
  replayIsPlaying = false;
  clearTimeout(replayTimer);
  const btn = document.getElementById("btnReplayPlay");
  if (btn) btn.textContent = "▶ Play Stream";
  replayFrameIndex = 0;
  replayRiskHistory = [];
  renderCurrentReplayFrame();
}

function setReplaySpeed(speed) {
  replaySpeed = speed;
  [1, 2, 5].forEach(s => {
    const el = document.getElementById(`btnSpeed${s}`);
    if (el) el.classList.toggle("active", s === speed);
  });
}

function onTimelineScrub(val) {
  replayFrameIndex = parseInt(val, 10);
  renderCurrentReplayFrame();
}

// ============================================================
// Initialization on page load
// ============================================================
window.addEventListener("DOMContentLoaded", () => {
  generateFeatureSliders();
  loadSamples();
  loadModelInfo();
  loadPatientReplay("chb01");
});

window.addEventListener("resize", () => {
  resizeReplayCanvas();
  renderCurrentReplayFrame();
});
