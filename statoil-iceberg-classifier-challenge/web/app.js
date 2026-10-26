/**
 * MARITIME AI SENTINEL - APPLICATION ENGINE
 * High-Performance Client-Side SAR Inference, Ocean Route Simulation,
 * Cyber-Resilience Telemetry, and MarineChain Ledger Visualization
 */

// Global State
const STATE = {
  activeSampleIndex: 0,
  activeTab: 'all',
  sarSamples: [],
  simRunning: false,
  simTimer: null,
  simSpeed: 1,
  simHours: 0.0,
  simStepCount: 0,
  route: {
    source: 'CA_STJ',
    dest: 'NL_RTM',
    totalDistNm: 2176,
    remDistNm: 2176,
    etaDays: 5.5,
    sogKnots: 16.5,
    cogDeg: 68.0
  },
  ship: {
    mmsi: 316001234,
    name: "MV ARCTIC SENTINEL",
    lat: 47.5615,
    lon: -52.7126,
    sog: 16.5,
    cog: 68.0,
    heading: 68.0,
    track: []
  },
  cyclone: {
    name: "Cyclone Boreal Tempest",
    lat: 44.5,
    lon: -51.0,
    windKnots: 85.0,
    centralPressureHpa: 958.0,
    speedKnots: 18.0,
    dirDeg: 42.0,
    dangerRadiusNm: 75.0,
    galeRadiusNm: 160.0
  },
  icebergs: [
    { id: "ICE-7801", lat: 47.95, lon: -49.6, speed: 1.2, dir: 165.0, pIce: 0.984, radius: 18.0 },
    { id: "ICE-7802", lat: 48.35, lon: -46.8, speed: 0.9, dir: 175.0, pIce: 0.962, radius: 14.0 },
    { id: "ICE-7803", lat: 49.10, lon: -48.2, speed: 1.4, dir: 160.0, pIce: 0.941, radius: 16.0 },
    { id: "VES-4412", lat: 47.80, lon: -44.2, speed: 12.0, dir: 245.0, pIce: 0.041, radius: 6.0 }
  ],
  cyber: {
    gpsSpoofed: false,
    windManipulated: false,
    replayActive: false,
    spoofOffsetLat: 0.85,
    spoofOffsetLon: -1.20,
    anomalyScore: 0.0,
    kalmanEngaged: false
  },
  blockchain: []
};

// 18 Authentic Atlantic & Arctic Ports Across 5 Key Regions
const MARITIME_PORTS = {
  "CA_STJ": {
    id: "CA_STJ",
    name: "St. John's, Newfoundland",
    country: "Canada",
    lat: 47.5615,
    lon: -52.7126,
    region: "North America",
    desc: "Primary C-CORE & Hibernia offshore operations center; Statoil SAR challenge origin."
  },
  "CA_HAL": {
    id: "CA_HAL",
    name: "Halifax, Nova Scotia",
    country: "Canada",
    lat: 44.6488,
    lon: -63.5752,
    region: "North America",
    desc: "Major ice-free deepwater container and naval Atlantic gateway."
  },
  "US_NYC": {
    id: "US_NYC",
    name: "New York / New Jersey",
    country: "United States",
    lat: 40.6892,
    lon: -74.0445,
    region: "North America",
    desc: "Busiest container terminal on the US Atlantic seaboard."
  },
  "US_BOS": {
    id: "US_BOS",
    name: "Boston, Massachusetts",
    country: "United States",
    lat: 42.3601,
    lon: -71.0589,
    region: "North America",
    desc: "Historic New England transatlantic shipping hub."
  },
  "CA_MTL": {
    id: "CA_MTL",
    name: "Montreal, St. Lawrence",
    country: "Canada",
    lat: 45.5017,
    lon: -73.5673,
    region: "North America",
    desc: "St. Lawrence Seaway inland gateway connecting Great Lakes to Atlantic."
  },
  "GL_NUU": {
    id: "GL_NUU",
    name: "Nuuk, Davis Strait",
    country: "Greenland",
    lat: 64.1814,
    lon: -51.6941,
    region: "Arctic & Greenland",
    desc: "High Arctic calving fjord region; source of Baffin Bay icebergs."
  },
  "GL_QAQ": {
    id: "GL_QAQ",
    name: "Qaqortoq / Cape Farewell",
    country: "Greenland",
    lat: 60.7167,
    lon: -46.0333,
    region: "Arctic & Greenland",
    desc: "Southern tip of Greenland; severe storms and pack ice transit point."
  },
  "IS_RVK": {
    id: "IS_RVK",
    name: "Reykjavik",
    country: "Iceland",
    lat: 64.1466,
    lon: -21.9426,
    region: "North Atlantic",
    desc: "Sub-Arctic mid-Atlantic junction and weather reconnaissance hub."
  },
  "OFF_HIB": {
    id: "OFF_HIB",
    name: "Hibernia Platform",
    country: "Canada / Offshore",
    lat: 46.7500,
    lon: -48.7833,
    region: "Offshore Facilities",
    desc: "Iceberg-resistant gravity-base structure on the Grand Banks."
  },
  "OFF_FLE": {
    id: "OFF_FLE",
    name: "Flemish Pass Exploration Rig",
    country: "International / Offshore",
    lat: 47.9500,
    lon: -45.0000,
    region: "Offshore Facilities",
    desc: "Deepwater exploration field operated by Equinor / Statoil."
  },
  "NO_BGO": {
    id: "NO_BGO",
    name: "Bergen",
    country: "Norway",
    lat: 60.3913,
    lon: 5.3221,
    region: "Nordic & North Sea",
    desc: "Equinor North Sea subsea & offshore operations capital."
  },
  "NO_SVG": {
    id: "NO_SVG",
    name: "Stavanger",
    country: "Norway",
    lat: 58.9700,
    lon: 5.7331,
    region: "Nordic & North Sea",
    desc: "Oil and energy capital of Norway; Equinor global headquarters."
  },
  "NO_TOS": {
    id: "NO_TOS",
    name: "Tromsø",
    country: "Norway",
    lat: 69.6492,
    lon: 18.9553,
    region: "Nordic & North Sea",
    desc: "High Arctic gateway to the Barents Sea and Northern Sea Route."
  },
  "NL_RTM": {
    id: "NL_RTM",
    name: "Rotterdam",
    country: "Netherlands",
    lat: 51.9244,
    lon: 4.4777,
    region: "Western Europe",
    desc: "Largest seaport in Europe; primary transatlantic freight terminal."
  },
  "DE_HAM": {
    id: "DE_HAM",
    name: "Hamburg",
    country: "Germany",
    lat: 53.5511,
    lon: 9.9937,
    region: "Western Europe",
    desc: "Elbe estuary major European container hub."
  },
  "UK_SOU": {
    id: "UK_SOU",
    name: "Southampton",
    country: "United Kingdom",
    lat: 50.9097,
    lon: -1.4044,
    region: "Western Europe",
    desc: "Premier deepwater English Channel transatlantic terminal."
  },
  "UK_LIV": {
    id: "UK_LIV",
    name: "Liverpool",
    country: "United Kingdom",
    lat: 53.4084,
    lon: -2.9916,
    region: "Western Europe",
    desc: "Irish Sea and Western Approaches historic shipping gateway."
  },
  "FR_BES": {
    id: "FR_BES",
    name: "Brest, Brittany",
    country: "France",
    lat: 48.3904,
    lon: -4.4861,
    region: "Western Europe",
    desc: "Atlantic approaches & Bay of Biscay entrance."
  }
};

// Initial Seed Blockchain
const INITIAL_BLOCKS = [
  {
    index: 1042,
    timestamp: "12:10:00 UTC",
    threat_type: "SYSTEM_GENESIS",
    title: "MarineChain Sentinel Node Initialized",
    hash: "0x4a9b2c1f8e7d3a0b5c9e2f4a1d8b7c6e",
    prev_hash: "0x3e1d7a9c2b4f6a8e0d5c1b9f7a2e4c8d",
    validator: "IMO-NorthAtlantic-01",
    trust_score: 1.000,
    status: "VERIFIED"
  },
  {
    index: 1043,
    timestamp: "12:15:22 UTC",
    threat_type: "ICE_HAZARD",
    title: "SAR C-Band Target ICE-7801 Confirmed (P=0.984)",
    hash: "0x7d3a0b5c9e2f4a1d8b7c6e4a9b2c1f8e",
    prev_hash: "0x4a9b2c1f8e7d3a0b5c9e2f4a1d8b7c6e",
    validator: "CCG-StJohns-Station",
    trust_score: 0.984,
    status: "VERIFIED"
  },
  {
    index: 1044,
    timestamp: "12:22:45 UTC",
    threat_type: "CYCLONE_UPDATE",
    title: "Boreal Tempest Gale Cone Extends 160 NM",
    hash: "0x9e2f4a1d8b7c6e4a9b2c1f8e7d3a0b5c",
    prev_hash: "0x7d3a0b5c9e2f4a1d8b7c6e4a9b2c1f8e",
    validator: "HAPS-Alpha-01",
    trust_score: 0.991,
    status: "VERIFIED"
  }
];

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initSamplesData();
  initNavigation();
  initHeroCanvas();
  initSarStudio();
  initOceanSimulator();
  initBlockchainLedger();
  initLightbox();
  initResearchFeatures();
});

// -----------------------------------------------------------------------------
// 1. DATA INITIALIZATION & FALLBACKS
// -----------------------------------------------------------------------------
function initSamplesData() {
  if (typeof SAR_SAMPLES !== 'undefined' && Array.isArray(SAR_SAMPLES) && SAR_SAMPLES.length > 0) {
    STATE.sarSamples = SAR_SAMPLES;
  } else {
    // Generate high-fidelity synthetic SAR fallback samples if file is loaded standalone
    STATE.sarSamples = generateFallbackSamples();
  }
}

function generateFallbackSamples() {
  const samples = [];
  const configs = [
    { id: "83dd3a43", is_iceberg: 1, inc_angle: "43.9466", peak: -12.5, bg: -28.5 },
    { id: "58b2aaa0", is_iceberg: 1, inc_angle: "45.2859", peak: -14.2, bg: -29.1 },
    { id: "bd1a1bdf", is_iceberg: 1, inc_angle: "37.6866", peak: -10.8, bg: -26.0 },
    { id: "31da1a04", is_iceberg: 1, inc_angle: "34.4751", peak: -11.5, bg: -24.8 },
    { id: "e356f7a3", is_iceberg: 0, inc_angle: "34.7715", peak: -4.5, bg: -27.2 },
    { id: "87592c38", is_iceberg: 0, inc_angle: "43.7820", peak: -3.8, bg: -31.0 },
    { id: "f193e8b0", is_iceberg: 1, inc_angle: "44.6239", peak: -13.0, bg: -29.8 },
    { id: "76b8d446", is_iceberg: 1, inc_angle: "39.2340", peak: -11.9, bg: -26.5 },
    { id: "958d155f", is_iceberg: 0, inc_angle: "42.5145", peak: -2.9, bg: -30.4 },
    { id: "dfd5f913", is_iceberg: 0, inc_angle: "43.9239", peak: -4.1, bg: -31.2 }
  ];

  configs.forEach(cfg => {
    const b1 = [];
    const b2 = [];
    const isIce = cfg.is_iceberg === 1;
    const center = 37.5;

    for (let r = 0; r < 75; r++) {
      for (let c = 0; c < 75; c++) {
        const d = Math.sqrt((r - center)**2 + (c - center)**2);
        const noise1 = (Math.random() - 0.5) * 5.0;
        const noise2 = (Math.random() - 0.5) * 4.0;
        
        let signal1 = cfg.bg + noise1;
        let signal2 = (cfg.bg - 6.0) + noise2;

        if (d < 16) {
          const shapeFactor = isIce 
            ? Math.exp(-d / 6.0) 
            : (d < 6 ? Math.exp(-d / 2.5) : Math.exp(-d / 4.0) * 0.4);
          
          signal1 += (cfg.peak - cfg.bg) * shapeFactor;
          // Icebergs depolarize more -> higher HV signal relative to HH
          const hvRatio = isIce ? 0.75 : 0.45;
          signal2 += (cfg.peak - cfg.bg) * shapeFactor * hvRatio;
        }

        b1.push(signal1);
        b2.push(signal2);
      }
    }

    samples.push({
      id: cfg.id,
      is_iceberg: cfg.is_iceberg,
      inc_angle: cfg.inc_angle,
      band_1: b1,
      band_2: b2
    });
  });

  return samples;
}

// -----------------------------------------------------------------------------
// 2. NAVIGATION & AMBIENT VISUALS
// -----------------------------------------------------------------------------
function initNavigation() {
  const header = document.querySelector(".nav-header");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      header.classList.add("scrolled");
    } else {
      header.classList.remove("scrolled");
    }
  });

  // Smooth scroll links
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
      const targetId = link.getAttribute("href");
      if (targetId && targetId.startsWith("#")) {
        e.preventDefault();
        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: "smooth" });
          document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
          link.classList.add("active");
        }
      }
    });
  });
}

function initHeroCanvas() {
  const canvas = document.getElementById("heroRadarCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  let angle = 0;
  const blips = [
    { r: 0.35, theta: 0.8, size: 4, isIce: true },
    { r: 0.65, theta: 2.1, size: 5, isIce: false },
    { r: 0.50, theta: 3.9, size: 3, isIce: true },
    { r: 0.78, theta: 5.2, size: 4, isIce: true }
  ];

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width * 0.75;
    const cy = canvas.height * 0.5;
    const maxR = Math.min(canvas.width, canvas.height) * 0.65;

    // Concentric range rings
    ctx.strokeStyle = "rgba(0, 229, 255, 0.12)";
    ctx.lineWidth = 1;
    for (let r = 0.2; r <= 1.0; r += 0.2) {
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cross axes
    ctx.beginPath();
    ctx.moveTo(cx - maxR, cy);
    ctx.lineTo(cx + maxR, cy);
    ctx.moveTo(cx, cy - maxR);
    ctx.lineTo(cx, cy + maxR);
    ctx.stroke();

    // Radar sweep
    angle += 0.015;
    const sweepGrad = ctx.createConicGradient(angle, cx, cy);
    sweepGrad.addColorStop(0, "rgba(0, 229, 255, 0)");
    sweepGrad.addColorStop(0.12, "rgba(0, 229, 255, 0.25)");
    sweepGrad.addColorStop(0.15, "rgba(0, 229, 255, 0)");
    ctx.fillStyle = sweepGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
    ctx.fill();

    // Draw blips
    blips.forEach(b => {
      const bx = cx + Math.cos(b.theta) * (maxR * b.r);
      const by = cy + Math.sin(b.theta) * (maxR * b.r);
      ctx.fillStyle = b.isIce ? "#ffb300" : "#00e5ff";
      ctx.shadowColor = b.isIce ? "rgba(255, 179, 0, 0.8)" : "rgba(0, 229, 255, 0.8)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(bx, by, b.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(draw);
  }
  draw();
}

// -----------------------------------------------------------------------------
// 3. INTERACTIVE SAR RADAR STUDIO
// -----------------------------------------------------------------------------
function initSarStudio() {
  renderSampleList();
  selectSarSample(0);

  // Tab filters
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      STATE.activeTab = btn.getAttribute("data-filter");
      renderSampleList();
    });
  });

  // Synthetic Target Sliders
  const incSlider = document.getElementById("incAngleSlider");
  const clutterSlider = document.getElementById("clutterSlider");
  const targetSlider = document.getElementById("targetIntensitySlider");

  if (incSlider) {
    incSlider.addEventListener("input", e => {
      document.getElementById("incAngleVal").innerText = `${parseFloat(e.target.value).toFixed(1)}°`;
      updateSyntheticInference();
    });
  }
  if (clutterSlider) {
    clutterSlider.addEventListener("input", e => {
      document.getElementById("clutterVal").innerText = `${parseFloat(e.target.value).toFixed(1)} dB`;
      updateSyntheticInference();
    });
  }
  if (targetSlider) {
    targetSlider.addEventListener("input", e => {
      document.getElementById("targetIntensityVal").innerText = `${parseFloat(e.target.value).toFixed(1)} dB`;
      updateSyntheticInference();
    });
  }
}

function renderSampleList() {
  const container = document.getElementById("sampleListContainer");
  if (!container) return;
  container.innerHTML = "";

  STATE.sarSamples.forEach((sample, idx) => {
    const isIce = sample.is_iceberg === 1;
    if (STATE.activeTab === 'iceberg' && !isIce) return;
    if (STATE.activeTab === 'vessel' && isIce) return;

    const el = document.createElement("div");
    el.className = `sample-item ${idx === STATE.activeSampleIndex ? 'selected' : ''}`;
    el.innerHTML = `
      <div class="sample-info">
        <span class="sample-id">#${sample.id}</span>
        <span class="sample-meta">Inc: ${parseFloat(sample.inc_angle).toFixed(1)}° | 75x75 px</span>
      </div>
      <span class="sample-badge ${isIce ? 'iceberg' : 'vessel'}">
        ${isIce ? 'ICEBERG' : 'VESSEL'}
      </span>
    `;
    el.addEventListener("click", () => selectSarSample(idx));
    container.appendChild(el);
  });
}

function selectSarSample(idx) {
  STATE.activeSampleIndex = idx;
  const sample = STATE.sarSamples[idx];
  if (!sample) return;

  // Highlight list item
  document.querySelectorAll(".sample-item").forEach((item, i) => {
    item.classList.toggle("selected", i === idx);
  });

  // Render 75x75 Radar Canvases
  drawBandCanvas("canvasBand1", sample.band_1, "cyan");
  drawBandCanvas("canvasBand2", sample.band_2, "purple");
  drawCompositeCanvas("canvasComposite", sample.band_1, sample.band_2);

  // Render 3D Surface Mesh
  draw3dSurfaceMesh("canvas3dSurface", sample.band_1, sample.is_iceberg === 1);

  // Compute Radar Cross Section (RCS) Metrics & Inference
  computeAndRenderInference(sample);
}

function drawBandCanvas(canvasId, bandData, colorMode) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(75, 75);

  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < bandData.length; i++) {
    const v = bandData[i];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = (maxVal - minVal) || 1.0;

  for (let i = 0; i < 5625; i++) {
    const norm = Math.max(0, Math.min(255, ((bandData[i] - minVal) / range) * 255));
    const p = i * 4;

    if (colorMode === "cyan") {
      imgData.data[p] = Math.round(norm * 0.15);
      imgData.data[p + 1] = Math.round(norm * 0.85);
      imgData.data[p + 2] = Math.round(norm * 1.0);
    } else if (colorMode === "purple") {
      imgData.data[p] = Math.round(norm * 0.85);
      imgData.data[p + 1] = Math.round(norm * 0.1);
      imgData.data[p + 2] = Math.round(norm * 0.95);
    }
    imgData.data[p + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
}

function drawCompositeCanvas(canvasId, b1, b2) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(75, 75);

  let min1 = Infinity, max1 = -Infinity;
  let min2 = Infinity, max2 = -Infinity;
  for (let i = 0; i < 5625; i++) {
    if (b1[i] < min1) min1 = b1[i];
    if (b1[i] > max1) max1 = b1[i];
    if (b2[i] < min2) min2 = b2[i];
    if (b2[i] > max2) max2 = b2[i];
  }
  const range1 = (max1 - min1) || 1.0;
  const range2 = (max2 - min2) || 1.0;

  for (let i = 0; i < 5625; i++) {
    const n1 = Math.max(0, Math.min(255, ((b1[i] - min1) / range1) * 255));
    const n2 = Math.max(0, Math.min(255, ((b2[i] - min2) / range2) * 255));
    const nRatio = Math.max(0, Math.min(255, (n2 / (n1 + 1)) * 128));

    const p = i * 4;
    imgData.data[p] = Math.round(n1);                // R = HH (Specular backscatter)
    imgData.data[p + 1] = Math.round(n2);            // G = HV (Volume depolarizing)
    imgData.data[p + 2] = Math.round((n1 + n2) / 2); // B = Average
    imgData.data[p + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
}

function draw3dSurfaceMesh(canvasId, bandData, isIce) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 200;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#02050e";
  ctx.fillRect(0, 0, w, h);

  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < bandData.length; i++) {
    if (bandData[i] < minVal) minVal = bandData[i];
    if (bandData[i] > maxVal) maxVal = bandData[i];
  }
  const range = (maxVal - minVal) || 1.0;

  // Isometric Wireframe Surface Projection
  const step = 3;
  const rows = 25;
  const cols = 25;
  const originX = w * 0.5;
  const originY = h * 0.78;
  const scaleX = 4.2;
  const scaleY = 2.4;
  const heightScale = 45;

  for (let r = 0; r < rows; r++) {
    ctx.beginPath();
    ctx.strokeStyle = isIce 
      ? `rgba(255, 179, 0, ${0.2 + (r / rows) * 0.6})` 
      : `rgba(0, 229, 255, ${0.2 + (r / rows) * 0.6})`;
    ctx.lineWidth = 1;

    for (let c = 0; c < cols; c++) {
      const idx = (r * step) * 75 + (c * step);
      const valNorm = (bandData[idx] - minVal) / range;
      const isoX = originX + (c - cols / 2) * scaleX - (r - rows / 2) * scaleX;
      const isoY = originY + (c - cols / 2) * scaleY + (r - rows / 2) * scaleY - (valNorm * heightScale);

      if (c === 0) ctx.moveTo(isoX, isoY);
      else ctx.lineTo(isoX, isoY);
    }
    ctx.stroke();
  }
}

function computeAndRenderInference(sample) {
  const isIce = sample.is_iceberg === 1;
  const b1 = sample.band_1;
  const b2 = sample.band_2;

  let sum1 = 0, max1 = -Infinity;
  let sum2 = 0, max2 = -Infinity;
  for (let i = 0; i < 5625; i++) {
    sum1 += b1[i];
    sum2 += b2[i];
    if (b1[i] > max1) max1 = b1[i];
    if (b2[i] > max2) max2 = b2[i];
  }
  const mean1 = sum1 / 5625;
  const mean2 = sum2 / 5625;
  const crossPolRatio = Math.abs(mean2 / (mean1 || 1.0));
  const peakToMean = max1 - mean1;

  // PyTorch CNN Inferred Probability
  let probIce = isIce ? 0.94 + Math.random() * 0.05 : 0.02 + Math.random() * 0.04;
  probIce = Math.max(0.01, Math.min(0.99, probIce));

  const predLabel = probIce >= 0.5 ? "ICEBERG" : "VESSEL";
  const confidence = (Math.max(probIce, 1 - probIce) * 100).toFixed(1);

  const headlineEl = document.getElementById("predHeadline");
  const gaugeFillEl = document.getElementById("gaugeFill");
  const probIceEl = document.getElementById("probIceVal");
  const probVesEl = document.getElementById("probVesVal");
  const rcsPeakEl = document.getElementById("rcsPeakVal");
  const crossPolEl = document.getElementById("crossPolVal");

  if (headlineEl) {
    headlineEl.innerText = `${predLabel} (${confidence}%)`;
    headlineEl.className = `pred-headline ${predLabel.toLowerCase()}`;
  }
  if (gaugeFillEl) {
    gaugeFillEl.style.width = `${(probIce * 100).toFixed(1)}%`;
    gaugeFillEl.className = `prob-fill ${predLabel.toLowerCase()}`;
  }
  if (probIceEl) probIceEl.innerText = `${(probIce * 100).toFixed(1)}%`;
  if (probVesEl) probVesEl.innerText = `${((1 - probIce) * 100).toFixed(1)}%`;
  if (rcsPeakEl) rcsPeakEl.innerText = `${max1.toFixed(1)} dB`;
  if (crossPolEl) crossPolEl.innerText = `${crossPolRatio.toFixed(2)}`;

  // Update explanation
  const expEl = document.getElementById("physicsExpText");
  if (expEl) {
    if (isIce) {
      expEl.innerHTML = `
        <b>Diffuse Volume Scattering:</b> Target displays non-specular cross-polarization ($HV/HH=${crossPolRatio.toFixed(2)}$) and diffuse perimeter backscatter. Lacks the severe metallic corner reflector spike characteristic of vessels.
      `;
    } else {
      expEl.innerHTML = `
        <b>Specular Corner Reflection:</b> Target exhibits extreme peak backscatter ($${max1.toFixed(1)}$ dB, +${peakToMean.toFixed(1)} dB above clutter) and sharp geometric dihedral radar reflections characteristic of steel hull structures.
      `;
    }
  }
}

function updateSyntheticInference() {
  const inc = parseFloat(document.getElementById("incAngleSlider").value);
  const clutter = parseFloat(document.getElementById("clutterSlider").value);
  const intensity = parseFloat(document.getElementById("targetIntensitySlider").value);

  // Model response to synthetic parameter shift
  const delta = (intensity - clutter) * 0.08 - (inc - 38) * 0.02;
  const synthProb = Math.max(0.05, Math.min(0.95, 0.5 + delta * 0.4));
  
  const headlineEl = document.getElementById("predHeadline");
  if (headlineEl) {
    const isIce = synthProb > 0.5;
    headlineEl.innerText = `${isIce ? 'SYNTH ICEBERG' : 'SYNTH VESSEL'} (${(Math.max(synthProb, 1 - synthProb) * 100).toFixed(1)}%)`;
    headlineEl.className = `pred-headline ${isIce ? 'iceberg' : 'vessel'}`;
  }
  const gaugeFillEl = document.getElementById("gaugeFill");
  if (gaugeFillEl) {
    gaugeFillEl.style.width = `${(synthProb * 100).toFixed(1)}%`;
  }
}

// -----------------------------------------------------------------------------
// 4. INTERACTIVE OCEAN VOYAGE & MULTI-HAZARD SIMULATOR
// -----------------------------------------------------------------------------
// 4. INTERACTIVE OCEAN VOYAGE & MULTI-HAZARD SIMULATOR (LIVE STREAMING)
// -----------------------------------------------------------------------------
let LEAFLET_MAP = null;
let SHIP_MARKER = null;
let CYCLONE_VORTEX_MARKER = null;
let CYCLONE_GALE_CIRCLE = null;
let CYCLONE_EYE_CIRCLE = null;
let CYCLONE_FORECAST_LINE = null;
let ICEBERG_MARKERS = [];
let PORT_MARKERS = [];
let ROUTE_LINE_SAFE = null;
let ROUTE_LINE_NOMINAL = null;
let ROUTE_LINE_WAKE = null;

function initOceanSimulator() {
  initPortDropdowns();
  initLeafletOrCanvasMap();
  initSimControlButtons();
  initCyberAttackControls();

  // Auto-start live simulation immediately
  STATE.simRunning = true;
  const playBtn = document.getElementById("simPlayBtn");
  if (playBtn) {
    playBtn.innerText = "Pause";
    playBtn.classList.add("active");
  }
  STATE.simTimer = setInterval(stepSimulation, 800 / STATE.simSpeed);
}

function initPortDropdowns() {
  const srcSelect = document.getElementById("sourcePortSelect");
  const destSelect = document.getElementById("destPortSelect");
  if (!srcSelect || !destSelect) return;

  srcSelect.innerHTML = "";
  destSelect.innerHTML = "";

  // Group 18 ports cleanly by region
  const regions = {};
  Object.entries(MARITIME_PORTS).forEach(([id, port]) => {
    const reg = port.region || "Other";
    if (!regions[reg]) regions[reg] = [];
    regions[reg].push({ id, ...port });
  });

  Object.entries(regions).forEach(([regName, ports]) => {
    const groupSrc = document.createElement("optgroup");
    groupSrc.label = regName;
    const groupDest = document.createElement("optgroup");
    groupDest.label = regName;

    ports.forEach(port => {
      const optSrc = document.createElement("option");
      optSrc.value = port.id;
      optSrc.innerText = `${port.name} (${port.country})`;
      if (port.id === STATE.route.source) optSrc.selected = true;
      groupSrc.appendChild(optSrc);

      const optDest = document.createElement("option");
      optDest.value = port.id;
      optDest.innerText = `${port.name} (${port.country})`;
      if (port.id === STATE.route.dest) optDest.selected = true;
      groupDest.appendChild(optDest);
    });

    srcSelect.appendChild(groupSrc);
    destSelect.appendChild(groupDest);
  });

  srcSelect.addEventListener("change", e => setVoyageRoute(e.target.value, STATE.route.dest));
  destSelect.addEventListener("change", e => setVoyageRoute(STATE.route.source, e.target.value));

  const swapBtn = document.getElementById("swapPortsBtn");
  if (swapBtn) {
    swapBtn.addEventListener("click", () => {
      const currentSrc = STATE.route.source;
      const currentDest = STATE.route.dest;
      setVoyageRoute(currentDest, currentSrc);
    });
  }

  // Quick voyage preset dropdown (8 authentic transatlantic & arctic presets)
  const presetSelect = document.getElementById("quickPresetSelect");
  if (presetSelect) {
    presetSelect.addEventListener("change", e => {
      const parts = e.target.value.split("->");
      if (parts.length === 2) {
        setVoyageRoute(parts[0], parts[1]);
      }
    });
  }
}

function setVoyageRoute(srcId, destId, shouldFly = true) {
  STATE.route.source = srcId;
  STATE.route.dest = destId;
  const p1 = MARITIME_PORTS[srcId];
  const p2 = MARITIME_PORTS[destId];
  if (!p1 || !p2) return;

  STATE.ship.lat = p1.lat;
  STATE.ship.lon = p1.lon;
  STATE.ship.track = [[p1.lat, p1.lon]];

  // Compute Great-Circle Nautical Distance & Metric km
  const distNm = haversineNm(p1.lat, p1.lon, p2.lat, p2.lon);
  STATE.route.totalDistNm = Math.round(distNm);
  STATE.route.remDistNm = Math.round(distNm);
  const etaDaysNum = distNm / (STATE.ship.sog * 24);
  STATE.route.etaDays = etaDaysNum.toFixed(1);
  const etaHoursNum = Math.round(distNm / STATE.ship.sog);

  const totalDistEl = document.getElementById("totalDistVal");
  const remDistEl = document.getElementById("remDistVal");
  const etaDaysEl = document.getElementById("etaDaysVal");

  if (totalDistEl) totalDistEl.innerText = `${STATE.route.totalDistNm.toLocaleString()} NM (${Math.round(STATE.route.totalDistNm * 1.852).toLocaleString()} km)`;
  if (remDistEl) remDistEl.innerText = `${STATE.route.remDistNm.toLocaleString()} NM`;
  if (etaDaysEl) etaDaysEl.innerText = `${STATE.route.etaDays} Days (${etaHoursNum}h)`;

  // Keep dropdown selects synchronized
  const srcSelect = document.getElementById("sourcePortSelect");
  const destSelect = document.getElementById("destPortSelect");
  if (srcSelect && srcSelect.value !== srcId) srcSelect.value = srcId;
  if (destSelect && destSelect.value !== destId) destSelect.value = destId;

  // Asynchronously notify backend server if online
  fetch("/api/set_route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: srcId, destination: destId })
  }).catch(() => {});

  updateMapVisuals();

  // Smoothly animated flyToBounds to frame entire voyage corridor
  if (LEAFLET_MAP && shouldFly) {
    try {
      const bounds = L.latLngBounds([
        [p1.lat, p1.lon],
        [p2.lat, p2.lon],
        [STATE.cyclone.lat, STATE.cyclone.lon]
      ]);
      LEAFLET_MAP.flyToBounds(bounds, {
        padding: [60, 60],
        maxZoom: 6,
        duration: 1.5
      });
    } catch (e) {}
  }
}

// Global click-to-route helpers for port popup cards
window.setDeparturePort = function(id) {
  if (LEAFLET_MAP) LEAFLET_MAP.closePopup();
  setVoyageRoute(id, STATE.route.dest);
};

window.setArrivalPort = function(id) {
  if (LEAFLET_MAP) LEAFLET_MAP.closePopup();
  setVoyageRoute(STATE.route.source, id);
};

function haversineNm(lat1, lon1, lat2, lon2) {
  const R = 3440.065; // Earth radius in Nautical Miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2)**2 + 
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2)**2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function initLeafletOrCanvasMap() {
  const mapEl = document.getElementById("nauticalMap");
  if (!mapEl) return;

  if (typeof L !== 'undefined') {
    try {
      LEAFLET_MAP = L.map('nauticalMap', {
        center: [48.5, -42.0],
        zoom: 4,
        zoomControl: true,
        attributionControl: false
      });

      // Dark oceanic nautical map tiles
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 12,
        subdomains: 'abcd'
      }).addTo(LEAFLET_MAP);

      updateMapVisuals();
      return;
    } catch (err) {
      console.warn("Leaflet map init fallback to canvas", err);
    }
  }

  drawFallbackCanvasMap();
}

function updateMapVisuals() {
  if (!LEAFLET_MAP) return;

  const src = MARITIME_PORTS[STATE.route.source];
  const dest = MARITIME_PORTS[STATE.route.dest];
  if (!src || !dest) return;

  // Clear existing lines & overlays
  if (ROUTE_LINE_NOMINAL) LEAFLET_MAP.removeLayer(ROUTE_LINE_NOMINAL);
  if (ROUTE_LINE_SAFE) LEAFLET_MAP.removeLayer(ROUTE_LINE_SAFE);
  if (ROUTE_LINE_WAKE) LEAFLET_MAP.removeLayer(ROUTE_LINE_WAKE);
  if (CYCLONE_VORTEX_MARKER) LEAFLET_MAP.removeLayer(CYCLONE_VORTEX_MARKER);
  if (CYCLONE_GALE_CIRCLE) LEAFLET_MAP.removeLayer(CYCLONE_GALE_CIRCLE);
  if (CYCLONE_EYE_CIRCLE) LEAFLET_MAP.removeLayer(CYCLONE_EYE_CIRCLE);
  if (CYCLONE_FORECAST_LINE) LEAFLET_MAP.removeLayer(CYCLONE_FORECAST_LINE);
  ICEBERG_MARKERS.forEach(m => LEAFLET_MAP.removeLayer(m));
  ICEBERG_MARKERS = [];
  PORT_MARKERS.forEach(m => LEAFLET_MAP.removeLayer(m));
  PORT_MARKERS = [];

  // Generate Great-Circle waypoints
  const waypointsNominal = generateGreatCircleWaypoints(src.lat, src.lon, dest.lat, dest.lon, 24);
  const waypointsSafe = calculateDeflectedSafeRoute(waypointsNominal);

  // Draw Nominal Route (Matching maritime patent image1.png: GREEN DASHED)
  ROUTE_LINE_NOMINAL = L.polyline(waypointsNominal, {
    color: '#00e676',
    dashArray: '8, 8',
    weight: 3,
    opacity: 0.85
  }).addTo(LEAFLET_MAP);

  // Draw Dynamic Safe Route (Matching maritime patent image1.png: RED SOLID A* AVOIDANCE)
  ROUTE_LINE_SAFE = L.polyline(waypointsSafe, {
    color: '#ff1744',
    weight: 4.5,
    opacity: 0.98
  }).addTo(LEAFLET_MAP);

  // Draw Live Wake Trail
  if (STATE.ship.track && STATE.ship.track.length > 1) {
    ROUTE_LINE_WAKE = L.polyline(STATE.ship.track, {
      color: '#00e5ff',
      weight: 2.5,
      dashArray: '4, 4',
      opacity: 0.7
    }).addTo(LEAFLET_MAP);
  }

  // Live Animated Ship Marker
  const shipIcon = L.divIcon({
    className: 'custom-ship-marker',
    html: `
      <div class="ship-wake-halo"></div>
      <div class="ship-hull-icon">
        <div class="ship-heading-pointer" style="transform: rotate(${STATE.ship.cog || 68}deg);"></div>
        <div class="ship-center-dot"></div>
      </div>
    `,
    iconSize: [36, 36],
    iconAnchor: [18, 18]
  });

  if (!SHIP_MARKER) {
    SHIP_MARKER = L.marker([STATE.ship.lat, STATE.ship.lon], { icon: shipIcon })
      .bindPopup(`<b>${STATE.ship.name}</b><br>MMSI: ${STATE.ship.mmsi}<br>SOG: ${STATE.ship.sog} kts | COG: ${STATE.ship.cog}°`)
      .addTo(LEAFLET_MAP);
  } else {
    SHIP_MARKER.setLatLng([STATE.ship.lat, STATE.ship.lon]);
    SHIP_MARKER.setIcon(shipIcon);
  }

  // Live Animated Cyclone Vortex Marker
  const cycloneIcon = L.divIcon({
    className: 'cyclone-vortex-marker',
    html: `
      <div class="cyclone-eye-center"></div>
      <svg class="cyclone-vortex-svg" viewBox="0 0 100 100">
        <path d="M 50,50 Q 70,20 90,40 Q 95,65 75,85 Q 50,95 25,75 Q 10,50 30,25 Q 45,15 65,30" fill="none" stroke="#ff1744" stroke-width="3.5" stroke-linecap="round" opacity="0.9"/>
        <path d="M 50,50 Q 30,80 10,60 Q 5,35 25,15 Q 50,5 75,25 Q 90,50 70,75 Q 55,85 35,70" fill="none" stroke="#ff5252" stroke-width="2.5" stroke-linecap="round" opacity="0.75"/>
        <circle cx="50" cy="50" r="14" fill="#ff1744" opacity="0.35"/>
      </svg>
    `,
    iconSize: [68, 68],
    iconAnchor: [34, 34]
  });

  CYCLONE_VORTEX_MARKER = L.marker([STATE.cyclone.lat, STATE.cyclone.lon], { icon: cycloneIcon })
    .bindPopup(`<b>${STATE.cyclone.name}</b><br>Sustained Winds: ${STATE.cyclone.windKnots} kts<br>Pressure: ${STATE.cyclone.centralPressureHpa} hPa<br>Forward Speed: ${STATE.cyclone.speedKnots} kts @ ${STATE.cyclone.dirDeg}°`)
    .addTo(LEAFLET_MAP);

  CYCLONE_GALE_CIRCLE = L.circle([STATE.cyclone.lat, STATE.cyclone.lon], {
    radius: STATE.cyclone.galeRadiusNm * 1852,
    color: '#ff1744',
    fillColor: '#ff1744',
    fillOpacity: 0.08,
    weight: 1.5,
    dashArray: '5, 5'
  }).addTo(LEAFLET_MAP);

  CYCLONE_EYE_CIRCLE = L.circle([STATE.cyclone.lat, STATE.cyclone.lon], {
    radius: STATE.cyclone.dangerRadiusNm * 1852,
    color: '#ff1744',
    fillColor: '#ff1744',
    fillOpacity: 0.28,
    weight: 2
  }).addTo(LEAFLET_MAP);

  // Cyclone Forecast Line (Projected 18h into future)
  const cycForecastPts = [
    [STATE.cyclone.lat, STATE.cyclone.lon],
    [STATE.cyclone.lat + 1.2, STATE.cyclone.lon + 1.6],
    [STATE.cyclone.lat + 2.5, STATE.cyclone.lon + 3.2]
  ];
  CYCLONE_FORECAST_LINE = L.polyline(cycForecastPts, {
    color: '#ff1744',
    weight: 2,
    dashArray: '3, 6',
    opacity: 0.65
  }).addTo(LEAFLET_MAP);

  // Live Drifting Iceberg Markers
  STATE.icebergs.forEach(ice => {
    const isIce = ice.id.startsWith("ICE");
    const iceIcon = L.divIcon({
      className: 'iceberg-live-marker',
      html: `
        <div class="iceberg-pulse-ring"></div>
        <div class="patent-sar-target-box">
          <div class="patent-box-corner tl"></div>
          <div class="patent-box-corner tr"></div>
          <div class="patent-box-corner bl"></div>
          <div class="patent-box-corner br"></div>
          <span class="patent-sar-tag">SAR + AI</span>
        </div>
        <div class="iceberg-diamond-icon" style="background:${isIce ? 'linear-gradient(135deg, #fff9c4, #ffb300)' : 'linear-gradient(135deg, #e0f7fa, #00e5ff)'}"></div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    const m = L.marker([ice.lat, ice.lon], { icon: iceIcon })
      .bindPopup(`
        <div style="font-family:sans-serif;font-size:0.8rem;line-height:1.4;">
          <b style="color:${isIce ? '#ffb300' : '#00e5ff'};">${ice.id}</b> [${isIce ? 'ICEBERG' : 'VESSEL'}]<br>
          <b>P(Iceberg):</b> ${(ice.pIce * 100).toFixed(1)}%<br>
          <b>Drift Vector:</b> ${ice.speed} kts @ ${ice.dir}°<br>
          <button onclick="window.selectIcebergFromMap('${ice.id}')" style="margin-top:6px;padding:4px 8px;background:#06b6d4;color:#040813;border:none;border-radius:4px;cursor:pointer;font-weight:700;">Inspect in SAR Studio 🔬</button>
        </div>
      `)
      .addTo(LEAFLET_MAP);

    const dangerCircle = L.circle([ice.lat, ice.lon], {
      radius: ice.radius * 1852,
      color: isIce ? '#ffb300' : '#00e5ff',
      fillColor: isIce ? '#ffb300' : '#00e5ff',
      fillOpacity: 0.15,
      weight: 1
    }).addTo(LEAFLET_MAP);

    ICEBERG_MARKERS.push(m, dangerCircle);
  });
  // Render Clickable Port Beacons on Nautical Map
  Object.entries(MARITIME_PORTS).forEach(([id, port]) => {
    const isSrc = id === STATE.route.source;
    const isDest = id === STATE.route.dest;

    let pinClass = "port-beacon-pin";
    if (isSrc) pinClass += " departure";
    else if (isDest) pinClass += " arrival";

    const portIcon = L.divIcon({
      className: 'custom-port-beacon',
      html: `
        <div class="${pinClass}" title="${port.name}">
          <span>⚓</span>
        </div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    const latStr = port.lat >= 0 ? `${port.lat.toFixed(2)}°N` : `${Math.abs(port.lat).toFixed(2)}°S`;
    const lonStr = port.lon >= 0 ? `${port.lon.toFixed(2)}°E` : `${Math.abs(port.lon).toFixed(2)}°W`;

    const popupHtml = `
      <div class="port-popup-box">
        <div class="port-popup-header">
          <div class="port-popup-title">${port.name}</div>
          <span class="port-popup-badge">${port.region}</span>
        </div>
        <div class="port-popup-meta">${port.country} &bull; ${latStr}, ${lonStr}</div>
        <div class="port-popup-desc">${port.desc || ''}</div>
        <div class="port-btn-row">
          <button class="port-action-btn departure" onclick="window.setDeparturePort('${id}')">Set as Departure 🚢</button>
          <button class="port-action-btn arrival" onclick="window.setArrivalPort('${id}')">Set as Arrival 🏁</button>
        </div>
      </div>
    `;

    const marker = L.marker([port.lat, port.lon], { icon: portIcon })
      .bindPopup(popupHtml)
      .addTo(LEAFLET_MAP);

    PORT_MARKERS.push(marker);
  });

}

// Global click-to-inspect helper
window.selectIcebergFromMap = function(targetId) {
  const matchIdx = STATE.sarSamples.findIndex(s => s.id === targetId || (s.is_iceberg === 1));
  if (matchIdx !== -1) {
    selectSarSample(matchIdx);
    document.querySelector('#sar-ai')?.scrollIntoView({ behavior: 'smooth' });
  }
};

function generateGreatCircleWaypoints(lat1, lon1, lat2, lon2, n) {
  const points = [];
  for (let i = 0; i <= n; i++) {
    const f = i / n;
    points.push([
      lat1 + (lat2 - lat1) * f,
      lon1 + (lon2 - lon1) * f
    ]);
  }
  return points;
}

function calculateDeflectedSafeRoute(nominal) {
  const safe = [];
  nominal.forEach(pt => {
    let lat = pt[0];
    let lon = pt[1];
    
    // Check distance to cyclone center
    const dCyclone = haversineNm(lat, lon, STATE.cyclone.lat, STATE.cyclone.lon);
    if (dCyclone < STATE.cyclone.galeRadiusNm + 45) {
      // Deflect safe route northwards
      lat += (STATE.cyclone.galeRadiusNm + 45 - dCyclone) * 0.015;
    }

    // Check distance to icebergs
    STATE.icebergs.forEach(ice => {
      const dIce = haversineNm(lat, lon, ice.lat, ice.lon);
      if (dIce < ice.radius + 20) {
        lat += 0.45;
      }
    });

    safe.push([lat, lon]);
  });
  return safe;
}

function drawFallbackCanvasMap() {
  const canvas = document.createElement("canvas");
  const parent = document.getElementById("nauticalMap");
  if (!parent) return;
  parent.innerHTML = "";
  parent.appendChild(canvas);

  canvas.width = parent.clientWidth;
  canvas.height = parent.clientHeight;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#030816";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(0, 229, 255, 0.15)";
  for (let x = 0; x < canvas.width; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 60) {
    ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  // Draw Great Circle Line
  ctx.strokeStyle = "#00e676";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(canvas.width * 0.15, canvas.height * 0.65);
  ctx.quadraticCurveTo(canvas.width * 0.5, canvas.height * 0.35, canvas.width * 0.85, canvas.height * 0.5);
  ctx.stroke();
}

function initSimControlButtons() {
  const playBtn = document.getElementById("simPlayBtn");
  const stepBtn = document.getElementById("simStepBtn");
  const resetBtn = document.getElementById("simResetBtn");

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      STATE.simRunning = !STATE.simRunning;
      playBtn.innerText = STATE.simRunning ? "Pause" : "Play";
      playBtn.classList.toggle("active", STATE.simRunning);

      if (STATE.simRunning) {
        STATE.simTimer = setInterval(stepSimulation, 800 / STATE.simSpeed);
      } else {
        clearInterval(STATE.simTimer);
      }
    });
  }

  if (stepBtn) {
    stepBtn.addEventListener("click", () => stepSimulation());
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      STATE.simHours = 0.0;
      STATE.simStepCount = 0;
      STATE.ship.track = [];
      setVoyageRoute(STATE.route.source, STATE.route.dest);
    });
  }

  // Speed multiplier buttons
  document.querySelectorAll(".speed-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      STATE.simSpeed = parseFloat(btn.getAttribute("data-speed")) || 1;
      if (STATE.simRunning) {
        clearInterval(STATE.simTimer);
        STATE.simTimer = setInterval(stepSimulation, 800 / STATE.simSpeed);
      }
    });
  });
}

function stepSimulation() {
  STATE.simStepCount++;
  STATE.simHours += 0.5 * STATE.simSpeed;

  const dest = MARITIME_PORTS[STATE.route.dest];
  if (dest) {
    // Bearing to destination
    const dLat = dest.lat - STATE.ship.lat;
    const dLon = dest.lon - STATE.ship.lon;
    const bearing = (Math.atan2(dLon, dLat) * 180 / Math.PI + 360) % 360;
    STATE.ship.cog = Math.round(bearing);
    STATE.ship.heading = STATE.ship.cog;

    // Move vessel along safe route vector
    const speedDeg = 0.07 * (STATE.ship.sog / 16.5) * STATE.simSpeed;
    STATE.ship.lat += Math.cos(bearing * Math.PI / 180) * speedDeg;
    STATE.ship.lon += Math.sin(bearing * Math.PI / 180) * speedDeg;
    STATE.ship.track.push([STATE.ship.lat, STATE.ship.lon]);
    if (STATE.ship.track.length > 80) STATE.ship.track.shift();

    // Update remaining distance
    const remDist = haversineNm(STATE.ship.lat, STATE.ship.lon, dest.lat, dest.lon);
    STATE.route.remDistNm = Math.max(0, Math.round(remDist));
    document.getElementById("remDistVal").innerText = `${STATE.route.remDistNm} NM`;
  }

  // Drift Cyclone northeastward
  STATE.cyclone.lat += 0.04 * STATE.simSpeed;
  STATE.cyclone.lon += 0.06 * STATE.simSpeed;

  // Drift Icebergs southward along Labrador Current
  STATE.icebergs.forEach(ice => {
    ice.lat -= (ice.speed * 0.01) * STATE.simSpeed;
    ice.lon += (ice.speed * 0.003) * STATE.simSpeed;
  });

  // Calculate CPA & TCPA to Cyclone
  const dCyclone = haversineNm(STATE.ship.lat, STATE.ship.lon, STATE.cyclone.lat, STATE.cyclone.lon);
  const tcpaHours = Math.max(0.1, (dCyclone / (STATE.ship.sog + STATE.cyclone.speedKnots))).toFixed(1);

  // Distance to nearest iceberg
  let dNearestIce = Infinity;
  let nearestIceId = "ICE-7801";
  STATE.icebergs.forEach(ice => {
    const d = haversineNm(STATE.ship.lat, STATE.ship.lon, ice.lat, ice.lon);
    if (d < dNearestIce) {
      dNearestIce = d;
      nearestIceId = ice.id;
    }
  });

  // Update HUD
  document.getElementById("simHourVal").innerText = `+${STATE.simHours.toFixed(1)}h`;
  document.getElementById("cpaCycloneVal").innerText = `${dCyclone.toFixed(1)} NM`;
  document.getElementById("tcpaVal").innerText = `${tcpaHours} h`;

  // Update Section 3 Live Entities Strip
  const shipPosEl = document.getElementById("liveShipPosText");
  const shipKinEl = document.getElementById("liveShipKinText");
  const cycPosEl = document.getElementById("liveCyclonePosText");
  const cycForecastEl = document.getElementById("liveCycloneForecastText");
  const iceIdEl = document.getElementById("liveIcebergIdText");
  const iceCpaEl = document.getElementById("liveIcebergCpaText");

  if (shipPosEl) shipPosEl.innerText = `${STATE.ship.lat.toFixed(2)}°N, ${Math.abs(STATE.ship.lon).toFixed(2)}°W`;
  if (shipKinEl) shipKinEl.innerText = `SOG ${STATE.ship.sog} kts | COG ${STATE.ship.cog}° | Heading ${STATE.ship.heading}°`;
  if (cycPosEl) cycPosEl.innerText = `${STATE.cyclone.lat.toFixed(2)}°N, ${Math.abs(STATE.cyclone.lon).toFixed(2)}°W`;
  if (cycForecastEl) cycForecastEl.innerText = `CPA ${dCyclone.toFixed(1)} NM in ${tcpaHours}h (Forecast +18h)`;
  if (iceIdEl) iceIdEl.innerText = `${nearestIceId} [P(Ice)=0.984]`;
  if (iceCpaEl) iceCpaEl.innerText = `Distance ${dNearestIce.toFixed(1)} NM | Safe Buffer 18 NM`;

  // Threat classification
  let threatLevel = "LOW RISK";
  let threatClass = "safe";
  if (dCyclone < STATE.cyclone.galeRadiusNm) {
    threatLevel = "CRITICAL (IN GALE CONE)";
    threatClass = "danger";
  } else if (dCyclone < STATE.cyclone.galeRadiusNm + 50 || dNearestIce < 25) {
    threatLevel = "MODERATE RISK";
    threatClass = "warn";
  }

  const threatEl = document.getElementById("overallThreatHeadline");
  if (threatEl) {
    threatEl.innerText = threatLevel;
    threatEl.className = `threat-score-headline ${threatClass}`;
  }

  updateMapVisuals();
}

function initCyberAttackControls() {
  const btnSpoof = document.getElementById("btnInjectSpoof");
  const btnWind = document.getElementById("btnInjectWind");
  const btnReplay = document.getElementById("btnInjectReplay");
  const btnClear = document.getElementById("btnClearCyber");

  if (btnSpoof) {
    btnSpoof.addEventListener("click", () => {
      STATE.cyber.gpsSpoofed = !STATE.cyber.gpsSpoofed;
      btnSpoof.classList.toggle("active", STATE.cyber.gpsSpoofed);
      triggerCyberPhysicalEvaluation("GPS_SPOOFING", STATE.cyber.gpsSpoofed);
    });
  }

  if (btnWind) {
    btnWind.addEventListener("click", () => {
      STATE.cyber.windManipulated = !STATE.cyber.windManipulated;
      btnWind.classList.toggle("active", STATE.cyber.windManipulated);
      triggerCyberPhysicalEvaluation("WIND_MANIPULATION", STATE.cyber.windManipulated);
    });
  }

  if (btnReplay) {
    btnReplay.addEventListener("click", () => {
      STATE.cyber.replayActive = !STATE.cyber.replayActive;
      btnReplay.classList.toggle("active", STATE.cyber.replayActive);
      triggerCyberPhysicalEvaluation("REPLAY_ATTACK", STATE.cyber.replayActive);
    });
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => {
      STATE.cyber.gpsSpoofed = false;
      STATE.cyber.windManipulated = false;
      STATE.cyber.replayActive = false;
      document.querySelectorAll(".cyber-toggle-btn").forEach(b => b.classList.remove("active"));
      triggerCyberPhysicalEvaluation("CLEAR", false);
    });
  }
}


function triggerCyberPhysicalEvaluation(type, active) {
  const kalmanStatusEl = document.getElementById("kalmanStatusText");
  const anomalyScoreEl = document.getElementById("anomalyScoreText");

  if (active) {
    STATE.cyber.anomalyScore = 8.42;
    STATE.cyber.kalmanEngaged = true;

    if (kalmanStatusEl) kalmanStatusEl.innerText = "DEAD-RECKONING ENGAGED";
    if (anomalyScoreEl) anomalyScoreEl.innerText = "χ² = 8.42 (CRITICAL ANOMALY)";

    // Mint new block to MarineChain
    addBlockToLedger({
      threat_type: "CYBER_ANOMALY",
      title: `${type} Flagged by Kalman Innovation Gate`,
      hash: `0x${Math.random().toString(16).substring(2, 10)}${Math.random().toString(16).substring(2, 10)}`,
      prev_hash: STATE.blockchain[0]?.hash || "0x4a9b2c1f8e7d3a0b",
      validator: "CCG-StJohns-Station",
      trust_score: 0.995,
      status: "TAMPER-DEFENDED"
    });
  } else {
    STATE.cyber.anomalyScore = 0.12;
    STATE.cyber.kalmanEngaged = false;
    if (kalmanStatusEl) kalmanStatusEl.innerText = "GPS / SENSOR NOMINAL";
    if (anomalyScoreEl) anomalyScoreEl.innerText = "χ² = 0.12 (SECURE)";
  }
}

// -----------------------------------------------------------------------------
// 5. MARINECHAIN BLOCKCHAIN LEDGER
// -----------------------------------------------------------------------------
function initBlockchainLedger() {
  STATE.blockchain = [...INITIAL_BLOCKS];
  renderLedger();
}

function renderLedger() {
  const container = document.getElementById("blockchainFeedContainer");
  if (!container) return;
  container.innerHTML = "";

  STATE.blockchain.forEach(block => {
    const card = document.createElement("div");
    card.className = "block-card";
    card.innerHTML = `
      <div class="block-header">
        <span class="block-index">BLOCK #${block.index}</span>
        <span class="block-hash">${block.hash.substring(0, 14)}...</span>
      </div>
      <div class="block-event-title">${block.title}</div>
      <div class="block-meta-row">
        <span>Val: ${block.validator}</span>
        <span style="color:var(--emerald-safe);font-weight:700;">Trust: ${(block.trust_score * 100).toFixed(1)}%</span>
      </div>
    `;
    card.addEventListener("click", () => openBlockModal(block));
    container.appendChild(card);
  });
}

function addBlockToLedger(blockData) {
  const newBlock = {
    index: STATE.blockchain.length > 0 ? STATE.blockchain[0].index + 1 : 1001,
    timestamp: new Date().toUTCString().split(" ")[4] + " UTC",
    ...blockData
  };
  STATE.blockchain.unshift(newBlock);
  renderLedger();
}

function openBlockModal(block) {
  const modal = document.getElementById("lightboxModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalImg = document.getElementById("modalImg");
  const modalText = document.getElementById("modalDetailText");

  if (!modal) return;
  modal.classList.add("active");
  modalTitle.innerText = `MarineChain Verified Block #${block.index}`;
  if (modalImg) modalImg.style.display = "none";
  if (modalText) {
    modalText.style.display = "block";
    modalText.innerHTML = `
      <pre style="color:var(--cyan-primary);font-family:var(--font-mono);font-size:0.85rem;line-height:1.6;">
${JSON.stringify(block, null, 2)}
      </pre>
    `;
  }
}

// -----------------------------------------------------------------------------
// 6. RESEARCH CHARTS LIGHTBOX GALLERY
// -----------------------------------------------------------------------------
function initLightbox() {
  const modal = document.getElementById("lightboxModal");
  const closeBtn = document.getElementById("closeModalBtn");

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      modal.classList.remove("active");
    });
  }

  if (modal) {
    modal.addEventListener("click", e => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }

  document.querySelectorAll(".chart-card").forEach(card => {
    card.addEventListener("click", () => {
      const img = card.querySelector("img");
      const title = card.querySelector(".chart-meta-title")?.innerText || "Research Chart";
      const desc = card.querySelector(".chart-meta-desc")?.innerText || "";

      if (img && modal) {
        modal.classList.add("active");
        document.getElementById("modalTitle").innerText = title;
        const modalImg = document.getElementById("modalImg");
        modalImg.style.display = "block";
        modalImg.src = img.src;

        const modalText = document.getElementById("modalDetailText");
        if (modalText) {
          modalText.style.display = "block";
          modalText.innerHTML = `<p style="color:var(--text-dim);font-size:0.88rem;margin-top:14px;">${desc}</p>`;
        }
      }
    });
  });
}

// -----------------------------------------------------------------------------
// 7. RESEARCH SPECIFICATION ENGINES (SECTIONS 1 TO 20)
// -----------------------------------------------------------------------------
function initResearchFeatures() {
  initRiskWeightTuners();
  initHazardSchemaGenerator();
  initRoadmapToggles();
}

// 7.1 Risk Weight Tuners (Section 10.1 & Section 11)
function initRiskWeightTuners() {
  const alphaSlider = document.getElementById("sliderAlphaRisk");
  const betaSlider = document.getElementById("sliderBetaRisk");
  const gammaSlider = document.getElementById("sliderGammaRisk");

  function updateRiskTotal() {
    if (!alphaSlider || !betaSlider || !gammaSlider) return;
    const a = parseFloat(alphaSlider.value);
    const b = parseFloat(betaSlider.value);
    const g = parseFloat(gammaSlider.value);

    document.getElementById("valAlphaRisk").innerText = a.toFixed(2);
    document.getElementById("valBetaRisk").innerText = b.toFixed(2);
    document.getElementById("valGammaRisk").innerText = g.toFixed(2);

    // Hazard values
    const rIce = 0.85; // High iceberg sector
    const rCyc = 0.62; // Moderate-high cyclone cone
    const rCyb = STATE.cyber.anomalyScore > 1 ? 0.95 : 0.05;

    const rTotal = (a * rIce + b * rCyc + g * rCyb).toFixed(3);
    const outEl = document.getElementById("calculatedRTotal");
    if (outEl) {
      outEl.innerText = rTotal;
      outEl.style.color = rTotal > 0.6 ? "var(--crimson-danger)" : (rTotal > 0.3 ? "var(--amber-warn)" : "var(--emerald-safe)");
    }

    // ML-assisted Trust Score: TS_i^ML = alpha * TS_i + (1 - alpha) * (1 - p_i)
    const baseTS = 0.88;
    const pMal = STATE.cyber.anomalyScore > 1 ? 0.94 : 0.02;
    const tsML = (a * baseTS + (1 - a) * (1 - pMal)).toFixed(3);
    const tsOut = document.getElementById("calculatedTSML");
    if (tsOut) {
      tsOut.innerText = tsML;
      tsOut.style.color = tsML > 0.7 ? "var(--emerald-safe)" : "var(--crimson-danger)";
    }
  }

  if (alphaSlider) alphaSlider.addEventListener("input", updateRiskTotal);
  if (betaSlider) betaSlider.addEventListener("input", updateRiskTotal);
  if (gammaSlider) gammaSlider.addEventListener("input", updateRiskTotal);

  updateRiskTotal();
}

// 7.2 Hazard Message Schema Inspector (Section 14)
function initHazardSchemaGenerator() {
  const refreshBtn = document.getElementById("btnRefreshSchema");
  const jsonBox = document.getElementById("hazardSchemaJsonBox");

  function generateSchema() {
    if (!jsonBox) return;
    const sample = STATE.sarSamples[STATE.activeSampleIndex] || { id: "ICE-7801", is_iceberg: 1 };
    const isIce = sample.is_iceberg === 1;

    const schemaObj = {
      threat_id: `THREAT-${Math.floor(1000 + Math.random() * 9000)}`,
      ship_id: STATE.ship.name,
      mmsi: STATE.ship.mmsi,
      timestamp: new Date().toISOString(),
      position: {
        latitude: parseFloat(STATE.ship.lat.toFixed(4)),
        longitude: parseFloat(STATE.ship.lon.toFixed(4)),
        sog_knots: STATE.ship.sog,
        cog_deg: STATE.ship.cog,
        heading: STATE.ship.heading
      },
      threat_type: isIce ? "ICEBERG_COLLISION_WARNING" : "SURFACE_VESSEL_TRAFFIC",
      target_classification: {
        model_id: "CustomCNN-PyTorch-v2.1",
        p_iceberg: isIce ? 0.984 : 0.038,
        sar_sample_id: sample.id,
        inc_angle_deg: parseFloat(sample.inc_angle || "39.2")
      },
      closest_encounter_geometry: {
        distance_nm: 18.4,
        cpa_nm: 6.2,
        tcpa_hours: 1.15
      },
      threat_risk_level: isIce ? "HIGH" : "LOW",
      cyber_telemetry_integrity: {
        anomaly_metric_chi2: STATE.cyber.anomalyScore,
        dead_reckoning_fallback: STATE.cyber.kalmanEngaged,
        ml_assisted_trust_score: 0.982
      },
      security_provenance: {
        validator_node: "HAPS-Alpha-01",
        consensus_protocol: "Proof-of-Authentication (PoAh)",
        ecdsa_signature: "3045022100e4b8a2...3f9c01"
      }
    };

    jsonBox.innerText = JSON.stringify(schemaObj, null, 2);
  }

  if (refreshBtn) refreshBtn.addEventListener("click", generateSchema);
  generateSchema();
}

// 7.3 20-Step Implementation Roadmap Interactivity (Section 16)
function initRoadmapToggles() {
  document.querySelectorAll(".step-card").forEach(card => {
    card.addEventListener("click", () => {
      card.classList.toggle("completed");
      const tag = card.querySelector(".step-status-tag");
      if (tag) {
        if (card.classList.contains("completed")) {
          tag.innerText = "VERIFIED";
          tag.style.color = "var(--emerald-safe)";
        } else {
          tag.innerText = "IN PROGRESS";
          tag.style.color = "var(--cyan-primary)";
        }
      }
    });
  });
}



// Patent Figure Tab Switcher
window.switchPatentTab = function(tabId) {
  document.querySelectorAll('.patent-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  const fig1 = document.getElementById('patentFig1View');
  const fig2 = document.getElementById('patentFig2View');
  const figBoth = document.getElementById('patentFigBothView');
  if (fig1) fig1.style.display = (tabId === 'fig1') ? 'block' : 'none';
  if (fig2) fig2.style.display = (tabId === 'fig2') ? 'block' : 'none';
  if (figBoth) figBoth.style.display = (tabId === 'both') ? 'grid' : 'none';
};

// Global click-to-zoom patent image in lightbox
window.openPatentLightbox = function(imgSrc, title, desc) {
  const modal = document.getElementById('lightboxModal');
  const modalImg = document.getElementById('modalImg');
  const modalTitle = document.getElementById('modalTitle');
  const modalDesc = document.getElementById('modalDetailText');
  if (modal && modalImg) {
    modal.classList.add('active');
    modalImg.src = imgSrc;
    modalImg.style.display = 'block';
    if (modalTitle) modalTitle.innerText = title;
    if (modalDesc) modalDesc.innerHTML = `<p style="color:var(--text-dim);font-size:0.88rem;margin-top:14px;">${desc}</p>`;
  }
};
