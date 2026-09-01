const methodLabels = {
  oe: "Over Exposure",
  ue: "Under Exposure",
  "hdr-transformer": "HDR-Transformer",
  safnet: "SAFNet",
  sctnet: "SCTNet",
  afunet: "AFUNet",
  defusion: "DeFusion",
  hsdsmef: "HSDSMEF",
  meflut: "MEFLUT",
  retinexmef: "RetinexMEF",
  ultrafusion: "UltraFusion",
  ours: "SwiftFusion (Ours)",
};

const standardMethods = [
  "oe",
  "ue",
  "hdr-transformer",
  "safnet",
  "sctnet",
  "defusion",
  "hsdsmef",
  "meflut",
  "ultrafusion",
  "ours",
];

const ufPlusMethods = [
  "oe",
  "ue",
  "hdr-transformer",
  "safnet",
  "afunet",
  "hsdsmef",
  "meflut",
  "retinexmef",
  "ultrafusion",
  "ours",
];

const compactFusionMethods = [
  "oe",
  "ue",
  "hsdsmef",
  "retinexmef",
  "ultrafusion",
  "ours",
];

const indoorVideoMethods = [
  "oe",
  "ue",
  "hdr-transformer",
  "hsdsmef",
  "ultrafusion",
  "ours",
];

const windowVideoMethods = [
  "oe",
  "ue",
  "safnet",
  "hsdsmef",
  "ultrafusion",
  "ours",
];

const scenes = [
  {
    id: "realhdrv-outdoor",
    title: "Outdoor Motion",
    description: "Fast subject motion under a large exposure difference.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 78.444, y: 24.049, w: 19.142, h: 28.397 }, methods: standardMethods }],
  },
  {
    id: "video-sparkler",
    title: "Sparkler at Night",
    description: "Fast local motion and an extremely saturated light source in a low-light scene.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 20.714, y: 57.25, w: 22.891, h: 36.788 }, methods: compactFusionMethods }],
  },
  {
    id: "ufplus-seaside",
    title: "Seaside Dancer",
    description: "Large body motion against a bright sky and water surface.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 42.446, y: 55.224, w: 9.636, h: 13.348 }, methods: ufPlusMethods }],
  },
  {
    id: "video-sunset",
    title: "Sunset Highlight",
    description: "Extreme highlight recovery around the setting sun while preserving fine structures.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 32.613, y: 6.698, w: 19.717, h: 32.196 }, methods: compactFusionMethods }],
  },
  {
    id: "realhdrv-indoor",
    title: "Indoor Motion",
    description: "Strong backlighting and motion around the face and hands.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 33.596, y: 17.076, w: 19.143, h: 29.398 }, methods: standardMethods }],
  },
  {
    id: "video-indoor",
    title: "Indoor Backlight",
    description: "Large hand motion across a severely over-exposed display in an indoor scene.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 3.426, y: 31.016, w: 19.717, h: 32.196 }, methods: indoorVideoMethods }],
  },
  {
    id: "ufplus-street",
    title: "Street Motion",
    description: "Overlapping pedestrians in a narrow high-contrast street.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 43.1, y: 48.704, w: 11.19, h: 16.028 }, methods: ufPlusMethods }],
  },
  {
    id: "video-window",
    title: "Window Motion",
    description: "Foreground motion across a bright window with large spatial exposure differences.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 49.726, y: 54.51, w: 16.786, h: 27.198 }, methods: windowVideoMethods }],
  },
  {
    id: "uf-night",
    title: "Night Signage",
    description: "4K reconstruction of saturated highlights, fine text, and moving foliage.",
    regions: [
      { label: "Region 1", color: "#e6c900", roi: { x: 69.076, y: 35.876, w: 8.47, h: 13.589 }, methods: standardMethods },
      { label: "Region 2", color: "#ef2b2d", roi: { x: 55.354, y: 54.773, w: 12.601, h: 19.77 }, methods: standardMethods },
    ],
  },
  {
    id: "uf-stairs",
    title: "Outdoor Stairs",
    description: "4K reconstruction with foreground motion and fine foliage.",
    regions: [
      { label: "Region 1", color: "#e6c900", roi: { x: 55.611, y: 33.319, w: 11.429, h: 19.141 }, methods: standardMethods },
      { label: "Region 2", color: "#ef2b2d", roi: { x: 37.892, y: 63.216, w: 12.614, h: 19.723 }, methods: standardMethods },
    ],
  },
  {
    id: "ufplus-trees",
    title: "Backlit Trees",
    description: "4K detail recovery around severely saturated foliage.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 50.559, y: 41.18, w: 9.759, h: 15.608 }, methods: standardMethods }],
  },
  {
    id: "ufplus-family",
    title: "Seaside Family",
    description: "Large foreground motion under severe backlighting.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 51.363, y: 59.079, w: 13.509, h: 21.82 }, methods: standardMethods }],
  },
];

const fullViewLabels = {
  under: "Under Exposure",
  over: "Over Exposure",
  ours: "SwiftFusion (Ours)",
};

const sceneGallery = document.getElementById("scene-gallery");
const lightbox = document.getElementById("result-lightbox");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxBackdrop = lightbox.querySelector(".lightbox-backdrop");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxDescription = document.getElementById("lightbox-description");
const fullImage = document.getElementById("full-result-image");
const fullSwitches = document.getElementById("full-switches");
const regionBox = document.getElementById("region-box");
const regionTabs = document.getElementById("region-tabs");
const patchBase = document.getElementById("patch-base");
const patchOurs = document.getElementById("patch-ours");
const patchMethodLabel = document.getElementById("patch-method-label");
const patchSlider = document.getElementById("patch-slider");
const patchDivider = document.getElementById("patch-divider");
const patchMethods = document.getElementById("patch-methods");

let activeScene = null;
let activeRegionIndex = 0;
let activeFullView = "ours";
let activeMethod = "ultrafusion";
let lastTrigger = null;
let galleryScene = scenes[0];
let galleryView = "ours";
let gallerySequenceIndex = 1;
let galleryTimer = null;
let galleryRestartTimer = null;

const gallerySequence = ["under", "ours", "over", "ours"];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function fullAsset(scene, view) {
  return `static/images/results/${scene.id}/${view}.webp`;
}

function patchAsset(scene, regionIndex, method) {
  return `static/images/results/${scene.id}/region-${regionIndex + 1}-${method}.webp`;
}

function setGalleryView(view) {
  galleryView = view;
  sceneGallery.querySelectorAll(".exposure-layer").forEach((image) => {
    image.classList.toggle("active", image.dataset.view === view);
  });
  sceneGallery.querySelectorAll(".exposure-button").forEach((button) => {
    const isActive = button.dataset.view === view;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
  const label = sceneGallery.querySelector(".exposure-view-label");
  if (label) label.textContent = fullViewLabels[view];
}

function stopGalleryAuto() {
  window.clearInterval(galleryTimer);
  window.clearTimeout(galleryRestartTimer);
  galleryTimer = null;
}

function startGalleryAuto(delay = 0) {
  stopGalleryAuto();
  if (reduceMotion.matches || !lightbox.hidden) return;
  const begin = () => {
    galleryTimer = window.setInterval(() => {
      gallerySequenceIndex = (gallerySequenceIndex + 1) % gallerySequence.length;
      setGalleryView(gallerySequence[gallerySequenceIndex]);
    }, 1850);
  };
  if (delay) galleryRestartTimer = window.setTimeout(begin, delay);
  else begin();
}

function setGalleryScene(scene, trigger) {
  if (galleryScene === scene && trigger?.classList.contains("active")) return;
  galleryScene = scene;
  const stage = sceneGallery.querySelector(".exposure-stage-button");
  stage.setAttribute("aria-label", `Open ${scene.title} comparison`);
  stage.dataset.scene = scene.id;
  sceneGallery.querySelectorAll(".exposure-layer").forEach((image) => {
    image.src = fullAsset(scene, image.dataset.view);
  });
  sceneGallery.querySelectorAll(".scene-thumbnail").forEach((button) => {
    const isActive = button.dataset.scene === scene.id;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-current", isActive ? "true" : "false");
  });
  gallerySequenceIndex = 1;
  setGalleryView("ours");
  startGalleryAuto(650);
}

function renderGallery() {
  sceneGallery.innerHTML = `
    <div class="exposure-stage">
      <button class="exposure-stage-button" type="button" aria-label="Open ${galleryScene.title} comparison">
        <span class="exposure-image-stack">
          <img class="exposure-layer" data-view="under" src="${fullAsset(galleryScene, "under")}" alt="" />
          <img class="exposure-layer active" data-view="ours" src="${fullAsset(galleryScene, "ours")}" alt="" />
          <img class="exposure-layer" data-view="over" src="${fullAsset(galleryScene, "over")}" alt="" />
          <span class="exposure-scan" aria-hidden="true"></span>
        </span>
        <span class="exposure-view-label">${fullViewLabels.ours}</span>
        <span class="stage-open" aria-hidden="true">
          <span>Inspect details</span><strong>↗</strong>
        </span>
      </button>
      <div class="exposure-controls" aria-label="Exposure preview">
        <button class="exposure-button" type="button" data-view="under" aria-pressed="false">UE</button>
        <button class="exposure-button active" type="button" data-view="ours" aria-pressed="true">SwiftFusion</button>
        <button class="exposure-button" type="button" data-view="over" aria-pressed="false">OE</button>
      </div>
    </div>
    <div class="scene-filmstrip" aria-label="Result scenes"></div>
  `;

  const stage = sceneGallery.querySelector(".exposure-stage-button");
  stage.addEventListener("click", () => openScene(galleryScene, stage));
  stage.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") return;
    stopGalleryAuto();
    const bounds = stage.getBoundingClientRect();
    const position = (event.clientX - bounds.left) / bounds.width;
    const view = position < 0.3 ? "under" : position > 0.7 ? "over" : "ours";
    if (view !== galleryView) setGalleryView(view);
  });
  stage.addEventListener("pointerleave", () => startGalleryAuto(700));

  sceneGallery.querySelectorAll(".exposure-button").forEach((button) => {
    button.addEventListener("click", () => {
      stopGalleryAuto();
      const view = button.dataset.view;
      gallerySequenceIndex = gallerySequence.indexOf(view);
      setGalleryView(view);
      startGalleryAuto(4200);
    });
  });

  const filmstrip = sceneGallery.querySelector(".scene-filmstrip");
  const fragment = document.createDocumentFragment();
  scenes.forEach((scene, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `scene-thumbnail${index === 0 ? " active" : ""}`;
    button.dataset.scene = scene.id;
    button.setAttribute("aria-current", index === 0 ? "true" : "false");
    button.setAttribute("aria-label", `Open ${scene.title} result`);
    button.innerHTML = `
      <span class="thumbnail-triptych" aria-hidden="true">
        <img class="thumbnail-layer thumbnail-under" src="${fullAsset(scene, "under")}" alt="" loading="lazy" />
        <img class="thumbnail-layer thumbnail-ours" src="${fullAsset(scene, "ours")}" alt="" loading="lazy" />
        <img class="thumbnail-layer thumbnail-over" src="${fullAsset(scene, "over")}" alt="" loading="lazy" />
      </span>
      <span class="thumbnail-open-cue" aria-hidden="true">View details ↗</span>
    `;
    button.addEventListener("pointerenter", () => setGalleryScene(scene, button));
    button.addEventListener("focus", () => setGalleryScene(scene, button));
    button.addEventListener("click", () => openScene(scene, button));
    fragment.appendChild(button);
  });
  filmstrip.appendChild(fragment);
  startGalleryAuto(900);
}

function setRegionBox() {
  const region = activeScene.regions[activeRegionIndex];
  const roi = region.roi;
  regionBox.style.left = `${roi.x}%`;
  regionBox.style.top = `${roi.y}%`;
  regionBox.style.width = `${roi.w}%`;
  regionBox.style.height = `${roi.h}%`;
  regionBox.style.borderColor = region.color;
  regionBox.hidden = activeFullView !== "ours";
}

function selectFullView(view) {
  activeFullView = view;
  fullImage.src = fullAsset(activeScene, view);
  fullImage.alt = `${activeScene.title} · ${fullViewLabels[view]}`;
  fullSwitches.querySelectorAll(".view-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  setRegionBox();
}

function renderFullSwitches() {
  fullSwitches.innerHTML = "";
  ["under", "over", "ours"].forEach((view) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "view-button";
    button.dataset.view = view;
    button.textContent = fullViewLabels[view];
    button.addEventListener("click", () => selectFullView(view));
    fullSwitches.appendChild(button);
  });
  selectFullView("ours");
}

function updatePatchSlider() {
  const value = Number(patchSlider.value);
  patchOurs.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  patchDivider.style.left = `${value}%`;
}

function selectPatchMethod(method) {
  activeMethod = method;
  const region = activeScene.regions[activeRegionIndex];
  const selectedLabel = methodLabels[method];
  patchBase.src = patchAsset(activeScene, activeRegionIndex, method);
  patchBase.alt = `${activeScene.title} · ${region.label} · ${selectedLabel}`;
  patchOurs.src = patchAsset(activeScene, activeRegionIndex, "ours");
  patchOurs.alt = `${activeScene.title} · ${region.label} · SwiftFusion (Ours)`;
  patchMethodLabel.textContent = selectedLabel;
  patchMethods.querySelectorAll(".patch-method-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.method === method);
  });
}

function renderPatchMethods() {
  patchMethods.innerHTML = "";
  const region = activeScene.regions[activeRegionIndex];
  region.methods.forEach((method) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "patch-method-button";
    button.dataset.method = method;
    button.title = methodLabels[method];
    button.innerHTML = `
      <img src="${patchAsset(activeScene, activeRegionIndex, method)}" alt="" loading="lazy" />
      <span>${methodLabels[method]}</span>
    `;
    button.addEventListener("click", () => selectPatchMethod(method));
    patchMethods.appendChild(button);
  });
  activeMethod = region.methods.includes("ultrafusion") ? "ultrafusion" : region.methods[0];
  selectPatchMethod(activeMethod);
}

function selectRegion(index) {
  activeRegionIndex = index;
  regionTabs.querySelectorAll(".region-button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.region) === index);
  });
  patchSlider.value = "50";
  updatePatchSlider();
  setRegionBox();
  renderPatchMethods();
}

function renderRegionTabs() {
  regionTabs.innerHTML = "";
  if (activeScene.regions.length < 2) {
    regionTabs.hidden = true;
    return;
  }
  regionTabs.hidden = false;
  activeScene.regions.forEach((region, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "region-button";
    button.dataset.region = String(index);
    button.textContent = region.label;
    button.addEventListener("click", () => selectRegion(index));
    regionTabs.appendChild(button);
  });
}

function openScene(scene, trigger) {
  stopGalleryAuto();
  activeScene = scene;
  activeRegionIndex = 0;
  lastTrigger = trigger;
  lightboxTitle.textContent = scene.title;
  lightboxDescription.textContent = scene.description;
  renderFullSwitches();
  renderRegionTabs();
  selectRegion(0);
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
  lightboxClose.focus();
}

function closeScene() {
  lightbox.hidden = true;
  document.body.style.overflow = "";
  if (lastTrigger) lastTrigger.focus();
  startGalleryAuto(900);
}

patchSlider.addEventListener("input", updatePatchSlider);
lightboxClose.addEventListener("click", closeScene);
lightboxBackdrop.addEventListener("click", closeScene);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) closeScene();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopGalleryAuto();
  else startGalleryAuto(700);
});

renderGallery();
