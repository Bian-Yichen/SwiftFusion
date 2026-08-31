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

const scenes = [
  {
    id: "realhdrv-outdoor",
    title: "Outdoor Motion",
    description: "Fast subject motion under a large exposure difference.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 78.444, y: 24.049, w: 19.142, h: 28.397 }, methods: standardMethods }],
  },
  {
    id: "realhdrv-indoor",
    title: "Indoor Motion",
    description: "Strong backlighting and motion around the face and hands.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 33.596, y: 17.076, w: 19.143, h: 29.398 }, methods: standardMethods }],
  },
  {
    id: "ufplus-seaside",
    title: "Seaside Dancer",
    description: "Large body motion against a bright sky and water surface.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 42.446, y: 55.224, w: 9.636, h: 13.348 }, methods: ufPlusMethods }],
  },
  {
    id: "ufplus-street",
    title: "Street Motion",
    description: "Overlapping pedestrians in a narrow high-contrast street.",
    regions: [{ label: "Detail", color: "#ef2b2d", roi: { x: 43.1, y: 48.704, w: 11.19, h: 16.028 }, methods: ufPlusMethods }],
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
const sliderMethodLabel = document.getElementById("slider-method-label");
const patchSlider = document.getElementById("patch-slider");
const patchDivider = document.getElementById("patch-divider");
const patchMethods = document.getElementById("patch-methods");

let activeScene = null;
let activeRegionIndex = 0;
let activeFullView = "ours";
let activeMethod = "ultrafusion";
let lastTrigger = null;

function fullAsset(scene, view) {
  return `static/images/results/${scene.id}/${view}.webp`;
}

function patchAsset(scene, regionIndex, method) {
  return `static/images/results/${scene.id}/region-${regionIndex + 1}-${method}.webp`;
}

function renderGallery() {
  const fragment = document.createDocumentFragment();
  scenes.forEach((scene) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-card";
    button.innerHTML = `
      <img src="${fullAsset(scene, "ours")}" alt="${scene.title}" loading="lazy" />
      <span class="scene-card-text">
        <span class="scene-card-title">${scene.title}</span>
        <span class="scene-card-action">View details →</span>
      </span>
    `;
    button.addEventListener("click", () => openScene(scene, button));
    fragment.appendChild(button);
  });
  sceneGallery.appendChild(fragment);
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
  sliderMethodLabel.textContent = selectedLabel;
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
}

patchSlider.addEventListener("input", updatePatchSlider);
lightboxClose.addEventListener("click", closeScene);
lightboxBackdrop.addEventListener("click", closeScene);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) closeScene();
});

renderGallery();
