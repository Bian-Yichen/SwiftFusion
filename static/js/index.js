const scenes = {
  dancer: {
    label: "Dancer",
    methods: [
      { key: "over", label: "Over Exposure", image: "static/images/dancer-over.webp" },
      { key: "under", label: "Under Exposure", image: "static/images/dancer-under.webp" },
      { key: "ours", label: "SwiftFusion (Ours)", image: "static/images/dancer-ours.webp" },
    ],
  },
  store: {
    label: "Indoor motion",
    methods: [
      { key: "over", label: "Over Exposure", image: "static/images/store-over.webp" },
      { key: "under", label: "Under Exposure", image: "static/images/store-under.webp" },
      { key: "ours", label: "SwiftFusion (Ours)", image: "static/images/store-ours.webp" },
    ],
  },
};

const sceneSelect = document.getElementById("scene-select");
const mainImage = document.getElementById("comparison-image");
const status = document.getElementById("comparison-status");
const methodButtons = document.getElementById("method-buttons");
const blinkButton = document.getElementById("blink-button");

let currentScene = "dancer";
let currentMethod = "ours";
let blinkTimer = null;
let blinkIndex = 0;

function stopBlink() {
  if (blinkTimer) {
    window.clearInterval(blinkTimer);
    blinkTimer = null;
  }
  blinkButton.classList.remove("active");
  blinkButton.textContent = "Blink Inputs";
}

function selectMethod(methodKey, stopAnimation = true) {
  if (stopAnimation) stopBlink();
  const scene = scenes[currentScene];
  const method = scene.methods.find((item) => item.key === methodKey);
  if (!method) return;

  currentMethod = methodKey;
  mainImage.src = method.image;
  mainImage.alt = `${scene.label} · ${method.label}`;
  status.textContent = `${scene.label} · ${method.label}`;

  document.querySelectorAll(".method-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.method === methodKey);
  });
}

function renderMethods() {
  methodButtons.innerHTML = "";
  const scene = scenes[currentScene];

  scene.methods.forEach((method, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "method-button";
    button.dataset.method = method.key;
    button.setAttribute("aria-label", `Show ${method.label}`);
    button.innerHTML = `
      <img src="${method.image}" alt="" />
      <span>${index + 1}. ${method.label}</span>
    `;
    button.addEventListener("click", () => selectMethod(method.key));
    methodButtons.appendChild(button);
  });

  selectMethod(currentMethod);
}

sceneSelect.addEventListener("change", (event) => {
  stopBlink();
  currentScene = event.target.value;
  currentMethod = "ours";
  renderMethods();
});

blinkButton.addEventListener("click", () => {
  if (blinkTimer) {
    stopBlink();
    selectMethod("ours", false);
    return;
  }

  blinkButton.classList.add("active");
  blinkButton.textContent = "Stop Blinking";
  blinkIndex = 0;
  selectMethod("over", false);
  blinkTimer = window.setInterval(() => {
    blinkIndex = (blinkIndex + 1) % 2;
    selectMethod(blinkIndex === 0 ? "over" : "under", false);
  }, 650);
});

document.addEventListener("keydown", (event) => {
  const index = Number(event.key) - 1;
  const methods = scenes[currentScene].methods;
  if (index >= 0 && index < methods.length) {
    selectMethod(methods[index].key);
  }
});

renderMethods();
