const state = {
  health: null,
  folders: [],
  folder: null,
  pollTimer: null,
  token: null,
  faces: [],
  faceIndex: 0,
  matches: [],
  threshold: 0.5,
  searching: false,
  searchQueued: false,
  searchError: "",
  searched: false,
  searchedFaces: 0,
  uploading: false,
  folderPosting: false,
  zipBusy: false,
  lightbox: -1,
};

let gridSig = "";
let chipSig = "";
let toastTimer = 0;

const $ = (id) => document.getElementById(id);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function percent(similarity) {
  if (!Number.isFinite(similarity)) return 0;
  return Math.round(clamp(similarity, 0, 1) * 100);
}

function driveFileUrl(fileId) {
  return "https://drive.google.com/file/d/" + encodeURIComponent(fileId) + "/view";
}

function driveFolderUrl(folderId) {
  return "https://drive.google.com/drive/folders/" + encodeURIComponent(folderId);
}

function previewUrl(fileId) {
  return "/api/preview/" + encodeURIComponent(fileId);
}

function detailMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || "Check that field.").join(" ");
  }
  return "";
}

async function readBody(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(detailMessage(data.detail) || "Something went wrong.");
  }
  return data;
}

async function getJSON(url) {
  return readBody(await fetch(url));
}

async function postJSON(url, payload) {
  return readBody(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }));
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4600);
}

function visibleMatches() {
  return state.matches.filter((match) => match.similarity >= state.threshold - 1e-6);
}

function safeBox(bbox) {
  if (!bbox) return null;
  const x = Number(bbox.x);
  const y = Number(bbox.y);
  const w = Number(bbox.w);
  const h = Number(bbox.h);
  if (![x, y, w, h].every(Number.isFinite)) return null;
  return { x, y, w, h };
}

function placeBox(el, bbox) {
  const box = safeBox(bbox);
  if (!box) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.style.left = (box.x * 100) + "%";
  el.style.top = (box.y * 100) + "%";
  el.style.width = (box.w * 100) + "%";
  el.style.height = (box.h * 100) + "%";
}

function sliderPercent() {
  const slider = $("threshold");
  const min = Number(slider.min);
  const max = Number(slider.max);
  const value = Number(slider.value);
  if (max <= min) return 0;
  return ((value - min) / (max - min)) * 100;
}

function updateSliderFill() {
  $("threshold").style.setProperty("--pct", sliderPercent() + "%");
  $("thresholdLabel").textContent = percent(state.threshold) + "%";
  const shown = percent(state.threshold);
  $("threshold").setAttribute("aria-valuenow", String(shown));
  $("threshold").setAttribute("aria-valuetext", shown + " percent");
}

function progressPct(folder) {
  if (!folder) return 0;
  if (folder.status === "discovering") return 8;
  const total = Number(folder.total_photos) || 0;
  if (!total) return folder.status === "completed" ? 100 : 0;
  const done = Math.min(total, Number(folder.done != null ? folder.done : folder.indexed_photos) || 0);
  return Math.round((done / total) * 100);
}

function statusText(folder) {
  if (!folder) return "";
  if (folder.status === "discovering") return "Looking through the folder…";
  if (folder.status === "indexing") {
    const total = Number(folder.total_photos) || 0;
    const done = Number(folder.done != null ? folder.done : folder.indexed_photos) || 0;
    const name = folder.current_file ? " · " + folder.current_file : "";
    return "Reading photos · " + done.toLocaleString() + " of " + total.toLocaleString() + name;
  }
  if (folder.status === "error") return folder.error || folder.message || "Indexing failed.";
  const photos = Number(folder.indexed_photos) || 0;
  const faces = Number(folder.face_count) || 0;
  return photos.toLocaleString() + " photos · " + faces.toLocaleString() + " faces";
}

function folderButtonLabel() {
  const typed = $("folderUrl").value.trim();
  const current = state.folder;
  const same = current && typed && (typed === current.url || typed.includes(current.id));
  if (same && (current.status === "indexing" || current.status === "discovering")) return "Indexing…";
  if (same && (current.status === "completed" || current.status === "error")) return "Check for new photos";
  return "Index folder";
}

function welcomeCopy() {
  if (!state.folder && state.token) return "Face ready. Paste a Drive folder to search it.";
  if (!state.folder) return "Paste a public Google Drive folder, add one photo of a person, and download every match.";
  if (state.folder.status === "error") return state.folder.error || "That folder couldn't be indexed.";
  if (state.folder.status === "discovering" || state.folder.status === "indexing") {
    return "Indexing is running. Add a photo of the person whenever you're ready.";
  }
  return "Folder ready. Add a clear photo of the person you want to find.";
}

function emptyCopy() {
  const level = percent(state.threshold);
  if (state.searchError) return state.searchError;
  const faces = state.searchedFaces || 0;
  const running = state.folder && (state.folder.status === "indexing" || state.folder.status === "discovering");
  if (faces === 0 && running) return "No faces indexed yet. Matches will show up as photos are read.";
  if (faces === 0) return "No faces were found in this folder.";
  if (state.matches.length) return "No photos above " + level + "%. Lower the match level to see more.";
  return "No photos above " + level + "%. Try another photo of the same person, or lower the match level.";
}

function selectedCrop() {
  const face = state.faces[state.faceIndex];
  return face ? face.crop_url : "";
}

function renderPanel() {
  const folder = state.folder;
  const label = folderButtonLabel();
  $("folderBtn").textContent = label;
  $("folderBtn").disabled = state.folderPosting || label === "Indexing…";
  $("folderName").hidden = !folder;
  $("folderName").textContent = folder ? (folder.name || "Photo folder") : "";
  $("folderOpen").hidden = !folder;
  if (folder) $("folderOpen").href = driveFolderUrl(folder.id);
  const detail = $("folderDetail");
  detail.textContent = statusText(folder);
  detail.classList.toggle("is-error", Boolean(folder && folder.status === "error"));
  const running = folder && (folder.status === "indexing" || folder.status === "discovering");
  const bar = $("folderBar");
  bar.hidden = !running;
  if (running) {
    const pct = progressPct(folder);
    bar.setAttribute("aria-valuenow", String(pct));
    bar.firstElementChild.style.width = pct + "%";
  }
  const note = folder && folder.status === "completed" && folder.message;
  $("folderNote").hidden = !note;
  $("folderNote").textContent = note || "";
  $("removeFolder").hidden = !folder;
  renderChips();
  renderDrop();
  renderFaces();
  const slider = $("threshold");
  if (document.activeElement !== slider) slider.value = String(state.threshold);
  updateSliderFill();
}

function renderChips() {
  const wrap = $("folderList");
  const sig = state.folders.map((folder) => folder.id + ":" + folder.name).join("|")
    + ">" + (state.folder ? state.folder.id : "");
  if (!state.folders.length) {
    wrap.hidden = true;
    wrap.replaceChildren();
    chipSig = sig;
    return;
  }
  wrap.hidden = false;
  if (sig === chipSig) return;
  chipSig = sig;
  wrap.replaceChildren();
  state.folders.forEach((folder) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip" + (state.folder && state.folder.id === folder.id ? " is-on" : "");
    button.textContent = folder.name || "Photo folder";
    button.addEventListener("click", () => selectFolder(folder.id));
    wrap.appendChild(button);
  });
}

function renderDrop() {
  $("drop").classList.toggle("is-busy", state.uploading);
  if (state.uploading) $("dropText").textContent = "Reading faces…";
  else if (state.token) $("dropText").textContent = "Drop another photo to replace this person";
  else $("dropText").textContent = "Drop a clear photo of one person";
}

function renderFaces() {
  const row = $("faceRow");
  const label = $("facePick");
  if (!state.faces.length) {
    row.hidden = true;
    label.hidden = true;
    row.replaceChildren();
    return;
  }
  label.hidden = false;
  label.textContent = state.faces.length > 1 ? "Which person?" : "Face found";
  row.hidden = false;
  row.replaceChildren();
  state.faces.forEach((face, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "face" + (index === state.faceIndex ? " is-on" : "");
    button.setAttribute("aria-pressed", index === state.faceIndex ? "true" : "false");
    button.setAttribute("aria-label", "Use face " + (index + 1));
    const img = document.createElement("img");
    img.src = face.crop_url;
    img.alt = "";
    button.appendChild(img);
    button.addEventListener("click", () => {
      if (state.faceIndex === index) return;
      state.faceIndex = index;
      maybeSearch();
    });
    row.appendChild(button);
  });
}

function renderResults() {
  const hasQuery = Boolean(state.token && state.folder);
  $("welcome").hidden = hasQuery;
  if (!hasQuery) {
    $("welcomeText").textContent = welcomeCopy();
    $("toolbar").hidden = true;
    $("grid").hidden = true;
    $("empty").hidden = true;
    gridSig = "";
    return;
  }
  const visible = visibleMatches();
  $("toolbar").hidden = false;
  updateToolbar(visible);
  const showGrid = state.searched && visible.length > 0;
  $("grid").hidden = !showGrid;
  if (!showGrid) {
    $("empty").hidden = false;
    $("empty").textContent = state.searched || state.searchError
      ? emptyCopy()
      : "Looking for this person…";
    $("grid").replaceChildren();
    gridSig = "";
    return;
  }
  $("empty").hidden = true;
  const sig = visible.map((match) => match.file_id + "@" + Number(match.similarity).toFixed(4)).join("|");
  if (sig === gridSig) return;
  gridSig = sig;
  buildGrid(visible);
}

function updateToolbar(visible) {
  const count = visible.length;
  const noun = count === 1 ? "photo" : "photos";
  $("resultCount").textContent = count ? count.toLocaleString() + " " + noun : "No photos yet";
  const hint = $("resultHint");
  if (state.searching) hint.textContent = "Updating…";
  else if (state.folder && (state.folder.status === "indexing" || state.folder.status === "discovering")) {
    hint.textContent = "Still indexing. Results update on their own.";
  } else hint.textContent = "Above " + percent(state.threshold) + "% match";
  const crop = selectedCrop();
  const mini = $("refMini");
  if (crop) mini.src = crop;
  mini.alt = "Selected reference face";
  const button = $("downloadBtn");
  button.disabled = state.zipBusy || count === 0;
  button.textContent = state.zipBusy
    ? "Preparing zip…"
    : (count ? "Download " + count.toLocaleString() + " " + noun : "Download photos");
}

function buildGrid(visible) {
  const grid = $("grid");
  grid.replaceChildren();
  visible.forEach((match, index) => {
    const card = document.createElement("article");
    card.className = "card";
    if (index < 12) card.style.animationDelay = (index * 30) + "ms";
    const shot = document.createElement("button");
    shot.type = "button";
    shot.className = "shot";
    shot.setAttribute("aria-label", "View match, " + percent(match.similarity) + " percent");
    const img = document.createElement("img");
    img.src = previewUrl(match.file_id);
    img.alt = match.name || "Matched photo";
    img.loading = "lazy";
    if (match.width && match.height) img.style.aspectRatio = match.width + " / " + match.height;
    img.addEventListener("error", () => card.classList.add("is-broken"));
    const box = document.createElement("span");
    box.className = "box";
    placeBox(box, match.bbox);
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = percent(match.similarity) + "%";
    shot.append(img, box, badge);
    shot.addEventListener("click", () => openLightbox(index));
    const jump = document.createElement("a");
    jump.className = "jump";
    jump.href = driveFileUrl(match.file_id);
    jump.target = "_blank";
    jump.rel = "noopener noreferrer";
    jump.textContent = "Drive";
    card.append(shot, jump);
    grid.appendChild(card);
  });
}

function render() {
  if (state.health) {
    $("enginePill").textContent = state.health.engine_label || "CPU";
    const mode = state.health.drive_mode === "api" ? "Drive API" : "Public Drive links";
    $("engineFoot").textContent = (state.health.engine_label || "CPU") + " · " + mode;
    const warn = $("engineWarn");
    warn.hidden = Boolean(state.health.accurate);
    warn.textContent = "Accurate face models aren't available, so matches will be rough.";
  }
  renderPanel();
  renderResults();
  if (state.lightbox >= 0) paintLightbox();
}

async function refreshFolders() {
  const data = await getJSON("/api/folders");
  state.folders = data.folders || [];
}

async function selectFolder(id) {
  const folder = await getJSON("/api/folders/" + encodeURIComponent(id));
  state.folder = folder;
  $("folderUrl").value = folder.url || "";
  try { localStorage.setItem("facescan.folder", id); } catch (err) { /* private mode */ }
  ensurePoll();
  render();
  maybeSearch();
}

function ensurePoll() {
  clearInterval(state.pollTimer);
  if (!state.folder) return;
  if (state.folder.status === "discovering" || state.folder.status === "indexing") {
    state.pollTimer = setInterval(poll, 1000);
  }
}

async function poll() {
  if (!state.folder) return;
  try {
    const previous = state.folder.indexed_photos;
    const previousStatus = state.folder.status;
    const folder = await getJSON("/api/folders/" + encodeURIComponent(state.folder.id));
    state.folder = folder;
    if (folder.status !== "discovering" && folder.status !== "indexing") clearInterval(state.pollTimer);
    if (folder.status !== previousStatus) {
      refreshFolders().then(render).catch(() => {});
    }
    render();
    if (state.token && folder.indexed_photos !== previous) maybeSearch();
  } catch (err) {
    clearInterval(state.pollTimer);
  }
}

async function maybeSearch() {
  if (!state.token || !state.folder || !state.health) {
    render();
    return;
  }
  if (state.searching) {
    state.searchQueued = true;
    return;
  }
  state.searching = true;
  render();
  try {
    const data = await postJSON("/api/search", {
      token: state.token,
      face_index: state.faceIndex,
      folder_id: state.folder.id,
      threshold: state.health.threshold_min,
      limit: 400,
    });
    state.matches = data.matches || [];
    state.searchedFaces = data.searched_faces || 0;
    state.searchError = "";
    state.searched = true;
  } catch (err) {
    state.searchError = err.message;
    state.matches = [];
    state.searched = true;
  } finally {
    state.searching = false;
    render();
    if (state.searchQueued) {
      state.searchQueued = false;
      maybeSearch();
    }
  }
}

async function onFolderSubmit(event) {
  event.preventDefault();
  const url = $("folderUrl").value.trim();
  state.folderPosting = true;
  render();
  try {
    const folder = await postJSON("/api/folders", { url });
    state.folder = folder;
    try { localStorage.setItem("facescan.folder", folder.id); } catch (err) { /* private mode */ }
    await refreshFolders();
    ensurePoll();
    maybeSearch();
  } catch (err) {
    toast(err.message);
  } finally {
    state.folderPosting = false;
    render();
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    toast("Choose a photo.");
    return;
  }
  const body = new FormData();
  body.append("file", file);
  state.uploading = true;
  render();
  try {
    const response = await fetch("/api/detect", { method: "POST", body });
    const data = await readBody(response);
    state.token = data.token;
    state.faces = data.faces || [];
    state.faceIndex = 0;
    state.matches = [];
    state.searched = false;
    state.searchError = "";
    await maybeSearch();
  } catch (err) {
    toast(err.message);
  } finally {
    state.uploading = false;
    render();
  }
}

async function removeFolder() {
  if (!state.folder) return;
  if (!window.confirm("Remove the index for this folder? Photos on Google Drive stay put.")) return;
  const id = state.folder.id;
  try {
    const response = await fetch("/api/folders/" + encodeURIComponent(id), { method: "DELETE" });
    await readBody(response);
  } catch (err) {
    toast(err.message);
    return;
  }
  if (state.folder && state.folder.id === id) {
    state.folder = null;
    state.matches = [];
    state.searched = false;
    $("folderUrl").value = "";
    try { localStorage.removeItem("facescan.folder"); } catch (err) { /* private mode */ }
  }
  clearInterval(state.pollTimer);
  closeLightbox();
  await refreshFolders().catch(() => {});
  render();
}

async function downloadZip() {
  const ids = visibleMatches().map((match) => match.file_id);
  if (!ids.length || state.zipBusy) return;
  state.zipBusy = true;
  render();
  try {
    const response = await fetch("/api/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_ids: ids }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(detailMessage(data.detail) || "Download failed.");
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "facescan-matches.zip";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1500);
  } catch (err) {
    toast(err.message);
  } finally {
    state.zipBusy = false;
    render();
  }
}

function openLightbox(index) {
  state.lightbox = index;
  $("lightbox").hidden = false;
  document.body.style.overflow = "hidden";
  paintLightbox();
  $("lbClose").focus();
}

function closeLightbox() {
  state.lightbox = -1;
  $("lightbox").hidden = true;
  document.body.style.overflow = "";
}

function stepLightbox(delta) {
  const visible = visibleMatches();
  if (!visible.length || state.lightbox < 0) return;
  state.lightbox = (state.lightbox + delta + visible.length) % visible.length;
  paintLightbox();
}

function paintLightbox() {
  const visible = visibleMatches();
  if (state.lightbox < 0 || state.lightbox >= visible.length) {
    closeLightbox();
    return;
  }
  const match = visible[state.lightbox];
  $("lbRef").src = selectedCrop();
  $("lbImg").src = previewUrl(match.file_id);
  $("lbImg").alt = match.name || "Matched photo";
  placeBox($("lbBox"), match.bbox);
  $("lbCap").textContent = percent(match.similarity) + "% match" + (match.name ? " · " + match.name : "");
  $("lbOpen").href = driveFileUrl(match.file_id);
  $("lbCount").textContent = (state.lightbox + 1) + " of " + visible.length;
}

function applyHealth(health) {
  state.health = health;
  const slider = $("threshold");
  slider.min = String(health.threshold_min);
  slider.max = String(health.threshold_max);
  slider.setAttribute("aria-valuemin", String(Math.round(health.threshold_min * 100)));
  slider.setAttribute("aria-valuemax", String(Math.round(health.threshold_max * 100)));
  let stored = null;
  try { stored = localStorage.getItem("facescan.threshold"); } catch (err) { /* private mode */ }
  const initial = stored != null ? Number(stored) : Number(health.default_threshold);
  state.threshold = clamp(initial, Number(health.threshold_min), Number(health.threshold_max));
}

function bind() {
  $("folderForm").addEventListener("submit", onFolderSubmit);
  $("folderUrl").addEventListener("input", () => {
    $("folderBtn").textContent = folderButtonLabel();
    $("folderBtn").disabled = state.folderPosting || $("folderBtn").textContent === "Indexing…";
  });
  $("removeFolder").addEventListener("click", removeFolder);
  $("browse").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", () => {
    const file = $("file").files && $("file").files[0];
    $("file").value = "";
    uploadFile(file);
  });
  const drop = $("drop");
  drop.addEventListener("click", (event) => {
    if (event.target === $("browse") || event.target === $("file")) return;
    $("file").click();
  });
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("file").click();
    }
  });
  ["dragenter", "dragover"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
    });
  });
  drop.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    uploadFile(file);
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  document.addEventListener("paste", (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) uploadFile(file);
        return;
      }
    }
  });
  $("threshold").addEventListener("input", () => {
    state.threshold = Number($("threshold").value);
    try { localStorage.setItem("facescan.threshold", String(state.threshold)); } catch (err) { /* private mode */ }
    updateSliderFill();
    renderResults();
    if (state.lightbox >= 0) paintLightbox();
  });
  $("downloadBtn").addEventListener("click", downloadZip);
  $("lbClose").addEventListener("click", closeLightbox);
  $("lbPrev").addEventListener("click", () => stepLightbox(-1));
  $("lbNext").addEventListener("click", () => stepLightbox(1));
  $("lightbox").addEventListener("click", (event) => {
    if (event.target && event.target.dataset && event.target.dataset.close) closeLightbox();
  });
  document.addEventListener("keydown", (event) => {
    if ($("lightbox").hidden) return;
    if (event.key === "Escape") closeLightbox();
    if (event.key === "ArrowRight") stepLightbox(1);
    if (event.key === "ArrowLeft") stepLightbox(-1);
  });
}

async function init() {
  bind();
  try {
    applyHealth(await getJSON("/api/health"));
  } catch (err) {
    toast("The server isn't responding.");
    return;
  }
  try {
    await refreshFolders();
  } catch (err) {
    toast(err.message);
  }
  let last = null;
  try { last = localStorage.getItem("facescan.folder"); } catch (err) { /* private mode */ }
  const known = last && state.folders.some((folder) => folder.id === last);
  try {
    if (known) await selectFolder(last);
    else if (state.folders[0]) await selectFolder(state.folders[0].id);
    else render();
  } catch (err) {
    toast(err.message);
    render();
  }
}

init();
