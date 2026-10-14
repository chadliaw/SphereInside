// 拍攝、讀入 ZIP、手動對齊與拼貼、產生展開圖。
"use strict";

// ---- 方向 ----------------------------------------------------------------
// 世界座標與 Python 端相同：+x 朝方位 0°，+y 朝方位 270°，+z 朝正上方。

const RAD = Math.PI / 180;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// 裝置座標（x 朝螢幕右方、y 朝螢幕上方、z 朝使用者）→ 世界座標。
// alpha、beta、gamma 是 deviceorientation 事件的三個角度。
function deviceToWorld(alpha, beta, gamma) {
  const cA = Math.cos(alpha * RAD), sA = Math.sin(alpha * RAD);
  const cB = Math.cos(beta * RAD), sB = Math.sin(beta * RAD);
  const cG = Math.cos(gamma * RAD), sG = Math.sin(gamma * RAD);
  // 規格定義的矩陣，結果是（東、北、上）
  const m = [
    [cA * cG - sA * sB * sG, -cB * sA, cG * sA * sB + cA * sG],
    [cG * sA + cA * sB * sG, cA * cB, sA * sG - cA * cG * sB],
    [-cB * sG, sB, cB * cG],
  ];
  return (v) => {
    const e = dot(m[0], v), n = dot(m[1], v), u = dot(m[2], v);
    return [n, -e, u];
  };
}

// 鏡頭的前、右、上（與 sphereinside/camera.py 的 basis 相同）
function basis(azimuth, elevation, roll) {
  const th = -azimuth * RAD, e = elevation * RAD, ro = roll * RAD;
  const fwd = [Math.cos(e) * Math.cos(th), Math.cos(e) * Math.sin(th), Math.sin(e)];
  const right0 = [Math.sin(th), -Math.cos(th), 0];
  const up0 = cross(right0, fwd);
  const c = Math.cos(ro), s = Math.sin(ro);
  return {
    fwd,
    right: right0.map((x, i) => c * x + s * up0[i]),
    up: right0.map((x, i) => -s * x + c * up0[i]),
  };
}

// basis 的反運算：由鏡頭的前、右兩個方向求方位、仰角與傾斜
function paramsFromBasis(fwd, right) {
  const azimuth = ((Math.atan2(-fwd[1], fwd[0]) / RAD) % 360 + 360) % 360;
  const elevation = Math.asin(Math.max(-1, Math.min(1, fwd[2]))) / RAD;
  const ref = basis(azimuth, elevation, 0);
  const roll = Math.atan2(dot(right, ref.up), dot(right, ref.right)) / RAD;
  return { azimuth, elevation, roll };
}

// 由裝置的姿態求後鏡頭的方位、仰角與傾斜。screenAngle 是 screen.orientation.angle。
function pose(alpha, beta, gamma, screenAngle) {
  const toWorld = deviceToWorld(alpha, beta, gamma);
  const a = screenAngle * RAD;
  return paramsFromBasis(toWorld([0, 0, -1]), toWorld([Math.cos(a), -Math.sin(a), 0]));
}

// 沿影像寬度方向的視角
function horizontalFov(longFov, width, height) {
  const focal = Math.max(width, height) / 2 / Math.tan(longFov * RAD / 2);
  return 2 * Math.atan(width / 2 / focal) / RAD;
}

// ---- 涵蓋範圍 --------------------------------------------------------------

const FULL_CROP = [0, 0, 1, 1];

function fibonacciSphere(n) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const z = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - z * z), t = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
    points.push([r * Math.cos(t), r * Math.sin(t), z]);
  }
  return points;
}

// 方向 d 是否落在這張照片（剪裁後）的範圍內；沒有使用的照片不算
function covers(shot, d) {
  if (shot.use === false) return false;
  const b = shot.basis, c = dot(d, b.fwd);
  if (c <= 1e-9) return false;
  const [left, top, right, bottom] = shot.crop || FULL_CROP;
  const focal = shot.width / 2 / Math.tan(shot.hfov * RAD / 2);
  const u = shot.width / 2 + focal * dot(d, b.right) / c, v = shot.height / 2 - focal * dot(d, b.up) / c;
  return u >= left * shot.width && u <= right * shot.width && v >= top * shot.height && v <= bottom * shot.height;
}

function coverage(shots, points) {
  return points.map((d) => shots.some((s) => covers(s, d)));
}

// ---- ZIP ------------------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// files: [{ name, data: Uint8Array }] → Blob（只儲存、不壓縮）
function makeZip(files) {
  const encoder = new TextEncoder();
  const parts = [], central = [];
  let offset = 0;
  for (const file of files) {
    const name = encoder.encode(file.name), crc = crc32(file.data), size = file.data.length;
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true);
    local.setUint16(10, 0, true);
    local.setUint16(12, 0x21, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, size, true);
    local.setUint32(22, size, true);
    local.setUint16(26, name.length, true);
    parts.push(local.buffer, name, file.data);

    const entry = new DataView(new ArrayBuffer(46));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    entry.setUint16(6, 20, true);
    entry.setUint16(8, 0x0800, true);
    entry.setUint16(14, 0x21, true);
    entry.setUint32(16, crc, true);
    entry.setUint32(20, size, true);
    entry.setUint32(24, size, true);
    entry.setUint16(28, name.length, true);
    entry.setUint32(42, offset, true);
    central.push(entry.buffer, name);
    offset += 30 + name.length + size;
  }
  const centralSize = central.reduce((n, p) => n + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
}

// 讀取 ZIP：回傳 { 檔名: Uint8Array }。支援不壓縮與 deflate 壓縮
async function readZip(buffer) {
  const view = new DataView(buffer), bytes = new Uint8Array(buffer), decoder = new TextDecoder();
  let end = -1;
  for (let i = buffer.byteLength - 22; i >= Math.max(0, buffer.byteLength - 22 - 65535); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { end = i; break; }
  }
  if (end < 0) throw new Error(t("zip.notZip"));
  const files = {};
  let p = view.getUint32(end + 16, true);
  for (let n = view.getUint16(end + 10, true); n > 0; n--) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new Error(t("zip.corrupt"));
    const method = view.getUint16(p + 10, true), size = view.getUint32(p + 20, true);
    const nameLength = view.getUint16(p + 28, true), extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
    const name = decoder.decode(bytes.subarray(p + 46, p + 46 + nameLength));
    const local = view.getUint32(p + 42, true);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    const raw = bytes.subarray(start, start + size);
    if (method === 0) files[name] = raw;
    else if (method === 8) files[name] = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
    p += 46 + nameLength + extra + comment;
  }
  return files;
}

// ---- 拍攝 ------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const POINTS = fibonacciSphere(3000);
// { name, blob, azimuth, elevation, roll, hfov, width, height, basis, priority, use, crop, thumb }
// 拍攝或讀入 ZIP 後都放在這裡
const shots = [];
let orientation = null, covered = POINTS.map(() => false);

function currentPose() {
  if (!orientation) return null;
  const angle = screen.orientation ? screen.orientation.angle : (window.orientation || 0);
  return pose(orientation.alpha, orientation.beta, orientation.gamma, angle);
}

function say(key) {
  if (key) setText($("message"), key);
  else setRawText($("message"), "");
}

function drawMap(now) {
  const canvas = $("map"), g = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  g.clearRect(0, 0, w, h);
  POINTS.forEach((d, i) => {
    const az = ((Math.atan2(-d[1], d[0]) / RAD) + 360) % 360, el = Math.asin(d[2]) / RAD;
    g.fillStyle = covered[i] ? "#4cc38a" : "#b5452f";
    g.fillRect(az / 360 * w - 2, (90 - el) / 180 * h - 2, 4, 4);
  });
  if (now) {
    const x = now.azimuth / 360 * w, y = (90 - now.elevation) / 180 * h;
    g.strokeStyle = "#fff";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(x - 10, y); g.lineTo(x + 10, y);
    g.moveTo(x, y - 10); g.lineTo(x, y + 10);
    g.stroke();
  }
}

function coveredFraction() {
  return covered.filter(Boolean).length / POINTS.length;
}

function refresh() {
  covered = coverage(shots, POINTS);
  $("count").textContent = shots.length;
  $("cover").textContent = Math.floor(coveredFraction() * 100) + "%";
  $("undo").disabled = $("download").disabled = shots.length === 0;
  updateMake();
}

function tick() {
  const now = currentPose();
  if (now) {
    $("az").textContent = now.azimuth.toFixed(0) + "°";
    $("el").textContent = now.elevation.toFixed(0) + "°";
    $("roll").textContent = now.roll.toFixed(0) + "°";
  }
  drawMap(now);
  setTimeout(tick, 100);
}

// 帶有翻譯鍵值的錯誤
const failure = (key) => Object.assign(new Error(t(key)), { key });

async function start() {
  say(null);
  try {
    if (typeof DeviceOrientationEvent === "undefined") throw failure("shoot.noSensorApi");
    if (typeof DeviceOrientationEvent.requestPermission === "function") {
      if (await DeviceOrientationEvent.requestPermission() !== "granted") throw failure("shoot.permission");
    }
    window.addEventListener("deviceorientation", (e) => {
      if (e.alpha !== null && e.beta !== null && e.gamma !== null) orientation = { alpha: e.alpha, beta: e.beta, gamma: e.gamma };
    });
    if (!navigator.mediaDevices) throw failure("shoot.https");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } },
    });
    $("video").srcObject = stream;
    await $("video").play();
  } catch (error) {
    if (error.key) say(error.key);
    else setRawText($("message"), error.message || String(error));
    return;
  }
  $("intro").hidden = true;
  $("shoot").hidden = false;
  refresh();
  tick();
}

function shoot() {
  const now = currentPose(), video = $("video");
  if (!now) return say("shoot.noReading");
  if (!video.videoWidth) return say("shoot.notReady");
  say(null);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const hfov = horizontalFov(Number($("fov").value) || 65, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    shots.push({
      blob, hfov, width: canvas.width, height: canvas.height,
      azimuth: now.azimuth, elevation: now.elevation, roll: now.roll,
      basis: basis(now.azimuth, now.elevation, now.roll),
      priority: 0, use: true, crop: [...FULL_CROP],
    });
    refresh();
  }, "image/jpeg", 0.92);
}

// ---- 照片與對照檔 ------------------------------------------------------------

// 照片與對照檔，與下載的 ZIP 內容相同；沒有使用的照片也一起保留，之後可以再加回來
async function collectFiles() {
  const round = (x, k = 100) => Math.round(x * k) / k;
  const files = [], photos = [];
  for (let i = 0; i < shots.length; i++) {
    const s = shots[i], name = s.name || String(i + 1).padStart(3, "0") + ".jpg";
    files.push({ name, data: new Uint8Array(await s.blob.arrayBuffer()) });
    const entry = { file: name, azimuth: round(s.azimuth), elevation: round(s.elevation), roll: round(s.roll), hfov: round(s.hfov) };
    if (s.priority) entry.priority = s.priority;
    if (s.crop && s.crop.some((x, k) => x !== FULL_CROP[k])) entry.crop = s.crop.map((x) => round(x, 10000));
    if (s.use === false) entry.use = false;
    photos.push(entry);
  }
  files.push({ name: "meta.json", data: new TextEncoder().encode(JSON.stringify({ photos }, null, 2)) });
  return files;
}

async function download() {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(makeZip(await collectFiles()));
  link.download = "sphereinside-photos.zip";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 60000);
}

// 讀入先前下載的 ZIP，取代目前的照片
async function loadZip() {
  const file = $("zip").files[0];
  if (!file) return;
  setText($("loaded"), "zip.reading");
  try {
    const files = await readZip(await file.arrayBuffer());
    const metaName = Object.keys(files).find((n) => n.split("/").pop() === "meta.json");
    if (!metaName) throw new Error(t("zip.noMeta"));
    const meta = JSON.parse(new TextDecoder().decode(files[metaName]));
    const folder = metaName.slice(0, metaName.length - "meta.json".length);
    const loaded = [];
    for (const entry of meta.photos) {
      const data = files[folder + entry.file];
      if (!data) throw new Error(t("zip.missing", { file: entry.file }));
      const hfov = entry.hfov ?? meta.hfov;
      if (hfov == null) throw new Error(t("zip.noHfov", { file: entry.file }));
      const blob = new Blob([data], { type: entry.file.toLowerCase().endsWith(".png") ? "image/png" : "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      const roll = entry.roll || 0;
      loaded.push({
        name: entry.file.split("/").pop(), blob, hfov, width: bitmap.width, height: bitmap.height,
        azimuth: entry.azimuth, elevation: entry.elevation, roll, basis: basis(entry.azimuth, entry.elevation, roll),
        priority: entry.priority || 0, use: entry.use !== false, crop: Array.isArray(entry.crop) ? entry.crop.map(Number) : [...FULL_CROP],
      });
      bitmap.close();
    }
    shots.splice(0, shots.length, ...loaded);
    setText($("loaded"), "zip.loaded", { n: loaded.length });
    refresh();
  } catch (error) {
    setRawText($("loaded"), error.message || String(error));
  }
}

// ---- 產生展開圖 ------------------------------------------------------------

let worker = null, building = false;

function updateMake() {
  $("make").disabled = building || shots.length === 0;
  $("openAlign").disabled = $("openCompose").disabled = shots.length === 0;
}

function log(text) {
  $("log").hidden = false;
  $("log").textContent += text + "\n";
}

function finish() {
  building = false;
  updateMake();
}

async function make() {
  building = true;
  updateMake();
  $("log").textContent = "";
  $("result").hidden = true;
  const job = {
    lang,
    files: await collectFiles(),
    args: [
      "--level", $("level").value, "--diameter", String(Number($("diameter").value) || 150),
      "--paper", $("paper").value, "--dpi", $("dpi").value, "--labels", $("labels").value,
    ],
  };
  if ($("gaps").checked) job.args.push("--allow-gaps");

  if (!worker) {
    worker = new Worker("build-worker.js");
    worker.onmessage = (event) => {
      const m = event.data;
      if (m.type === "log") return log(m.text);
      if (m.type === "done") {
        const link = $("result");
        if (link.href) URL.revokeObjectURL(link.href);
        link.href = URL.createObjectURL(new Blob([m.pdf], { type: "application/pdf" }));
        link.hidden = false;
        log(t("build.done"));
      } else {
        log(t("build.failed"));
      }
      finish();
    };
    worker.onerror = (event) => {
      log(t("build.workerError", { message: event.message || t("build.cannotStart") }));
      worker = null;
      finish();
    };
  }
  worker.postMessage(job);
}

// ---- 手動對齊與拼貼 ------------------------------------------------------------

let aligner = null, polyhedra = null, composeTimer = 0;

function shotName(i) {
  return shots[i].name || t("align.photoN", { n: i + 1 });
}

function showPhotoInfo() {
  const a = aligner;
  if (!a) return;
  const s = a.shots[a.selected];
  if (!s) setText($("photoInfo"), "align.selectHint");
  else setText($("photoInfo"), s.use === false ? "align.selectedPreview" : "align.selected", { name: shotName(a.selected) });
  $("photoFields").hidden = $("cropFields").hidden = $("orderRow").hidden = !s;
  if (s) {
    const digits = { azimuth: 2, elevation: 2, roll: 2, hfov: 2, priority: 0 };
    // 正在輸入的欄位不要被覆蓋
    document.querySelectorAll("[data-field]").forEach((input) => {
      if (input !== document.activeElement) input.value = Number(s[input.dataset.field] || 0).toFixed(digits[input.dataset.field]);
    });
    document.querySelectorAll("[data-crop]").forEach((input) => {
      if (input !== document.activeElement) input.value = (s.crop[Number(input.dataset.crop)] * 100).toFixed(1);
    });
    setText($("useToggle"), s.use === false ? "align.paste" : "align.remove");
  }
  if (a.vertex < 0) setText($("vertexInfo"), "align.vertexCount", { n: a.order.length });
  else setText($("vertexInfo"), "align.vertexN", { i: a.vertex + 1, n: a.order.length });
  scheduleCompose();
}

// 候選照片與狀態列；視角或照片改變後稍等一下再更新，拖曳時才不會卡
function scheduleCompose() {
  clearTimeout(composeTimer);
  composeTimer = setTimeout(updateCompose, 120);
}

function updateCompose() {
  const a = aligner;
  if (!a) return;
  covered = coverage(shots, POINTS);
  setText($("alignStatus"), "align.status", {
    used: shots.filter((s) => s.use !== false).length, n: shots.length, cover: Math.floor(coveredFraction() * 100),
  });

  const center = a.viewBasis().fwd;
  const near = shots
    .map((s, i) => ({ i, angle: Math.acos(Math.max(-1, Math.min(1, dot(center, s.basis.fwd)))) / RAD }))
    .filter((c) => c.angle < 75)
    .sort((x, y) => x.angle - y.angle)
    .slice(0, 8);
  const cards = $("cards");
  cards.replaceChildren();
  if (near.length === 0) {
    const p = document.createElement("p");
    setText(p, "align.none");
    cards.append(p);
  }
  for (const { i, angle } of near) {
    const s = shots[i], card = document.createElement("div");
    card.className = "card" + (i === a.selected ? " current" : "");
    const img = document.createElement("img");
    img.src = s.thumb || "";
    img.alt = shotName(i);
    img.addEventListener("click", () => a.select(i));
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = shotName(i);
    const meta = document.createElement("div");
    meta.className = "meta" + (s.use !== false ? " used" : "");
    meta.textContent = `${t(s.use !== false ? "align.inUse" : "align.notUsed")} · ${t("align.away", { deg: Math.round(angle) })}`;
    const button = document.createElement("button");
    setText(button, s.use !== false ? "align.remove" : "align.paste");
    button.addEventListener("click", () => {
      a.setUse(i, s.use === false);
      a.select(i);
    });
    card.append(img, name, meta, button);
    cards.append(card);
  }
}

function setMode(mode) {
  aligner.mode = mode;
  for (const [id, m] of [["modeView", "view"], ["modeAdjust", "adjust"], ["modeCrop", "crop"]]) $(id).setAttribute("aria-pressed", mode === m);
}

// compose 為真時從空白開始（若先前已經拼過，就接著上次的進度）
async function openAlign(compose) {
  $("align").hidden = false;
  document.body.style.overflow = "hidden";
  setText($("alignStatus"), "align.loading");
  try {
    if (!polyhedra) polyhedra = await (await fetch("polyhedra.json")).json();
    const level = $("level").value;
    aligner = new Aligner($("gl"), $("overlay"), shots, polyhedra[level], Number(level));
    aligner.onchange = showPhotoInfo;
    aligner.onview = scheduleCompose;
    await aligner.init((i, n) => setText($("alignStatus"), "align.loadingN", { i: i + 1, n }));
    aligner.attach();
    aligner.translucent = $("translucent").checked;
    aligner.showMesh = $("showMesh").checked;
    aligner.allFov = $("allFov").checked;
    setMode("view");
    if (compose && shots.every((s) => s.use !== false)) aligner.setAll(false);
    aligner.goToVertex(1);
  } catch (error) {
    setRawText($("alignStatus"), error.message || String(error));
  }
}

function closeAlign() {
  $("align").hidden = true;
  document.body.style.overflow = "";
  if (aligner && aligner.gl) aligner.gl.getExtension("WEBGL_lose_context")?.loseContext();
  // 換上新的畫布，下次開啟時才能重新取得 WebGL
  for (const id of ["gl", "overlay"]) $(id).replaceWith($(id).cloneNode(false));
  aligner = null;
  refresh();
}

// 微調的角度（度）
function stepSize(e) {
  return Math.max(0.01, Number($("step").value) || 0.2) * (e && e.shiftKey ? 5 : 1);
}

function alignKey(e) {
  if (!aligner || $("align").hidden || e.target.tagName === "INPUT") return;
  const step = stepSize(e);
  let action = {
    ArrowLeft: () => aligner.nudge(-step, 0), ArrowRight: () => aligner.nudge(step, 0),
    ArrowUp: () => aligner.nudge(0, step), ArrowDown: () => aligner.nudge(0, -step),
    "[": () => aligner.roll(-step), "{": () => aligner.roll(-step), "]": () => aligner.roll(step), "}": () => aligner.roll(step),
    "-": () => aligner.widen(-step), "_": () => aligner.widen(-step), "=": () => aligner.widen(step), "+": () => aligner.widen(step),
  }[e.key];
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") action = () => aligner.undo();
  if (!action) return;
  e.preventDefault();
  action();
}

// ---- 事件 ------------------------------------------------------------------

$("lang").value = lang;
$("lang").addEventListener("change", (e) => {
  chooseLanguage(e.target.value);
  if (aligner) showPhotoInfo();
});
$("start").addEventListener("click", start);
$("shutter").addEventListener("click", shoot);
$("undo").addEventListener("click", () => { shots.pop(); refresh(); });
$("download").addEventListener("click", download);
$("zip").addEventListener("change", loadZip);
$("make").addEventListener("click", make);
$("openAlign").addEventListener("click", () => openAlign(false));
$("openCompose").addEventListener("click", () => openAlign(true));
$("alignDone").addEventListener("click", closeAlign);
$("alignDownload").addEventListener("click", download);
$("alignUndo").addEventListener("click", () => aligner && aligner.undo());
$("alignReset").addEventListener("click", () => aligner && aligner.resetSelected());
$("modeView").addEventListener("click", () => aligner && setMode("view"));
$("modeAdjust").addEventListener("click", () => aligner && setMode("adjust"));
$("modeCrop").addEventListener("click", () => aligner && setMode("crop"));
$("prevVertex").addEventListener("click", () => aligner && aligner.goToVertex(-1));
$("nextVertex").addEventListener("click", () => aligner && aligner.goToVertex(1));
$("translucent").addEventListener("change", (e) => { if (aligner) { aligner.translucent = e.target.checked; aligner.draw(); } });
$("showMesh").addEventListener("change", (e) => { if (aligner) { aligner.showMesh = e.target.checked; aligner.draw(); } });
$("allFov").addEventListener("change", (e) => { if (aligner) aligner.allFov = e.target.checked; });
document.querySelectorAll("[data-nudge]").forEach((b) => b.addEventListener("click", (e) => {
  const [x, y] = b.dataset.nudge.split(",").map(Number), step = stepSize(e);
  if (aligner) aligner.nudge(x * step, y * step);
}));
document.querySelectorAll("[data-roll]").forEach((b) => b.addEventListener("click", (e) => aligner && aligner.roll(Number(b.dataset.roll) * stepSize(e))));
document.querySelectorAll("[data-zoom]").forEach((b) => b.addEventListener("click", (e) => aligner && aligner.widen(Number(b.dataset.zoom) * stepSize(e))));
document.querySelectorAll("[data-field]").forEach((input) => input.addEventListener("change", () => {
  if (aligner) aligner.setValue(input.dataset.field, Number(input.value));
  showPhotoInfo();
}));
document.querySelectorAll("[data-crop]").forEach((input) => input.addEventListener("change", () => {
  if (aligner) aligner.setCrop(Number(input.dataset.crop), Number(input.value) / 100);
  showPhotoInfo();
}));
$("cropReset").addEventListener("click", () => aligner && aligner.resetCrop());
$("useToggle").addEventListener("click", () => {
  if (aligner && aligner.selected >= 0) aligner.setUse(aligner.selected, shots[aligner.selected].use === false);
});
$("toFront").addEventListener("click", () => aligner && aligner.arrange("front"));
$("toBack").addEventListener("click", () => aligner && aligner.arrange("back"));
$("autoOrder").addEventListener("click", () => aligner && aligner.arrange("auto"));
$("useNone").addEventListener("click", () => aligner && aligner.setAll(false));
$("useAll").addEventListener("click", () => aligner && aligner.setAll(true));
document.addEventListener("keydown", alignKey);
window.addEventListener("resize", () => aligner && aligner.resize());

applyLanguage();
