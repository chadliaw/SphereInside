// 手動對齊與拼貼：從球心往外看，拖曳照片來修正它的方位、仰角、傾斜與視角，
// 選擇要用哪些照片、誰蓋在上面，以及剪裁。
// 需要 app.js 裡的 basis、paramsFromBasis、dot、cross 與 RAD，以及 i18n.js 的 t。
"use strict";

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// 每個像素就是一個方向；與 Python 端相同，取順序最大、再來是鏡頭中心最接近的照片
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;
uniform mediump sampler2DArray photos;
// 每張照片一列：前、右、上、(tan 半寬, tan 半高, 順序, 是否使用)、剪裁（左, 上, 右, 下）
uniform highp sampler2D params;
uniform int count;
uniform int selected;
uniform float overlayInside;   // 選取的照片疊在上面時，剪裁範圍內的不透明度
uniform float overlayOutside;  // 剪裁範圍外的不透明度（剪裁時顯示被剪掉的部分）
uniform vec3 viewF, viewR, viewU;
uniform vec2 viewTan;
uniform vec2 resolution;
uniform float layerHeight;
out vec4 outColor;
const float PRIORITY_WEIGHT = 10.0;  // 與 sphereinside/camera.py 相同

// 方向 d 在照片 i 上的相對位置（0–1）；在鏡頭後方時回傳 false
bool project(int i, vec3 d, out vec2 uv, out float c) {
  vec3 f = texelFetch(params, ivec2(0, i), 0).xyz;
  c = dot(d, f);
  if (c <= 1e-4) return false;
  vec3 r = texelFetch(params, ivec2(1, i), 0).xyz;
  vec3 u = texelFetch(params, ivec2(2, i), 0).xyz;
  vec2 t = texelFetch(params, ivec2(3, i), 0).xy;
  uv = vec2(0.5 + dot(d, r) / c / (2.0 * t.x), 0.5 - dot(d, u) / c / (2.0 * t.y));
  return true;
}

bool within(vec2 uv, vec4 box) {
  return uv.x >= box.x && uv.y >= box.y && uv.x <= box.z && uv.y <= box.w;
}

vec3 photo(int i, vec2 uv) {
  float t = texelFetch(params, ivec2(3, i), 0).y;
  float lod = max(0.0, log2(layerHeight / (2.0 * t) / (resolution.y / (2.0 * viewTan.y))));
  return textureLod(photos, vec3(uv, float(i)), lod).rgb;
}

void main() {
  vec2 ndc = gl_FragCoord.xy / resolution * 2.0 - 1.0;
  vec3 d = normalize(viewF + ndc.x * viewTan.x * viewR + ndc.y * viewTan.y * viewU);
  bool overlay = selected >= 0 && (overlayInside > 0.0 || overlayOutside > 0.0);
  int best = -1;
  float bestScore = -1e9;
  vec2 bestUv = vec2(0.0);
  for (int i = 0; i < count; i++) {
    vec4 p = texelFetch(params, ivec2(3, i), 0);
    if (p.w < 0.5 || (overlay && i == selected)) continue;  // 沒用到的照片不畫；疊在上面的另外畫
    vec2 uv;
    float c;
    if (!project(i, d, uv, c) || !within(uv, texelFetch(params, ivec2(4, i), 0))) continue;
    float score = p.z * PRIORITY_WEIGHT + c;
    if (score > bestScore) { best = i; bestScore = score; bestUv = uv; }
  }
  vec3 color = best < 0 ? vec3(0.2, 0.22, 0.25) : photo(best, bestUv);
  if (overlay) {
    vec2 uv;
    float c;
    if (project(selected, d, uv, c) && within(uv, vec4(0.0, 0.0, 1.0, 1.0))) {
      float alpha = within(uv, texelFetch(params, ivec2(4, selected), 0)) ? overlayInside : overlayOutside;
      color = mix(color, photo(selected, uv), alpha);
    }
  }
  outColor = vec4(color, 1.0);
}`;

const TEXTURE_BUDGET = 96e6;  // 所有照片在顯示卡上的大小上限（位元組）
const VIEW_FOV = { 0: 120, 1: 95, 2: 70 };  // 檢視頂點時的視角，夠看到周圍一圈的面
const PRIORITY_WEIGHT = 10;  // 順序的權重，與 sphereinside/camera.py 相同
const THUMB_HEIGHT = 120;
const MIN_CROP = 0.05;  // 剪裁後至少保留的寬度或高度（比例）
const PARAM_COLUMNS = 5;

const norm = (v) => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0] / n, v[1] / n, v[2] / n]; };
const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// 繞單位向量 axis 旋轉 angle（弧度）
function rotate(v, axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), k = dot(axis, v) * (1 - c), x = cross(axis, v);
  return [v[0] * c + x[0] * s + axis[0] * k, v[1] * c + x[1] * s + axis[1] * k, v[2] * c + x[2] * s + axis[2] * k];
}

function tanHalf(shot) {
  const w = Math.tan(shot.hfov * RAD / 2);
  return [w, w * shot.height / shot.width];
}

function setTanHalf(shot, w) {
  shot.hfov = 2 * Math.atan(w) / RAD;
}

function setBasis(shot, fwd, right) {
  Object.assign(shot, paramsFromBasis(fwd, right));
  shot.basis = basis(shot.azimuth, shot.elevation, shot.roll);
}

// 方向 d 在照片上的相對位置 [u, v]（0–1，未剪裁）；在鏡頭後方時回傳 null
function photoUv(shot, d) {
  const b = shot.basis, c = dot(d, b.fwd), t = tanHalf(shot);
  if (c <= 1e-4) return null;
  return [0.5 + dot(d, b.right) / c / (2 * t[0]), 0.5 - dot(d, b.up) / c / (2 * t[1])];
}

// 照片上相對位置 [u, v] 的方向
function photoDirection(shot, u, v) {
  const b = shot.basis, t = tanHalf(shot);
  return add(add(b.fwd, b.right, (2 * u - 1) * t[0]), b.up, (1 - 2 * v) * t[1]);
}

class Aligner {
  constructor(canvas, overlay, shots, mesh, level) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.shots = shots;
    for (const s of shots) {
      if (s.use === undefined) s.use = true;
      if (!s.crop) s.crop = [0, 0, 1, 1];
      if (!s.priority) s.priority = 0;
    }
    this.view = { azimuth: 0, elevation: 0, fov: VIEW_FOV[level] || 90 };
    this.selected = -1;
    this.mode = "view";
    this.translucent = false;
    this.showMesh = true;
    this.allFov = true;
    this.history = [];
    this.original = shots.map((s) => this.snapshotOf(s));
    this.pointers = new Map();
    this.gesture = null;
    this.onchange = () => {};
    this.onview = () => {};

    // 印在外側時取色方向會鏡射，所以這裡把多面體也鏡射，才看得出哪一塊會落在哪個面上
    this.vertices = mesh.V.map((v) => [v[0], -v[1], v[2]]);
    const edges = new Set();
    for (const f of mesh.S) for (let k = 0; k < 3; k++) {
      const a = f[k], b = f[(k + 1) % 3];
      edges.add(a < b ? a + "," + b : b + "," + a);
    }
    this.edges = [...edges].map((e) => e.split(",").map(Number));
    // 由上往下、依方位排列，逐一檢視時比較好追蹤
    const band = (v) => Math.round(Math.asin(v[2]) / RAD / 20);
    const azimuth = (v) => (Math.atan2(-v[1], v[0]) + 2 * Math.PI) % (2 * Math.PI);
    this.order = this.vertices.map((_, i) => i).sort((i, j) => {
      const a = this.vertices[i], b = this.vertices[j];
      return band(b) - band(a) || azimuth(a) - azimuth(b);
    });
    this.vertex = -1;
  }

  // ---- 顯示卡 -------------------------------------------------------------

  async init(progress) {
    const gl = this.canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error(t("align.noWebgl"));
    this.gl = gl;
    this.program = this.compile();
    gl.bindVertexArray(gl.createVertexArray());

    const n = this.shots.length;
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const pixels = Math.min(1024 * 1365, TEXTURE_BUDGET / 4 / 1.34 / n);
    this.layer = [Math.min(maxSize, Math.round(Math.sqrt(pixels * 3 / 4))), Math.min(maxSize, Math.round(Math.sqrt(pixels * 4 / 3)))];
    const [w, h] = this.layer;
    const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;

    this.photoTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.photoTexture);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, levels, gl.RGBA8, w, h, n);
    // 照片一律縮放成同樣大小；取樣時用的是 0–1 的相對座標，所以不同長寬比也沒關係
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    const ctx = scratch.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    const thumb = document.createElement("canvas");
    const thumbCtx = thumb.getContext("2d");
    for (let i = 0; i < n; i++) {
      progress(i, n);
      const s = this.shots[i];
      const bitmap = await createImageBitmap(s.blob);
      ctx.drawImage(bitmap, 0, 0, w, h);
      if (!s.thumb) {
        thumb.height = THUMB_HEIGHT;
        thumb.width = Math.round(THUMB_HEIGHT * bitmap.width / bitmap.height);
        thumbCtx.drawImage(bitmap, 0, 0, thumb.width, thumb.height);
        s.thumb = thumb.toDataURL("image/jpeg", 0.75);
      }
      bitmap.close();
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, w, h, 1, gl.RGBA, gl.UNSIGNED_BYTE, scratch);
    }
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.paramTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.paramTexture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, PARAM_COLUMNS, n);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.uploadParams();
    this.resize();
  }

  compile() {
    const gl = this.gl;
    const shader = (type, source) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, source);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, shader(gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    gl.useProgram(p);
    this.uniforms = {};
    for (const name of ["photos", "params", "count", "selected", "overlayInside", "overlayOutside", "viewF", "viewR", "viewU", "viewTan", "resolution", "layerHeight"]) {
      this.uniforms[name] = gl.getUniformLocation(p, name);
    }
    gl.uniform1i(this.uniforms.photos, 0);
    gl.uniform1i(this.uniforms.params, 1);
    return p;
  }

  uploadParams() {
    const row = PARAM_COLUMNS * 4, data = new Float32Array(this.shots.length * row);
    this.shots.forEach((s, i) => {
      const b = s.basis, t = tanHalf(s);
      data.set([...b.fwd, 0, ...b.right, 0, ...b.up, 0, t[0], t[1], s.priority || 0, s.use === false ? 0 : 1, ...s.crop], i * row);
    });
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, PARAM_COLUMNS, this.shots.length, gl.RGBA, gl.FLOAT, data);
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * ratio)), h = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    for (const c of [this.canvas, this.overlay]) {
      c.width = w;
      c.height = h;
    }
    this.draw();
  }

  // ---- 視角與投影 -----------------------------------------------------------

  viewBasis() {
    const b = basis(this.view.azimuth, this.view.elevation, 0);
    const ty = Math.tan(this.view.fov * RAD / 2);
    return { ...b, tan: [ty * this.canvas.width / this.canvas.height, ty] };
  }

  // 畫面上的一點（CSS 像素）→ 方向
  directionAt(x, y) {
    const v = this.viewBasis();
    const nx = x / this.canvas.clientWidth * 2 - 1, ny = 1 - y / this.canvas.clientHeight * 2;
    return norm(add(add(v.fwd, v.right, nx * v.tan[0]), v.up, ny * v.tan[1]));
  }

  // 方向 → 畫布像素；在視線後方時回傳 null
  toScreen(d, v) {
    const c = dot(d, v.fwd);
    if (c <= 1e-3) return null;
    return [(dot(d, v.right) / c / v.tan[0] + 1) / 2 * this.canvas.width, (1 - dot(d, v.up) / c / v.tan[1]) / 2 * this.canvas.height];
  }

  // 方向 d 由哪張照片負責（與 shader 相同的規則）
  pick(d) {
    let best = -1, bestScore = -Infinity;
    this.shots.forEach((s, i) => {
      if (s.use === false) return;
      const uv = photoUv(s, d);
      if (!uv || uv[0] < s.crop[0] || uv[1] < s.crop[1] || uv[0] > s.crop[2] || uv[1] > s.crop[3]) return;
      const score = (s.priority || 0) * PRIORITY_WEIGHT + dot(d, s.basis.fwd);
      if (score > bestScore) {
        best = i;
        bestScore = score;
      }
    });
    return best;
  }

  // ---- 繪製 ---------------------------------------------------------------

  draw() {
    if (!this.gl || this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.render();
      this.onview();
    });
  }

  render() {
    const gl = this.gl, u = this.uniforms, v = this.viewBasis();
    const s = this.shots[this.selected];
    // 預覽還沒貼上的照片、半透明模式、剪裁模式時，選取的照片疊在最上面
    let inside = 0, outside = 0;
    if (s && this.mode === "crop") { inside = 0.85; outside = 0.3; }
    else if (s && s.use === false) inside = 0.65;
    else if (s && this.translucent) inside = 0.5;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform1i(u.count, this.shots.length);
    gl.uniform1i(u.selected, this.selected);
    gl.uniform1f(u.overlayInside, inside);
    gl.uniform1f(u.overlayOutside, outside);
    gl.uniform3fv(u.viewF, v.fwd);
    gl.uniform3fv(u.viewR, v.right);
    gl.uniform3fv(u.viewU, v.up);
    gl.uniform2fv(u.viewTan, v.tan);
    gl.uniform2f(u.resolution, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.layerHeight, this.layer[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.renderOverlay(v);
  }

  renderOverlay(v) {
    const g = this.overlay.getContext("2d"), scale = this.canvas.width / this.canvas.clientWidth;
    g.clearRect(0, 0, this.overlay.width, this.overlay.height);
    const segment = (a, b) => {
      // 切掉視線後方的部分；從球心看，多面體的邊與照片的邊框在畫面上都是直線
      const ca = dot(a, v.fwd) - 0.01, cb = dot(b, v.fwd) - 0.01;
      if (ca <= 0 && cb <= 0) return;
      if (ca <= 0) a = add(a, add(b, a, -1), ca / (ca - cb));
      if (cb <= 0) b = add(b, add(a, b, -1), cb / (cb - ca));
      const p = this.toScreen(a, v), q = this.toScreen(b, v);
      if (p && q) { g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); }
    };
    const box = (s, [u0, v0, u1, v1]) => {
      const c = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map(([x, y]) => photoDirection(s, x, y));
      for (let k = 0; k < 4; k++) segment(c[k], c[(k + 1) % 4]);
    };
    if (this.showMesh) {
      g.beginPath();
      for (const [a, b] of this.edges) segment(this.vertices[a], this.vertices[b]);
      // 深色襯底，白牆上也看得清楚
      g.setLineDash([]);
      g.strokeStyle = "rgba(0,0,0,0.35)";
      g.lineWidth = 3 * scale;
      g.stroke();
      g.strokeStyle = "rgba(255,255,255,0.7)";
      g.lineWidth = 1.2 * scale;
      g.stroke();
      if (this.vertex >= 0) {
        const p = this.toScreen(this.vertices[this.order[this.vertex]], v);
        if (p) {
          g.beginPath();
          g.arc(p[0], p[1], 8 * scale, 0, 2 * Math.PI);
          g.strokeStyle = "#000000";
          g.lineWidth = 4.5 * scale;
          g.stroke();
          g.strokeStyle = "#4cc38a";
          g.lineWidth = 2.5 * scale;
          g.stroke();
        }
      }
    }
    const s = this.shots[this.selected];
    if (s) {
      const color = s.use === false ? "#56ccf2" : "#f2c94c";
      // 整張照片的範圍用虛線，剪裁後保留的範圍用實線
      if (s.crop.some((x, k) => x !== [0, 0, 1, 1][k])) {
        g.beginPath();
        box(s, [0, 0, 1, 1]);
        g.setLineDash([6 * scale, 5 * scale]);
        g.strokeStyle = color;
        g.lineWidth = 1.2 * scale;
        g.stroke();
        g.setLineDash([]);
      }
      g.beginPath();
      box(s, s.crop);
      g.strokeStyle = color;
      g.lineWidth = (this.mode === "crop" ? 3.5 : 2.5) * scale;
      g.stroke();
    }
  }

  // ---- 修改照片 -------------------------------------------------------------

  snapshotOf(s) {
    return [s.azimuth, s.elevation, s.roll, s.hfov, s.priority || 0, s.use !== false, [...s.crop]];
  }

  restore(s, [azimuth, elevation, roll, hfov, priority, use, crop]) {
    Object.assign(s, { azimuth, elevation, roll, hfov, priority, use, crop: [...crop], basis: basis(azimuth, elevation, roll) });
  }

  remember() {
    this.history.push(this.shots.map((s) => this.snapshotOf(s)));
    if (this.history.length > 200) this.history.shift();
  }

  changed() {
    this.uploadParams();
    this.draw();
    this.onchange();
  }

  undo() {
    const state = this.history.pop();
    if (!state) return;
    this.shots.forEach((s, i) => this.restore(s, state[i]));
    this.changed();
  }

  // 方向、視角、順序與剪裁回到開啟時的樣子；是否使用維持不變
  resetSelected() {
    if (this.selected < 0) return;
    const s = this.shots[this.selected], use = s.use;
    this.remember();
    this.restore(s, this.original[this.selected]);
    s.use = use;
    this.changed();
  }

  // 把選取的照片繞 axis 旋轉；from 是旋轉前的 [前, 右]
  turnSelected(axis, angle, from) {
    const s = this.shots[this.selected];
    const [fwd, right] = from || [s.basis.fwd, s.basis.right];
    setBasis(s, rotate(fwd, axis, angle), rotate(right, axis, angle));
  }

  // 依畫面方向微調：dx 往右、dy 往上（度）
  nudge(dx, dy) {
    if (this.selected < 0) return;
    this.remember();
    const v = this.viewBasis();
    if (dx) this.turnSelected(norm(cross(v.fwd, v.right)), dx * RAD);
    if (dy) this.turnSelected(norm(cross(v.fwd, v.up)), dy * RAD);
    this.changed();
  }

  // 順時針為正（度）
  roll(angle) {
    if (this.selected < 0) return;
    this.remember();
    this.turnSelected(this.shots[this.selected].basis.fwd, angle * RAD);
    this.changed();
  }

  // 把選取照片的視角改變 delta 度；勾選「套用到所有照片」時，其他照片以相同比例改變
  widen(delta) {
    if (this.selected < 0) return;
    const h = this.shots[this.selected].hfov;
    const target = clamp(h + delta, 1, 170);
    this.remember();
    this.zoom(Math.tan(target * RAD / 2) / Math.tan(h * RAD / 2));
    this.changed();
  }

  // 直接設定選取照片的某個值
  setValue(field, value) {
    if (this.selected < 0 || !Number.isFinite(value)) return;
    const s = this.shots[this.selected];
    this.remember();
    if (field === "hfov") {
      value = clamp(value, 1, 170);
      for (const x of this.allFov ? this.shots : [s]) x.hfov = value;
    } else if (field === "priority") {
      s.priority = value;
    } else {
      s[field] = field === "elevation" ? clamp(value, -90, 90) : value;
      s.basis = basis(s.azimuth, s.elevation, s.roll);
    }
    this.changed();
  }

  // 移到最前、移到最後，或回到自動（依鏡頭中心的遠近）
  arrange(where) {
    if (this.selected < 0) return;
    const others = this.shots.filter((_, i) => i !== this.selected).map((s) => s.priority || 0);
    const value = where === "front" ? Math.max(0, ...others) + 1 : where === "back" ? Math.min(0, ...others) - 1 : 0;
    this.setValue("priority", value);
  }

  // 放大或縮小照片的視角；factor 乘在 tan(半視角) 上
  zoom(factor, from) {
    if (this.selected < 0) return;
    const targets = this.allFov ? this.shots.map((_, i) => i) : [this.selected];
    for (const i of targets) setTanHalf(this.shots[i], (from ? from[i] : tanHalf(this.shots[i])[0]) * factor);
  }

  // 加入或移出拼貼；剛貼上的照片放在最上層
  setUse(i, use) {
    const s = this.shots[i];
    this.remember();
    s.use = use;
    if (use) s.priority = Math.max(0, ...this.shots.filter((x, k) => k !== i && x.use !== false).map((x) => x.priority || 0)) + 1;
    this.changed();
  }

  // 全部加入或全部移出（從空白開始）
  setAll(use) {
    this.remember();
    for (const s of this.shots) {
      s.use = use;
      s.priority = 0;
    }
    this.changed();
  }

  // 設定剪裁的一邊：edge 是 0 左、1 上、2 右、3 下，value 是 0–1 的比例
  setCrop(edge, value, remember = true) {
    if (this.selected < 0 || !Number.isFinite(value)) return;
    const c = this.shots[this.selected].crop;
    if (remember) this.remember();
    if (edge === 0) c[0] = clamp(value, 0, c[2] - MIN_CROP);
    if (edge === 1) c[1] = clamp(value, 0, c[3] - MIN_CROP);
    if (edge === 2) c[2] = clamp(value, c[0] + MIN_CROP, 1);
    if (edge === 3) c[3] = clamp(value, c[1] + MIN_CROP, 1);
    this.changed();
  }

  resetCrop() {
    if (this.selected < 0) return;
    this.remember();
    this.shots[this.selected].crop = [0, 0, 1, 1];
    this.changed();
  }

  // ---- 檢視 ---------------------------------------------------------------

  lookAt(d) {
    this.view.azimuth = ((Math.atan2(-d[1], d[0]) / RAD) % 360 + 360) % 360;
    this.view.elevation = Math.asin(clamp(d[2], -1, 1)) / RAD;
  }

  goToVertex(step) {
    const n = this.order.length;
    this.vertex = this.vertex < 0 ? (step > 0 ? 0 : n - 1) : (this.vertex + step + n) % n;
    this.lookAt(this.vertices[this.order[this.vertex]]);
    this.draw();
    this.onchange();
  }

  setFov(fov) {
    this.view.fov = clamp(fov, 20, 120);
    this.draw();
  }

  // ---- 指標操作 -------------------------------------------------------------

  attach() {
    const el = this.overlay;
    const point = (e) => {
      const r = el.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    el.addEventListener("pointerdown", (e) => {
      try { el.setPointerCapture(e.pointerId); } catch (_) { /* 合成的事件沒有真正的指標 */ }
      this.pointers.set(e.pointerId, point(e));
      this.beginGesture();
    });
    el.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.set(e.pointerId, point(e));
      this.moveGesture();
    });
    const end = (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      const g = this.gesture;
      if (g && g.kind !== "pinch" && this.pointers.size === 1 && !g.moved) {
        if (g.kind === "photo" || g.kind === "crop") this.history.pop();  // 只是點一下，沒有移動
        this.select(this.pick(this.directionAt(...g.start)));
      }
      this.pointers.delete(e.pointerId);
      this.beginGesture();
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.001);
      if (this.mode === "adjust" && this.selected >= 0) {
        this.remember();
        this.zoom(factor);
        this.changed();
      } else {
        this.setFov(this.view.fov / factor);
      }
    }, { passive: false });
  }

  // 剪裁時要拖曳的邊：離按下的位置最近的那一邊（以角度計）
  nearestCropEdge(uv) {
    const s = this.shots[this.selected], c = s.crop, t = tanHalf(s);
    const distances = [Math.abs(uv[0] - c[0]) * t[0], Math.abs(uv[1] - c[1]) * t[1], Math.abs(uv[0] - c[2]) * t[0], Math.abs(uv[1] - c[3]) * t[1]];
    return distances.indexOf(Math.min(...distances));
  }

  beginGesture() {
    const pts = [...this.pointers.values()];
    const s = this.shots[this.selected];
    let kind = "view";
    if (s && this.mode === "adjust") kind = "photo";
    if (s && this.mode === "crop") kind = "crop";
    if (pts.length === 1) {
      const d0 = this.directionAt(...pts[0]);
      if (kind !== "view") this.remember();
      const uv = kind === "crop" ? photoUv(s, d0) : null;
      if (kind === "crop" && !uv) kind = "view";
      this.gesture = {
        kind,
        start: pts[0],
        moved: false,
        view: { ...this.view },
        from: s ? [s.basis.fwd, s.basis.right] : null,
        d0,
        edge: uv ? this.nearestCropEdge(uv) : -1,
      };
    } else if (pts.length === 2) {
      const [a, b] = pts, mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const adjust = kind === "photo";
      if (adjust && !(this.gesture && this.gesture.kind === "photo")) this.remember();
      this.gesture = {
        kind: "pinch",
        adjust,
        distance: Math.hypot(b[0] - a[0], b[1] - a[1]),
        angle: Math.atan2(b[1] - a[1], b[0] - a[0]),
        axis: this.directionAt(...mid),
        fov: this.view.fov,
        tans: this.shots.map((x) => tanHalf(x)[0]),
        from: s ? [s.basis.fwd, s.basis.right] : null,
      };
    } else {
      this.gesture = null;
    }
  }

  moveGesture() {
    const g = this.gesture, pts = [...this.pointers.values()];
    if (!g) return;
    if (g.kind === "pinch" && pts.length === 2) {
      const [a, b] = pts, ratio = Math.hypot(b[0] - a[0], b[1] - a[1]) / g.distance;
      if (g.adjust) {
        // 照片跟著手指：張開放大、轉動則傾斜
        this.zoom(ratio, g.tans);
        this.turnSelected(g.axis, Math.atan2(b[1] - a[1], b[0] - a[0]) - g.angle, g.from);
        this.changed();
      } else {
        this.setFov(g.fov / ratio);
      }
      return;
    }
    if (pts.length !== 1) return;
    const p = pts[0];
    if (Math.hypot(p[0] - g.start[0], p[1] - g.start[1]) > 4) g.moved = true;
    if (!g.moved) return;
    if (g.kind === "photo") {
      // 讓按下時指著的那一點一直留在手指下方
      const d1 = this.directionAt(...p), axis = cross(g.d0, d1), n = Math.hypot(...axis);
      if (n < 1e-9) return;
      this.turnSelected(axis.map((x) => x / n), Math.atan2(n, dot(g.d0, d1)), g.from);
      this.changed();
    } else if (g.kind === "crop") {
      const uv = photoUv(this.shots[this.selected], this.directionAt(...p));
      if (uv) this.setCrop(g.edge, uv[g.edge % 2], false);
    } else {
      const k = this.view.fov / this.canvas.clientHeight;
      this.view.azimuth = ((g.view.azimuth - (p[0] - g.start[0]) * k) % 360 + 360) % 360;
      this.view.elevation = clamp(g.view.elevation + (p[1] - g.start[1]) * k, -90, 90);
      this.draw();
    }
  }

  select(i) {
    this.selected = i;
    this.draw();
    this.onchange();
  }
}
