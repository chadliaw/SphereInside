// 手動對齊：從球心往外看，拖曳照片來修正它的方位、仰角、傾斜與視角。
// 需要 index.html 裡的 basis、paramsFromBasis、dot、cross 與 RAD。
"use strict";

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// 每個像素就是一個方向；與 Python 端相同，取鏡頭中心最接近該方向的照片
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;
uniform mediump sampler2DArray photos;
uniform highp sampler2D params;  // 每張照片一列：前、右、上、(tan 半寬, tan 半高, 順序)
uniform int count;
uniform int selected;
uniform float overlayAlpha;
uniform vec3 viewF, viewR, viewU;
uniform vec2 viewTan;
uniform vec2 resolution;
uniform float layerHeight;
out vec4 outColor;
const float PRIORITY_WEIGHT = 10.0;  // 與 sphereinside/camera.py 相同

bool project(int i, vec3 d, out vec2 uv, out float c) {
  vec3 f = texelFetch(params, ivec2(0, i), 0).xyz;
  c = dot(d, f);
  if (c <= 1e-4) return false;
  vec3 r = texelFetch(params, ivec2(1, i), 0).xyz;
  vec3 u = texelFetch(params, ivec2(2, i), 0).xyz;
  vec2 t = texelFetch(params, ivec2(3, i), 0).xy;
  uv = vec2(0.5 + dot(d, r) / c / (2.0 * t.x), 0.5 - dot(d, u) / c / (2.0 * t.y));
  return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
}

vec3 photo(int i, vec2 uv) {
  float t = texelFetch(params, ivec2(3, i), 0).y;
  float lod = max(0.0, log2(layerHeight / (2.0 * t) / (resolution.y / (2.0 * viewTan.y))));
  return textureLod(photos, vec3(uv, float(i)), lod).rgb;
}

void main() {
  vec2 ndc = gl_FragCoord.xy / resolution * 2.0 - 1.0;
  vec3 d = normalize(viewF + ndc.x * viewTan.x * viewR + ndc.y * viewTan.y * viewU);
  int best = -1;
  float bestScore = -1e9;
  vec2 bestUv = vec2(0.0);
  for (int i = 0; i < count; i++) {
    if (i == selected && overlayAlpha > 0.0) continue;  // 半透明時，底下顯示其他照片
    vec2 uv;
    float c;
    if (!project(i, d, uv, c)) continue;
    float score = texelFetch(params, ivec2(3, i), 0).z * PRIORITY_WEIGHT + c;
    if (score > bestScore) { best = i; bestScore = score; bestUv = uv; }
  }
  vec3 color = best < 0 ? vec3(0.2, 0.22, 0.25) : photo(best, bestUv);
  if (selected >= 0 && overlayAlpha > 0.0) {
    vec2 uv;
    float c;
    if (project(selected, d, uv, c)) color = mix(color, photo(selected, uv), overlayAlpha);
  }
  outColor = vec4(color, 1.0);
}`;

const TEXTURE_BUDGET = 96e6;  // 所有照片在顯示卡上的大小上限（位元組）
const VIEW_FOV = { 0: 120, 1: 95, 2: 70 };  // 檢視頂點時的視角，夠看到周圍一圈的面
const PRIORITY_WEIGHT = 10;  // 順序的權重，與 sphereinside/camera.py 相同

const norm = (v) => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0] / n, v[1] / n, v[2] / n]; };
const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];

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

class Aligner {
  constructor(canvas, overlay, shots, mesh, level) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.shots = shots;
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
    if (!gl) throw new Error("這個瀏覽器不支援 WebGL 2，無法顯示對齊畫面。");
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
    for (let i = 0; i < n; i++) {
      progress(i, n);
      const bitmap = await createImageBitmap(this.shots[i].blob);
      ctx.drawImage(bitmap, 0, 0, w, h);
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
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 4, n);
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
    for (const name of ["photos", "params", "count", "selected", "overlayAlpha", "viewF", "viewR", "viewU", "viewTan", "resolution", "layerHeight"]) {
      this.uniforms[name] = gl.getUniformLocation(p, name);
    }
    gl.uniform1i(this.uniforms.photos, 0);
    gl.uniform1i(this.uniforms.params, 1);
    return p;
  }

  uploadParams() {
    const data = new Float32Array(this.shots.length * 16);
    this.shots.forEach((s, i) => {
      const b = s.basis, t = tanHalf(s);
      data.set([...b.fwd, 0, ...b.right, 0, ...b.up, 0, t[0], t[1], s.priority || 0, 0], i * 16);
    });
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4, this.shots.length, gl.RGBA, gl.FLOAT, data);
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
      const b = s.basis, c = dot(d, b.fwd), t = tanHalf(s);
      if (c <= 1e-4) return;
      const score = (s.priority || 0) * PRIORITY_WEIGHT + c;
      if (score > bestScore && Math.abs(dot(d, b.right) / c) <= t[0] && Math.abs(dot(d, b.up) / c) <= t[1]) {
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
    });
  }

  render() {
    const gl = this.gl, u = this.uniforms, v = this.viewBasis();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform1i(u.count, this.shots.length);
    gl.uniform1i(u.selected, this.selected);
    gl.uniform1f(u.overlayAlpha, this.translucent && this.selected >= 0 ? 0.5 : 0);
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
    if (this.showMesh) {
      g.beginPath();
      for (const [a, b] of this.edges) segment(this.vertices[a], this.vertices[b]);
      // 深色襯底，白牆上也看得清楚
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
    if (this.selected >= 0) {
      const s = this.shots[this.selected], b = s.basis, t = tanHalf(s);
      const corners = [[-1, 1], [1, 1], [1, -1], [-1, -1]].map(([x, y]) => add(add(b.fwd, b.right, x * t[0]), b.up, y * t[1]));
      g.beginPath();
      for (let k = 0; k < 4; k++) segment(corners[k], corners[(k + 1) % 4]);
      g.strokeStyle = "#f2c94c";
      g.lineWidth = 2.5 * scale;
      g.stroke();
    }
  }

  // ---- 修改照片 -------------------------------------------------------------

  snapshotOf(s) {
    return [s.azimuth, s.elevation, s.roll, s.hfov, s.priority || 0];
  }

  restore(s, [azimuth, elevation, roll, hfov, priority]) {
    Object.assign(s, { azimuth, elevation, roll, hfov, priority, basis: basis(azimuth, elevation, roll) });
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

  resetSelected() {
    if (this.selected < 0) return;
    this.remember();
    this.restore(this.shots[this.selected], this.original[this.selected]);
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
    const target = Math.max(1, Math.min(170, h + delta));
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
      value = Math.max(1, Math.min(170, value));
      for (const x of this.allFov ? this.shots : [s]) x.hfov = value;
    } else if (field === "priority") {
      s.priority = value;
    } else {
      s[field] = field === "elevation" ? Math.max(-90, Math.min(90, value)) : value;
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

  // ---- 檢視 ---------------------------------------------------------------

  lookAt(d) {
    this.view.azimuth = ((Math.atan2(-d[1], d[0]) / RAD) % 360 + 360) % 360;
    this.view.elevation = Math.asin(Math.max(-1, Math.min(1, d[2]))) / RAD;
  }

  goToVertex(step) {
    const n = this.order.length;
    this.vertex = this.vertex < 0 ? (step > 0 ? 0 : n - 1) : (this.vertex + step + n) % n;
    this.lookAt(this.vertices[this.order[this.vertex]]);
    this.draw();
    this.onchange();
  }

  setFov(fov) {
    this.view.fov = Math.max(20, Math.min(120, fov));
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
        if (g.kind === "photo") this.history.pop();  // 只是點一下，沒有移動
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

  beginGesture() {
    const pts = [...this.pointers.values()];
    const adjust = this.mode === "adjust" && this.selected >= 0;
    if (pts.length === 1) {
      const s = adjust ? this.shots[this.selected] : null;
      if (adjust) this.remember();
      this.gesture = {
        kind: adjust ? "photo" : "view",
        start: pts[0],
        moved: false,
        view: { ...this.view },
        from: s ? [s.basis.fwd, s.basis.right] : null,
        d0: this.directionAt(...pts[0]),
      };
    } else if (pts.length === 2) {
      const [a, b] = pts, mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const s = adjust ? this.shots[this.selected] : null;
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
    } else {
      const k = this.view.fov / this.canvas.clientHeight;
      this.view.azimuth = ((g.view.azimuth - (p[0] - g.start[0]) * k) % 360 + 360) % 360;
      this.view.elevation = Math.max(-90, Math.min(90, g.view.elevation + (p[1] - g.start[1]) * k));
      this.draw();
    }
  }

  select(i) {
    this.selected = i;
    this.draw();
    this.onchange();
  }
}
