"""照片與拍攝方向 → 方向對顏色的對應。

世界座標：+x 朝方位 0°，+z 朝正上方。方位（azimuth）沿用羅盤的習慣，
從上方看為順時針增加，所以方位角 θ = −azimuth。

影像座標：u 向右、v 向下，像素中心在整數座標上。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_EXIF_IFD = 0x8769
_FOCAL_35MM = 0xA405
_DIAGONAL_35MM = 43.267
_PRIORITY_WEIGHT = 10.0


def basis(azimuth, elevation, roll=0.0):
    """鏡頭的前、右、上三個單位向量（世界座標）。角度以度為單位。"""
    th, e, ro = np.radians([-azimuth, elevation, roll])
    fwd = np.array([np.cos(e) * np.cos(th), np.cos(e) * np.sin(th), np.sin(e)])
    right0 = np.array([np.sin(th), -np.cos(th), 0.0])
    up0 = np.cross(right0, fwd)
    right = np.cos(ro) * right0 + np.sin(ro) * up0
    up = -np.sin(ro) * right0 + np.cos(ro) * up0
    return fwd, right, up


@dataclass
class Photo:
    image: np.ndarray  # (H, W, 3) uint8
    azimuth: float
    elevation: float
    hfov: float  # 沿影像寬度方向的視角
    roll: float = 0.0
    name: str = ""
    priority: float = 0.0  # 順序：涵蓋同一個方向時，數字大的優先
    fwd: np.ndarray = field(init=False, repr=False)
    right: np.ndarray = field(init=False, repr=False)
    up: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.fwd, self.right, self.up = basis(self.azimuth, self.elevation, self.roll)

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def focal(self):
        """以像素為單位的焦距。"""
        return (self.width / 2) / np.tan(np.radians(self.hfov) / 2)

    def pixel_to_direction(self, u, v):
        """像素 → 世界座標中的單位方向。"""
        u, v = np.asarray(u, float), np.asarray(v, float)
        x = (u - (self.width - 1) / 2)[..., None]
        y = (v - (self.height - 1) / 2)[..., None]
        d = x * self.right - y * self.up + self.focal * self.fwd
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def direction_to_pixel(self, d):
        """單位方向 → (u, v, c)。c 是方向與鏡頭中心夾角的餘弦，c ≤ 0 表示在鏡頭後方。"""
        c = d @ self.fwd
        z = np.where(c > 1e-9, c, np.nan)
        u = (self.width - 1) / 2 + self.focal * (d @ self.right) / z
        v = (self.height - 1) / 2 - self.focal * (d @ self.up) / z
        return u, v, c


class PhotoSet:
    """一組照片合起來的「方向 → 顏色」。"""

    def __init__(self, photos):
        if not photos:
            raise ValueError("沒有照片")
        self.photos = list(photos)

    def _assign(self, dirs):
        """每個方向取順序最大的那一張，順序相同時取鏡頭中心最接近的。

        回傳照片編號（−1 表示沒有涵蓋）與像素位置。
        """
        n = len(dirs)
        best = np.full(n, -1)
        score = np.full(n, -np.inf)
        bu, bv = np.zeros(n), np.zeros(n)
        for i, p in enumerate(self.photos):
            with np.errstate(invalid="ignore"):
                u, v, c = p.direction_to_pixel(dirs)
                inside = (c > 1e-9) & (u >= -0.5) & (u <= p.width - 0.5) & (v >= -0.5) & (v <= p.height - 0.5)
            # c 介於 0 與 1 之間，順序只要差 0.1 以上就優先
            s = p.priority * _PRIORITY_WEIGHT + c
            take = inside & (s > score)
            best[take], score[take] = i, s[take]
            bu[take], bv[take] = u[take], v[take]
        return best, bu, bv

    def covered(self, dirs):
        return self._assign(np.asarray(dirs, float))[0] >= 0

    def sample(self, dirs):
        """方向 (N, 3) → 顏色 (N, 3) 與是否有照片涵蓋 (N,)。"""
        dirs = np.asarray(dirs, float)
        best, u, v = self._assign(dirs)
        colors = np.zeros((len(dirs), 3))
        for i, p in enumerate(self.photos):
            m = best == i
            if m.any():
                colors[m] = _bilinear(p.image, u[m], v[m])
        return colors, best >= 0


def _bilinear(img, u, v):
    h, w = img.shape[:2]
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    x0 = np.minimum(u.astype(int), w - 2)
    y0 = np.minimum(v.astype(int), h - 2)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _hfov_from_exif(img):
    """由 EXIF 的 35mm 等效焦距換算水平視角；讀不到時回傳 None。"""
    try:
        f35 = img.getexif().get_ifd(_EXIF_IFD).get(_FOCAL_35MM)
    except Exception:
        return None
    if not f35:
        return None
    w, h = img.size
    focal_px = float(f35) / _DIAGONAL_35MM * np.hypot(w, h)
    return float(np.degrees(2 * np.arctan(w / 2 / focal_px)))


def load_photos(meta_path, max_focal=None):
    """讀取對照檔與其中列出的照片。

    對照檔是 JSON：{"hfov": 預設視角, "photos": [{"file", "azimuth", "elevation",
    "roll", "hfov", "priority"}, ...]}，角度以度為單位，檔案路徑相對於對照檔所在的資料夾。
    max_focal 給定時，解析度超過需求的照片會先縮小，避免取樣時產生雜訊並節省記憶體。
    """
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    photos = []
    for entry in meta["photos"]:
        with Image.open(meta_path.parent / entry["file"]) as raw:
            hfov = entry.get("hfov", meta.get("hfov"))
            if hfov is None:
                hfov = _hfov_from_exif(raw)
            if hfov is None:
                raise ValueError(f"{entry['file']}：對照檔沒有 hfov，照片的 EXIF 也沒有等效焦距")
            img = ImageOps.exif_transpose(raw).convert("RGB")
        focal = (img.width / 2) / np.tan(np.radians(hfov) / 2)
        if max_focal and focal > max_focal:
            k = max_focal / focal
            img = img.resize((max(2, round(img.width * k)), max(2, round(img.height * k))), Image.BOX)
        photos.append(
            Photo(
                image=np.asarray(img),
                azimuth=float(entry["azimuth"]),
                elevation=float(entry["elevation"]),
                hfov=float(hfov),
                roll=float(entry.get("roll", 0.0)),
                priority=float(entry.get("priority", 0.0)),
                name=entry["file"],
            )
        )
    return PhotoSet(photos)
