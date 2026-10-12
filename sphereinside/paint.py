"""對每個面取色（含重疊處理與鏡射）。"""

import numpy as np

_CHUNK = 500_000
_BLEED = 1.0  # 面的邊緣往外多塗的像素數，避免相鄰的面之間露出白線


def face_pixels(tri2d, tri3d, scale, height_px, width_px):
    """一個面涵蓋的像素與各像素對應的空間點。

    tri2d 是頁面上的座標（公釐，y 朝上），scale 是每公釐的像素數。
    回傳像素的一維索引與對應的 (N, 3) 空間點。
    """
    px = tri2d * scale
    x0, y0 = np.floor(px.min(0) - _BLEED).astype(int)
    x1, y1 = np.ceil(px.max(0) + _BLEED).astype(int)
    xs = np.arange(max(x0, 0), min(x1, width_px - 1) + 1)
    ys = np.arange(max(y0, 0), min(y1, height_px - 1) + 1)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(0, int), np.zeros((0, 3))
    gx, gy = np.meshgrid(xs, ys)
    p = np.stack([gx.ravel() + 0.5, gy.ravel() + 0.5], 1)

    a, b, c = px
    M = np.linalg.inv(np.array([b - a, c - a]).T)
    st = (p - a) @ M.T
    bary = np.column_stack([1 - st.sum(1), st])
    # 重心座標乘上對應的高，就是到各邊的距離（像素）
    area2 = abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
    heights = area2 / np.linalg.norm(np.array([c - b, a - c, b - a]), axis=1)
    inside = (bary * heights >= -_BLEED).all(1)

    rows = height_px - 1 - gy.ravel()[inside]  # 影像的第 0 列在頁面最上方
    return rows * width_px + gx.ravel()[inside], bary[inside] @ tri3d


def render_page(layout, V, S, photos, page_width, page_height, dpi=300, mirror=True, gap_color=(128, 128, 128)):
    """把一頁展開圖上的每個面塗上顏色。

    回傳 (H, W, 3) 的影像，以及沒有照片涵蓋的像素數。
    mirror 為真時取色方向對 xz 平面鏡射，印在外側後從外面看才不會左右相反。
    """
    scale = dpi / 25.4
    width_px, height_px = round(page_width * scale), round(page_height * scale)
    image = np.full((height_px * width_px, 3), 255, np.uint8)

    index, points = [], []
    for f, tri2d in layout.faces:
        i, p = face_pixels(tri2d, V[S[f]], scale, height_px, width_px)
        index.append(i)
        points.append(p)
    if not index:
        return image.reshape(height_px, width_px, 3), 0
    index, points = np.concatenate(index), np.concatenate(points)
    # 相鄰的面在邊緣會多塗到同一個像素，只留一次
    index, first = np.unique(index, return_index=True)
    points = points[first]

    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    if mirror:
        dirs[:, 1] *= -1

    uncovered = 0
    for s in range(0, len(index), _CHUNK):
        colors, ok = photos.sample(dirs[s : s + _CHUNK])
        colors[~ok] = gap_color
        image[index[s : s + _CHUNK]] = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
        uncovered += int((~ok).sum())
    return image.reshape(height_px, width_px, 3), uncovered
