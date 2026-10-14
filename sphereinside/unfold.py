"""多面體 → 展開圖。

展開圖是從多面體外側看到的樣子：每個面在平面上仍是逆時針，
座標以公釐為單位、y 軸朝上，原點在紙張左下角。

面數多時整個多面體無法攤成一片，所以會拆成數片，每一片都放得進一頁，
再把各片排進頁面。
"""

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .messages import t as _t

LABEL_MODES = ("inside", "outside", "none")  # 邊的編號：放在面上、放在圖片外、不標示
_MIN_LABEL = 1.0  # 編號最小的字高（公釐）
_TAB_SLOPE = 1.0  # 黏貼耳兩端內縮量與高度的比（1 即 45°）
# 黏貼耳佔邊的哪一段；整段放不下時改放在其中一端
_TAB_SPANS = ((0, 1), (0.3, 1), (0, 0.7), (0.55, 1), (0, 0.45))


@dataclass
class Label:
    pos: np.ndarray  # 文字中心
    along: np.ndarray  # 文字方向（沿著邊）
    text: str
    size: float  # 字高（公釐）
    on_face: bool = True  # 是否壓在圖片上


@dataclass
class PageLayout:
    faces: list = field(default_factory=list)  # [(面編號, (3, 2) 座標)]
    folds: list = field(default_factory=list)  # [(2, 2) 線段]
    cuts: list = field(default_factory=list)  # [(2, 2) 線段]
    labels: list = field(default_factory=list)


@dataclass
class _Piece:
    tris: dict  # 面編號 → (3, 2)
    tabs: list = field(default_factory=list)  # [(邊編號, (4, 2))]
    folds: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    boxes: list = field(default_factory=list)  # 放在圖片外的編號所佔的範圍


def _edge_key(i, j):
    return (i, j) if i < j else (j, i)


def _edge_faces(S):
    """邊 → [(面, 該邊在面中的位置)]。"""
    edges = {}
    for f, tri in enumerate(S):
        for k in range(3):
            edges.setdefault(_edge_key(tri[k], tri[(k + 1) % 3]), []).append((f, k))
    return edges


def _overlaps(tri, others, eps=1e-6):
    """tri 是否與 others 中任何一個三角形有面積上的重疊（共用邊或頂點不算）。"""
    if len(others) == 0:
        return False
    others = np.asarray(others)

    def axes(t):
        e = np.roll(t, -1, axis=-2) - t
        n = np.stack([-e[..., 1], e[..., 0]], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    A = np.concatenate([np.broadcast_to(axes(tri), (len(others), 3, 2)), axes(others)], 1)
    p = np.einsum("mad,kd->mak", A, tri)
    q = np.einsum("mad,mkd->mak", A, others)
    separated = (p.max(2) <= q.min(2) + eps) | (q.max(2) <= p.min(2) + eps)
    return bool((~separated.any(1)).any())


def _first_triangle(P):
    a, b, c = P
    d = np.linalg.norm(b - a)
    x = (c - a) @ (b - a) / d
    y = np.sqrt(max((c - a) @ (c - a) - x * x, 0))
    return np.array([[0, 0], [d, 0], [x, y]])


def _across(V, tri_g, i, j, pi, pj):
    """把面 tri_g 沿著邊 (i, j) 攤平；pi、pj 是該邊已知的平面座標，原本的面在 i→j 的左側。"""
    k = next(v for v in tri_g if v != i and v != j)
    d = np.linalg.norm(V[j] - V[i])
    ex = (pj - pi) / np.linalg.norm(pj - pi)
    ey = np.array([-ex[1], ex[0]])
    x = (V[k] - V[i]) @ (V[j] - V[i]) / d
    y = np.sqrt(max((V[k] - V[i]) @ (V[k] - V[i]) - x * x, 0))
    pos = {i: pi, j: pj, k: pi + x * ex - y * ey}
    return np.array([pos[v] for v in tri_g])


def _fits(lo, hi, w, h):
    dx, dy = hi - lo
    return (dx <= w and dy <= h) or (dx <= h and dy <= w)


def _grow(V, S, edges, w, h):
    """把面貪婪地長成一片片不重疊、放得進 w × h 的展開圖。"""
    neighbours = {f: [] for f in range(len(S))}
    for pair in edges.values():
        (f, _), (g, _) = pair
        neighbours[f].append(g)
        neighbours[g].append(f)

    unplaced = set(range(len(S)))
    pieces, joined = [], set()
    while unplaced:
        # 從剩餘鄰居最少的面開始，零碎的面才有機會併進較大的一片
        seed = min(unplaced, key=lambda f: (sum(g in unplaced for g in neighbours[f]), f))
        tri = _first_triangle(V[S[seed]])
        if not _fits(tri.min(0), tri.max(0), w, h):
            raise ValueError(_t("face_too_large"))
        unplaced.discard(seed)
        tris = {seed: tri}
        lo, hi = tri.min(0), tri.max(0)
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for k in range(3):
                i, j = S[f][k], S[f][(k + 1) % 3]
                g = next(x for x, _ in edges[_edge_key(i, j)] if x != f)
                if g not in unplaced:
                    continue
                tri = _across(V, S[g], i, j, tris[f][k], tris[f][(k + 1) % 3])
                nlo, nhi = np.minimum(lo, tri.min(0)), np.maximum(hi, tri.max(0))
                if not _fits(nlo, nhi, w, h) or _overlaps(tri, list(tris.values())):
                    continue
                tris[g] = tri
                lo, hi = nlo, nhi
                unplaced.discard(g)
                joined.add(_edge_key(i, j))
                queue.append(g)
        pieces.append(_Piece(tris))
    return pieces, joined


def _tab_quad(a, b, height):
    """邊 a→b 外側（右側）的梯形黏貼耳。"""
    length = np.linalg.norm(b - a)
    u = (b - a) / length
    n = np.array([u[1], -u[0]])
    height = min(height, length / (2 * _TAB_SLOPE + 0.4))
    inset = height * _TAB_SLOPE
    return np.array([a, b, b - u * inset + n * height, a + u * inset + n * height])


def _label_box(pos, u, n, text, size):
    """編號佔用的範圍（兩個三角形），用來檢查是否與其他東西重疊。"""
    half_w = (0.6 * size * len(text) + 0.6) / 2
    half_h = size * 0.55 + 0.3
    c = [pos + u * x * half_w + n * y * half_h for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return [np.array([c[0], c[1], c[2]]), np.array([c[0], c[2], c[3]])]


def _decorate(S, edges, pieces, joined, tab, labels):
    """加上摺線、裁切線、黏貼耳與邊的編號。

    回傳 (黏貼耳數, 剪開的邊數, 因為圖片外放不下而省略的編號數)。
    """
    owner = {f: p for p in pieces for f in p.tris}
    number = 0
    pending = []  # 黏貼耳全部決定之後才放編號，編號才不會被之後的黏貼耳蓋住
    for key in sorted(edges):
        (f, kf), (g, kg) = edges[key]
        if key in joined:
            p = owner[f]
            p.folds.append(p.tris[f][[kf, (kf + 1) % 3]])
            continue
        number += 1
        sides = [(f, kf), (g, kg)]
        ends = [owner[x].tris[x][[k, (k + 1) % 3]] for x, k in sides]
        length = np.linalg.norm(ends[0][1] - ends[0][0])
        size = float(np.clip(length * 0.11, 1.2, 2.6))

        # 黏貼耳放在不會與同一片其他部分重疊的那一側；放不下就縮短或壓低，都不行才不加
        tab_side = None
        for height, (t0, t1), s in itertools.product((tab, tab * 0.6, tab * 0.35), _TAB_SPANS, (0, 1)):
            p = owner[sides[s][0]]
            a, b = ends[s]
            quad = _tab_quad(a + t0 * (b - a), a + t1 * (b - a), height)
            obstacles = list(p.tris.values()) + [t for _, q in p.tabs for t in (q[[0, 1, 2]], q[[0, 2, 3]])]
            if not _overlaps(quad[[0, 1, 2]], obstacles) and not _overlaps(quad[[0, 2, 3]], obstacles):
                p.tabs.append((number, quad))
                tab_side = s
                break

        for s, (x, _) in enumerate(sides):
            p = owner[x]
            a, b = ends[s]
            u = (b - a) / length
            n = np.array([u[1], -u[0]])  # 朝面的外側
            if s == tab_side:
                quad = p.tabs[-1][1]
                p.folds.append(np.array([a, b]))
                p.cuts += [quad[[0, 3]], quad[[3, 2]], quad[[2, 1]]]
                p.cuts += [seg for seg in (np.array([a, quad[0]]), np.array([quad[1], b])) if np.linalg.norm(seg[1] - seg[0]) > 1e-9]
                # 黏貼耳上的編號本來就在圖片外；耳太小、字放不下時，改用另一側的放法
                depth = (quad[3] - quad[0]) @ n
                text = str(number)
                fit = min(size, (depth - 0.6) / 1.1, (np.linalg.norm(quad[2] - quad[3]) - 0.6) / (0.6 * len(text)))
                if fit >= _MIN_LABEL:
                    pending.append((p, Label((quad[0] + quad[1]) / 2 + n * depth / 2, u, text, fit, False), None))
                else:
                    pending.append((p, Label((a + b) / 2 - n * size * 0.9, u, text, size), (a, b, u, n)))
            else:
                p.cuts.append(np.array([a, b]))
                pending.append((p, Label((a + b) / 2 - n * size * 0.9, u, str(number), size), (a, b, u, n)))

    omitted = 0
    if labels == "none":
        return sum(len(p.tabs) for p in pieces), number, omitted
    for p, label, edge in pending:
        if labels == "outside" and edge is not None:
            a, b, u, n = edge
            obstacles = list(p.tris.values()) + [t for _, q in p.tabs for t in (q[[0, 1, 2]], q[[0, 2, 3]])] + p.boxes
            for k, t, gap in itertools.product((1.0, 0.75, 0.55), (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9), (0.5, 0.15)):
                size = max(_MIN_LABEL, label.size * k)
                pos = a + (b - a) * t + n * (gap + size * 0.55 + 0.3)
                box = _label_box(pos, u, n, label.text, size)
                if not any(_overlaps(tri, obstacles) for tri in box):
                    label = Label(pos, u, label.text, size, False)
                    p.boxes += box
                    break
            else:
                # 不壓到圖片：放不下就省略。這多半發生在同一片的狹窄缺口裡，對面就是成對的邊
                omitted += 1
                continue
        p.labels.append(label)
    return sum(len(p.tabs) for p in pieces), number, omitted


def _bounds(piece):
    pts = np.vstack(list(piece.tris.values()) + [q for _, q in piece.tabs] + piece.boxes)
    return pts.min(0), pts.max(0)


def _pack(pieces, w, h, margin, gap):
    """把各片排進頁面（依高度排成一列列）。"""
    items = []
    for p in pieces:
        lo, hi = _bounds(p)
        size = hi - lo
        rotate = not (size[0] <= w and size[1] <= h) or (size[1] > size[0] and size[1] <= w and size[0] <= h)
        if rotate:
            size = size[::-1]
        if size[0] > w + 1e-6 or size[1] > h + 1e-6:
            raise ValueError(_t("piece_too_large"))
        items.append((p, lo, hi, rotate, size))
    items.sort(key=lambda it: -it[4][1])

    pages = []  # 每頁是一串 [y, 高度, 已用寬度] 的列
    placed = []
    for p, lo, hi, rotate, size in items:
        spot = None
        for n, shelves in enumerate(pages):
            for shelf in shelves:
                if size[1] <= shelf[1] + 1e-6 and shelf[2] + size[0] <= w + 1e-6:
                    spot = (n, shelf[2], shelf[0])
                    shelf[2] += size[0] + gap
                    break
            if spot:
                break
            top = shelves[-1][0] + shelves[-1][1] + gap
            if top + size[1] <= h + 1e-6:
                shelves.append([top, size[1], size[0] + gap])
                spot = (n, 0.0, top)
                break
        if spot is None:
            pages.append([[0.0, size[1], size[0] + gap]])
            spot = (len(pages) - 1, 0.0, 0.0)
        placed.append((p, lo, hi, rotate, spot))

    layouts = [PageLayout() for _ in pages]
    for p, lo, hi, rotate, (n, x, y) in placed:
        if rotate:  # 逆時針轉 90°：(x, y) → (−y, x)
            R = np.array([[0.0, -1.0], [1.0, 0.0]])
            origin = np.array([-hi[1], lo[0]])
        else:
            R = np.eye(2)
            origin = lo
        shift = np.array([x, y]) + margin - origin

        def T(pts):
            return np.asarray(pts) @ R.T + shift

        page = layouts[n]
        page.faces += [(f, T(t)) for f, t in sorted(p.tris.items())]
        page.folds += [T(s) for s in p.folds]
        page.cuts += [T(s) for s in p.cuts]
        page.labels += [Label(T(l.pos), l.along @ R.T, l.text, l.size, l.on_face) for l in p.labels]
    return layouts


def unfold(V, S, page_width, page_height, margin=8.0, tab=5.0, gap=3.0, labels="inside"):
    """把多面體展開並排進頁面。V 的單位是公釐。

    labels 是邊的編號放在哪裡：LABEL_MODES 其中之一。
    回傳每一頁的 PageLayout，以及 (黏貼耳數, 剪開的邊數, 省略的編號數)；
    前兩者不相等時，表示有些邊因為空間不夠而沒有黏貼耳。
    """
    if labels not in LABEL_MODES:
        raise ValueError(labels)
    S = np.asarray(S)
    edges = _edge_faces(S)
    w, h = page_width - 2 * margin, page_height - 2 * margin
    # 生長時預留黏貼耳的空間
    pieces, joined = _grow(V, S, edges, w - 2 * tab, h - 2 * tab)
    counts = _decorate(S, edges, pieces, joined, tab, labels)
    return _pack(pieces, w, h, margin, gap), counts
