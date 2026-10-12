"""多面體：正二十面體、細分、球座標轉換。

V 是 (N, 3) 的頂點陣列，全部落在單位球面上。
S 是 (M, 3) 的整數陣列，每列是一個三角形面的三個頂點編號，
從外側看為逆時針。
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull

G = (1 + 5 ** 0.5) / 2  # 黃金比例


def to_spherical(p):
    """直角座標 → (r, θ, φ)。θ 是與 +x 的方位角，φ 是與 +z 的夾角。"""
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    return r, np.arctan2(y, x), np.arccos(np.clip(z / r, -1, 1))


def from_spherical(r, theta, phi):
    """(r, θ, φ) → 直角座標。"""
    s = np.sin(phi)
    return np.stack([r * s * np.cos(theta), r * s * np.sin(theta), r * np.cos(phi)], -1)


def orient_faces(V, S):
    """把每個面的頂點順序統一成從外側看為逆時針。"""
    S = np.array(S, int)
    a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
    S[inward] = S[inward][:, [0, 2, 1]]
    return S


def icosahedron():
    """正二十面體。"""
    V = []
    for a, b in itertools.product((1, -1), (G, -G)):
        V += [(a, b, 0), (0, a, b), (b, 0, a)]
    V = np.array(V, float)
    # 這組座標的邊長恰為 2，面就是兩兩距離都等於 2 的三點組
    S = [
        t
        for t in itertools.combinations(range(len(V)), 3)
        if all(abs(np.linalg.norm(V[i] - V[j]) - 2) < 1e-9 for i, j in itertools.combinations(t, 2))
    ]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, orient_faces(V, S)


def subdivide(V, S):
    """細分一輪：每個面的中心外推到球面成為新頂點，再以凸包重建面。"""
    centers = V[S].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    V = np.vstack([V, centers])
    return V, orient_faces(V, ConvexHull(V).simplices)


def build(level=1):
    """細分 level 輪後的多面體，面數為 20 × 3**level。"""
    V, S = icosahedron()
    for _ in range(level):
        V, S = subdivide(V, S)
    return V, S
