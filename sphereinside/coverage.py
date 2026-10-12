"""檢查照片是否涵蓋所有方向。"""

from dataclasses import dataclass

import numpy as np

_BIN = 30  # 回報缺漏位置時的格子大小（度）
_MAX_LINES = 12


def fibonacci_sphere(n):
    """球面上近似均勻分布的 n 個單位方向。"""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    t = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], 1)


@dataclass
class CoverageReport:
    fraction: float  # 已涵蓋的比例，0–1
    missing: list  # [(方位起點, 仰角起點, 該格缺漏的比例), ...]

    @property
    def complete(self):
        return not self.missing

    def describe(self):
        if self.complete:
            return "照片已涵蓋所有方向。"
        lines = [f"照片只涵蓋 {self.fraction:.1%} 的方向，以下範圍還沒拍到："]
        for az, el, frac in self.missing[:_MAX_LINES]:
            lines.append(f"  方位 {az:3d}°–{az + _BIN:3d}°、仰角 {el:+3d}°–{el + _BIN:+3d}°（缺 {frac:.0%}）")
        if len(self.missing) > _MAX_LINES:
            lines.append(f"  …以及另外 {len(self.missing) - _MAX_LINES} 個範圍")
        return "\n".join(lines)


def check(photos, n=20000):
    """以 n 個均勻分布的方向檢查涵蓋情形。"""
    dirs = fibonacci_sphere(n)
    ok = photos.covered(dirs)
    azimuth = np.degrees(-np.arctan2(dirs[:, 1], dirs[:, 0])) % 360
    elevation = np.degrees(np.arcsin(dirs[:, 2]))
    a = np.minimum((azimuth // _BIN).astype(int), 360 // _BIN - 1)
    e = np.minimum(((elevation + 90) // _BIN).astype(int), 180 // _BIN - 1)
    missing = []
    for ei in range(180 // _BIN - 1, -1, -1):
        for ai in range(360 // _BIN):
            cell = (a == ai) & (e == ei)
            if cell.any() and not ok[cell].all():
                missing.append((ai * _BIN, ei * _BIN - 90, float(1 - ok[cell].mean())))
    return CoverageReport(float(ok.mean()), missing)
