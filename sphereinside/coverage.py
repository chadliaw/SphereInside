"""檢查照片是否涵蓋所有方向。"""

from dataclasses import dataclass

import numpy as np

from .messages import t

_BIN = 30  # 回報缺漏位置時的格子大小（度）
_MAX_LINES = 12


def fibonacci_sphere(n):
    """球面上近似均勻分布的 n 個單位方向。"""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    t = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], 1)


def _percent(x, digits=0):
    """無條件捨去，避免把還有缺漏的情形顯示成 100%，或把缺了一點顯示成 0%。"""
    k = 10 ** digits
    floored = np.floor(x * 100 * k) / k
    if floored == 0 and x > 0:
        return t("less_than", x=f"{1 / k:g}")
    return f"{floored:.{digits}f}%"


@dataclass
class CoverageReport:
    fraction: float  # 已涵蓋的比例，0–1
    missing: list  # [(方位起點, 仰角起點, 該格缺漏的比例), ...]

    @property
    def complete(self):
        return not self.missing

    def describe(self):
        if self.complete:
            return t("coverage_complete")
        lines = [t("coverage_partial", percent=_percent(self.fraction, 2))]
        for az, el, frac in self.missing[:_MAX_LINES]:
            lines.append(t("coverage_cell", az0=az, az1=az + _BIN, el0=el, el1=el + _BIN, percent=_percent(frac)))
        if len(self.missing) > _MAX_LINES:
            lines.append(t("coverage_more", n=len(self.missing) - _MAX_LINES))
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
