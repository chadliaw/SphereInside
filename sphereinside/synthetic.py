"""由已知的場景算出照片，用於測試與範例。"""

import numpy as np

from .camera import Photo


def render_photo(scene, azimuth, elevation, hfov, width, height, roll=0.0, name=""):
    """scene 是「單位方向 (N, 3) → 顏色 (N, 3)」的函式；回傳從該方向拍到的 Photo。"""
    photo = Photo(np.zeros((height, width, 3), np.uint8), azimuth, elevation, hfov, roll, name)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dirs = photo.pixel_to_direction(u.ravel(), v.ravel())
    colors = np.clip(np.rint(scene(dirs)), 0, 255).astype(np.uint8)
    photo.image = colors.reshape(height, width, 3)
    return photo


def equirect_scene(image):
    """把等距長方全景圖（寬為方位 0°–360°，上緣為正上方）包成場景函式。"""
    image = np.asarray(image, float)
    h, w = image.shape[:2]

    def scene(dirs):
        azimuth = (-np.arctan2(dirs[:, 1], dirs[:, 0])) % (2 * np.pi)
        polar = np.arccos(np.clip(dirs[:, 2], -1, 1))
        x = np.minimum((azimuth / (2 * np.pi) * w).astype(int), w - 1)
        y = np.minimum((polar / np.pi * h).astype(int), h - 1)
        return image[y, x]

    return scene
