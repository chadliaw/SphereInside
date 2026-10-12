"""產生一組合成的範例照片與對照檔，不必實際拍攝就能試跑整個流程。

    python examples/make_demo.py
    python -m sphereinside build examples/demo/meta.json -o examples/demo/demo.pdf --png

場景是一張畫了經緯線與方位標示的全景圖，方便檢查貼圖的位置與方向是否正確。
"""

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sphereinside.synthetic import equirect_scene, render_photo  # noqa: E402

OUT = Path(__file__).resolve().parent / "demo"
HFOV = 50.0
SIZE = (900, 1200)  # 直拍


def panorama(width=4096):
    height = width // 2
    az = np.linspace(0, 1, width, endpoint=False)
    el = np.linspace(1, 0, height)
    rgb = np.zeros((height, width, 3))
    for x in range(width):
        top = np.array(colorsys.hsv_to_rgb(az[x], 0.35, 1.0))
        bottom = np.array(colorsys.hsv_to_rgb(az[x], 0.75, 0.55))
        rgb[:, x] = bottom + (top - bottom) * el[:, None]
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for deg in range(0, 360, 15):
        x = deg / 360 * width
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=5 if deg % 90 == 0 else 2)
    for deg in range(-75, 90, 15):
        y = (90 - deg) / 180 * height
        draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=5 if deg == 0 else 2)

    big, small = ImageFont.load_default(size=width // 40), ImageFont.load_default(size=width // 80)
    names = {0: "N", 90: "E", 180: "S", 270: "W"}
    for deg in range(0, 360, 30):
        x = (deg + 7.5) / 360 * width
        for el_deg in (-60, -30, 0, 30, 60):
            y = (90 - el_deg - 7.5) / 180 * height
            if deg in names and el_deg == 0:
                draw.text((x, y), names[deg], font=big, fill=(20, 20, 20), anchor="mm")
            else:
                draw.text((x, y), f"{deg},{el_deg:+d}", font=small, fill=(20, 20, 20), anchor="mm")
    return img


def shots():
    """直拍、相鄰照片互相重疊的拍攝方向。"""
    plan = [(0, 12), (38, 10), (-38, 10), (68, 6), (-68, 6), (90, 1), (-90, 1)]
    rng = np.random.default_rng(0)
    for elevation, count in plan:
        for k in range(count):
            # 加一點手持的晃動
            yield (
                (360 * k / count + rng.uniform(-3, 3)) % 360,
                float(np.clip(elevation + rng.uniform(-2, 2), -90, 90)),
                rng.uniform(-4, 4),
            )


def main():
    (OUT / "photos").mkdir(parents=True, exist_ok=True)
    pano = panorama()
    pano.save(OUT / "panorama.jpg", quality=90)
    scene = equirect_scene(pano)
    entries = []
    for n, (azimuth, elevation, roll) in enumerate(shots(), 1):
        name = f"photos/{n:03d}.jpg"
        photo = render_photo(scene, azimuth, elevation, HFOV, *SIZE, roll=roll)
        Image.fromarray(photo.image).save(OUT / name, quality=92)
        entries.append({"file": name, "azimuth": round(azimuth, 2), "elevation": round(elevation, 2), "roll": round(roll, 2)})
    meta = {"hfov": HFOV, "photos": entries}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"已產生 {len(entries)} 張照片：{OUT}")


if __name__ == "__main__":
    main()
