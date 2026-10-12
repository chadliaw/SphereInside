"""展開圖 → PDF，以及檢視用的 PNG。"""

import io

import numpy as np
from PIL import Image, ImageDraw
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

_CUT_WIDTH = 0.25  # 公釐
_FOLD_WIDTH = 0.15
_FOLD_DASH = (1.2, 1.2)
_PT_PER_MM = 72 / 25.4


def _text_angle(along):
    """文字沿著邊排列，但不要上下顛倒。"""
    angle = np.degrees(np.arctan2(along[1], along[0]))
    if angle > 90 or angle <= -90:
        angle += 180
    return angle


def write_pdf(path, layouts, images, page_width, page_height, title="SphereInside"):
    """每頁先放整頁的貼圖，再以向量畫上摺線、裁切線與邊的編號。"""
    c = canvas.Canvas(str(path), pagesize=(page_width * mm, page_height * mm))
    c.setTitle(title)
    for n, (layout, image) in enumerate(zip(layouts, images), 1):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, "JPEG", quality=92, subsampling=0)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=page_width * mm, height=page_height * mm)

        c.setLineCap(1)
        c.saveState()
        c.setStrokeColorRGB(0, 0, 0)
        c.setStrokeAlpha(0.45)
        c.setLineWidth(_FOLD_WIDTH * mm)
        c.setDash(_FOLD_DASH[0] * mm, _FOLD_DASH[1] * mm)
        for a, b in layout.folds:
            c.line(a[0] * mm, a[1] * mm, b[0] * mm, b[1] * mm)
        c.restoreState()

        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(_CUT_WIDTH * mm)
        for a, b in layout.cuts:
            c.line(a[0] * mm, a[1] * mm, b[0] * mm, b[1] * mm)

        for label in layout.labels:
            size = label.size * _PT_PER_MM
            c.saveState()
            c.translate(label.pos[0] * mm, label.pos[1] * mm)
            c.rotate(_text_angle(label.along))
            c.setFont("Helvetica-Bold", size)
            # 先畫白色外框，數字壓在照片上才看得清楚
            c.setStrokeColorRGB(1, 1, 1)
            c.setFillColorRGB(1, 1, 1)
            c.setLineWidth(size * 0.35)
            c.setLineJoin(1)
            c.drawCentredString(0, -size * 0.36, label.text, mode=2)
            c.setFillColorRGB(0, 0, 0)
            c.drawCentredString(0, -size * 0.36, label.text)
            c.restoreState()

        c.setFont("Helvetica", 7)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(4 * mm, 3 * mm, f"{title}  {n} / {len(layouts)}")
        c.showPage()
    c.save()


def write_png(path, layout, image, page_height, dpi):
    """把一頁連同線條存成 PNG，方便不開 PDF 就檢視結果（不含邊的編號）。"""
    scale = dpi / 25.4
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    def px(p):
        return (p[0] * scale, (page_height - p[1]) * scale)

    for a, b in layout.folds:
        draw.line([px(a), px(b)], fill=(110, 110, 110), width=max(1, round(_FOLD_WIDTH * scale)))
    for a, b in layout.cuts:
        draw.line([px(a), px(b)], fill=(0, 0, 0), width=max(1, round(_CUT_WIDTH * scale)))
    img.save(path)
