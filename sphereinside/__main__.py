"""命令列介面：python -m sphereinside {check,build} ..."""

import argparse
import sys
from pathlib import Path

from . import coverage, export, paint, polyhedron, unfold
from .camera import load_photos

PAPERS = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "Letter": (215.9, 279.4)}


def _check(args):
    report = coverage.check(load_photos(args.meta, max_focal=400))
    print(report.describe())
    return 0 if report.complete else 1


def _build(args):
    width, height = PAPERS[args.paper]
    radius = args.diameter / 2
    scale = args.dpi / 25.4
    # 照片的解析度只需要略高於印出來的解析度
    photos = load_photos(args.meta, max_focal=1.5 * radius * scale)
    print(f"讀入 {len(photos.photos)} 張照片")

    report = coverage.check(photos)
    print(report.describe())
    if not report.complete and not args.allow_gaps:
        print("請補拍後再試，或加上 --allow-gaps 以灰色填補沒拍到的地方。", file=sys.stderr)
        return 1

    V, S = polyhedron.build(args.level)
    V = V * radius
    layouts, (tabs, cuts) = unfold.unfold(V, S, width, height, margin=args.margin, tab=args.tab)
    print(f"多面體有 {len(S)} 個面，展開成 {len(layouts)} 頁，共 {cuts} 組要黏合的邊")
    if tabs < cuts:
        print(f"其中 {cuts - tabs} 組因為空間不夠而沒有黏貼耳，組裝時需要從內側另外貼紙條")

    out = Path(args.output)
    images = []
    for n, layout in enumerate(layouts, 1):
        image, _ = paint.render_page(layout, V, S, photos, width, height, args.dpi, mirror=not args.no_mirror)
        images.append(image)
        if args.png:
            export.write_png(out.with_name(f"{out.stem}-{n}.png"), layout, image, height, args.dpi)
        print(f"  第 {n} 頁完成")
    export.write_pdf(out, layouts, images, width, height)
    print(f"已輸出 {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sphereinside", description="把空間的內部印在多面體的外部")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="檢查照片是否涵蓋所有方向")
    p.add_argument("meta", help="拍攝網頁輸出的對照檔（meta.json）")
    p.set_defaults(run=_check)

    p = sub.add_parser("build", help="產生可列印的展開圖")
    p.add_argument("meta", help="拍攝網頁輸出的對照檔（meta.json）")
    p.add_argument("-o", "--output", default="sphereinside.pdf", help="輸出的 PDF 檔名")
    p.add_argument("--level", type=int, default=1, help="細分輪數，面數為 20 × 3^level（預設 1，即 60 面）")
    p.add_argument("--diameter", type=float, default=150.0, help="外接球直徑，公釐（預設 150）")
    p.add_argument("--paper", choices=PAPERS, default="A4")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--margin", type=float, default=8.0, help="紙張邊界，公釐")
    p.add_argument("--tab", type=float, default=5.0, help="黏貼耳高度，公釐")
    p.add_argument("--allow-gaps", action="store_true", help="照片沒有涵蓋所有方向時仍然輸出")
    p.add_argument("--no-mirror", action="store_true", help="不做鏡射（從外面看會左右相反）")
    p.add_argument("--png", action="store_true", help="另外輸出每一頁的 PNG")
    p.set_defaults(run=_build)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
