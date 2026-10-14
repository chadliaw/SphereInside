"""命令列輸出與錯誤訊息的中英文。"""

LANG = "zh"  # "zh" 或 "en"，由命令列的 --lang 設定

_TEXT = {
    # 照片
    "no_photos": ("沒有照片", "No photos"),
    "no_hfov": ("{file}：對照檔沒有 hfov，照片的 EXIF 也沒有等效焦距", "{file}: no hfov in the metadata, and no 35 mm equivalent focal length in EXIF"),
    "bad_crop": ("{file}：crop 必須是 0 到 1 之間的 [左, 上, 右, 下]", "{file}: crop must be [left, top, right, bottom] between 0 and 1"),
    "loaded": ("讀入 {n} 張照片", "Loaded {n} photos"),
    # 涵蓋檢查
    "coverage_complete": ("照片已涵蓋所有方向。", "The photos cover every direction."),
    "coverage_partial": ("照片涵蓋了 {percent} 的方向，以下範圍還沒拍到：", "The photos cover {percent} of all directions. Not yet covered:"),
    "coverage_cell": ("  方位 {az0:3d}°–{az1:3d}°、仰角 {el0:+3d}°–{el1:+3d}°（缺 {percent}）", "  azimuth {az0:3d}°–{az1:3d}°, elevation {el0:+3d}°–{el1:+3d}° ({percent} missing)"),
    "coverage_more": ("  …以及另外 {n} 個範圍", "  …and {n} more ranges"),
    "less_than": ("不到 {x}%", "less than {x}%"),
    "retry_gaps": ("請補拍後再試，或加上 --allow-gaps 以灰色填補沒拍到的地方。", "Take more photos and try again, or add --allow-gaps to fill the gaps with grey."),
    # 展開
    "face_too_large": ("單一個面就放不進紙張，請縮小直徑、增加細分輪數或改用較大的紙張", "A single face does not fit on the paper; use a smaller diameter, more subdivision or larger paper"),
    "piece_too_large": ("展開圖的一片放不進紙張", "A piece of the net does not fit on the paper"),
    "summary": ("多面體有 {faces} 個面，展開成 {pages} 頁，共 {cuts} 組要黏合的邊", "The polyhedron has {faces} faces on {pages} pages, with {cuts} pairs of edges to glue"),
    "no_tabs": ("其中 {n} 組因為空間不夠而沒有黏貼耳，組裝時需要從內側另外貼紙條", "{n} of them have no glue tab for lack of space; tape them from the inside"),
    "labels_omitted": ("有 {n} 個編號在圖片外放不下而省略；它們多半在缺口兩側，對面的邊上有相同的編號", "{n} edge numbers did not fit outside the picture and were left out; they are mostly across narrow gaps from their matching edge"),
    # 輸出
    "page_done": ("  第 {n} 頁完成", "  Page {n} done"),
    "written": ("已輸出 {path}", "Wrote {path}"),
    "error": ("錯誤：{message}", "Error: {message}"),
}


def t(key, **values):
    zh, en = _TEXT[key]
    return (en if LANG == "en" else zh).format(**values)
