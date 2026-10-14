"""產生 capture/polyhedra.json，讓網頁畫出與 PDF 相同的多面體。

    python capture/make_polyhedra.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sphereinside import polyhedron  # noqa: E402

LEVELS = (0, 1, 2)
OUT = Path(__file__).resolve().parent / "polyhedra.json"


def data():
    result = {}
    for level in LEVELS:
        V, S = polyhedron.build(level)
        result[str(level)] = {"V": [[round(float(c), 7) for c in v] for v in V], "S": S.tolist()}
    return result


def main():
    OUT.write_text(json.dumps(data(), separators=(",", ":")), encoding="utf-8")
    print(f"已輸出 {OUT}")


if __name__ == "__main__":
    main()
