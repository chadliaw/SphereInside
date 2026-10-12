import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sphereinside import paint, polyhedron
from sphereinside.__main__ import main
from sphereinside.unfold import unfold

from .test_camera import full_set, scene

W, H, DPI = 210.0, 297.0, 60


class PaintTest(unittest.TestCase):
    def render(self, mirror):
        V, S = polyhedron.build(1)
        V = V * 60
        layouts, _ = unfold(V, S, W, H)
        photos = full_set()
        for layout in layouts:
            image, uncovered = paint.render_page(layout, V, S, photos, W, H, DPI, mirror=mirror)
            self.assertEqual(uncovered, 0)
            for f, tri in layout.faces:
                # 面中心的像素，顏色應該是場景在（鏡射後的）對應方向的顏色
                x, y = tri.mean(0) * DPI / 25.4
                d = V[S[f]].mean(0)
                d = d / np.linalg.norm(d)
                if mirror:
                    d = d * [1, -1, 1]
                got = image[image.shape[0] - 1 - int(y), int(x)].astype(float)
                self.assertLess(np.abs(got - scene(d[None])[0]).max(), 6)

    def test_mirrored(self):
        self.render(True)

    def test_not_mirrored(self):
        self.render(False)

    def test_gaps_are_filled_and_counted(self):
        V, S = polyhedron.build(0)
        V = V * 40
        layouts, _ = unfold(V, S, W, H)
        photos = full_set()
        photos.photos = photos.photos[:3]
        total = 0
        for layout in layouts:
            image, uncovered = paint.render_page(layout, V, S, photos, W, H, DPI, gap_color=(1, 2, 3))
            self.assertEqual(uncovered, int((image == (1, 2, 3)).all(-1).sum()))
            total += uncovered
        self.assertGreater(total, 0)


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        entries = []
        for n, p in enumerate(full_set().photos):
            Image.fromarray(p.image).save(self.dir / f"{n}.png")
            entries.append({"file": f"{n}.png", "azimuth": p.azimuth, "elevation": p.elevation, "roll": p.roll})
        self.entries = entries

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self, entries):
        path = self.dir / "meta.json"
        path.write_text(json.dumps({"hfov": 100, "photos": entries}), encoding="utf-8")
        return str(path)

    def test_build(self):
        out = self.dir / "out.pdf"
        code = main(["build", self.write_meta(self.entries), "-o", str(out), "--dpi", "50", "--diameter", "100", "--png"])
        self.assertEqual(code, 0)
        self.assertEqual(out.read_bytes()[:5], b"%PDF-")
        self.assertTrue((self.dir / "out-1.png").exists())

    def test_incomplete_coverage_stops(self):
        meta = self.write_meta(self.entries[:5])
        out = self.dir / "out.pdf"
        self.assertEqual(main(["check", meta]), 1)
        self.assertEqual(main(["build", meta, "-o", str(out), "--dpi", "50"]), 1)
        self.assertFalse(out.exists())
        self.assertEqual(main(["build", meta, "-o", str(out), "--dpi", "50", "--allow-gaps"]), 0)
        self.assertTrue(out.exists())

    def test_missing_hfov(self):
        path = self.dir / "meta.json"
        path.write_text(json.dumps({"photos": self.entries[:1]}), encoding="utf-8")
        self.assertEqual(main(["check", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
