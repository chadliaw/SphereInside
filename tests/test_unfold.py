import unittest

import numpy as np

from sphereinside import polyhedron
from sphereinside.unfold import _overlaps, unfold

W, H = 210.0, 297.0


def signed_area(t):
    return ((t[1] - t[0])[0] * (t[2] - t[0])[1] - (t[1] - t[0])[1] * (t[2] - t[0])[0]) / 2


class OverlapTest(unittest.TestCase):
    def test_overlap(self):
        a = np.array([[0, 0], [2, 0], [0, 2.0]])
        self.assertTrue(_overlaps(a, [a + 0.5]))
        self.assertFalse(_overlaps(a, [a + [3, 0]]))
        # 共用一條邊或一個頂點不算重疊
        self.assertFalse(_overlaps(a, [np.array([[2, 0], [0, 2], [2, 2.0]])]))
        self.assertFalse(_overlaps(a, [np.array([[2, 0], [4, 0], [3, 1.0]])]))


class UnfoldTest(unittest.TestCase):
    def check(self, level, radius):
        V, S = polyhedron.build(level)
        V = V * radius
        layouts, (tabs, cuts) = unfold(V, S, W, H)

        placed = [f for page in layouts for f, _ in page.faces]
        self.assertEqual(sorted(placed), list(range(len(S))))

        for page in layouts:
            tris = [t for _, t in page.faces]
            for f, t in page.faces:
                # 邊長不變、沒有被翻面
                for k in range(3):
                    self.assertAlmostEqual(
                        np.linalg.norm(t[k] - t[(k + 1) % 3]),
                        np.linalg.norm(V[S[f][k]] - V[S[f][(k + 1) % 3]]),
                        places=7,
                    )
                self.assertGreater(signed_area(t), 0)
            for i, t in enumerate(tris):
                self.assertFalse(_overlaps(t, tris[:i] + tris[i + 1 :]))
            pts = np.vstack(tris + list(page.cuts))
            self.assertGreaterEqual(pts.min(), 8 - 1e-6)
            self.assertLessEqual(pts[:, 0].max(), W - 8 + 1e-6)
            self.assertLessEqual(pts[:, 1].max(), H - 8 + 1e-6)

        # 每條邊不是摺線就是剪開；剪開的邊在兩側各有一個相同的編號
        edges = len(S) * 3 // 2
        labels = sorted(int(l.text) for page in layouts for l in page.labels)
        self.assertEqual(labels, sorted(list(range(1, cuts + 1)) * 2))
        folds = sum(len(page.folds) for page in layouts)
        self.assertEqual(folds - tabs, edges - cuts)
        return tabs, cuts

    def test_icosahedron(self):
        tabs, cuts = self.check(0, 60)
        self.assertEqual(tabs, cuts)

    def test_level_1(self):
        tabs, cuts = self.check(1, 75)
        self.assertEqual(tabs, cuts)

    def test_level_2(self):
        self.check(2, 75)

    def test_face_too_large(self):
        V, S = polyhedron.icosahedron()
        with self.assertRaises(ValueError):
            unfold(V * 500, S, W, H)


if __name__ == "__main__":
    unittest.main()
