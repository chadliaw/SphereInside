import unittest

import numpy as np

from sphereinside import coverage
from sphereinside.camera import Photo, PhotoSet, basis
from sphereinside.synthetic import render_photo


def scene(dirs):
    """顏色隨方向平滑變化，方便比對取樣結果。"""
    return 127.5 * (1 + dirs)


def full_set():
    photos = [render_photo(scene, az, el, 100, 160, 160, roll=7) for el in (-45, 0, 45) for az in range(0, 360, 45)]
    photos += [render_photo(scene, 0, el, 100, 160, 160) for el in (-90, 90)]
    return PhotoSet(photos)


class CameraTest(unittest.TestCase):
    def test_basis_orientation(self):
        fwd, right, up = basis(0, 0)
        np.testing.assert_allclose(fwd, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
        # 方位從上方看順時針增加：朝方位 0° 時，右手邊是方位 90°
        np.testing.assert_allclose(right, basis(90, 0)[0], atol=1e-12)
        np.testing.assert_allclose(basis(0, 90)[0], [0, 0, 1], atol=1e-12)

    def test_basis_orthonormal(self):
        for args in [(30, 20, 10), (200, -80, -35), (0, 90, 0), (123, -90, 45)]:
            B = np.array(basis(*args))
            np.testing.assert_allclose(B @ B.T, np.eye(3), atol=1e-12)
            # 右 × 上 朝向鏡頭後方
            np.testing.assert_allclose(np.cross(B[1], B[2]), -B[0], atol=1e-12)

    def test_pixel_direction_round_trip(self):
        photo = Photo(np.zeros((300, 400, 3), np.uint8), 77, -33, 60, roll=12)
        rng = np.random.default_rng(1)
        u, v = rng.uniform(0, 399, 50), rng.uniform(0, 299, 50)
        u2, v2, c = photo.direction_to_pixel(photo.pixel_to_direction(u, v))
        np.testing.assert_allclose(u2, u, atol=1e-8)
        np.testing.assert_allclose(v2, v, atol=1e-8)
        self.assertTrue((c > 0).all())

    def test_center_and_field_of_view(self):
        photo = Photo(np.zeros((301, 401, 3), np.uint8), 40, 10, 60)
        np.testing.assert_allclose(photo.pixel_to_direction(200, 150), photo.fwd, atol=1e-12)
        left, right = photo.pixel_to_direction([-0.5, 400.5], [150, 150])
        self.assertAlmostEqual(np.degrees(np.arccos(left @ right)), 60, places=6)

    def test_sample_matches_scene(self):
        photos = full_set()
        dirs = coverage.fibonacci_sphere(3000)
        colors, ok = photos.sample(dirs)
        self.assertTrue(ok.all())
        self.assertLess(np.abs(colors - scene(dirs)).max(), 2.5)

    def test_priority_decides_overlap(self):
        a = render_photo(lambda d: np.full((len(d), 3), 10.0), 0, 0, 60, 40, 40)
        b = render_photo(lambda d: np.full((len(d), 3), 200.0), 20, 0, 60, 40, 40)
        d = np.array([[np.cos(np.radians(5)), -np.sin(np.radians(5)), 0]])  # 方位 5°，較靠近 a 的中心
        self.assertEqual(PhotoSet([a, b]).sample(d)[0][0, 0], 10)
        b.priority = 1
        self.assertEqual(PhotoSet([a, b]).sample(d)[0][0, 0], 200)
        # 順序只在照片涵蓋的範圍內有效
        far = np.array([[np.cos(np.radians(-25)), -np.sin(np.radians(-25)), 0]])
        self.assertEqual(PhotoSet([a, b]).sample(far)[0][0, 0], 10)

    def test_crop(self):
        photo = render_photo(scene, 0, 0, 60, 100, 100)
        center = np.array([[1.0, 0, 0]])
        right_edge = photo.pixel_to_direction(95, 50)[None]
        self.assertTrue(PhotoSet([photo]).covered(np.vstack([center, right_edge])).all())
        photo.crop = (0.0, 0.0, 0.6, 1.0)
        self.assertEqual(PhotoSet([photo]).covered(np.vstack([center, right_edge])).tolist(), [True, False])

    def test_behind_camera_is_not_covered(self):
        photos = PhotoSet([render_photo(scene, 0, 0, 60, 80, 80)])
        self.assertFalse(photos.covered(np.array([[-1.0, 0, 0], [0, 1.0, 0]])).any())
        self.assertTrue(photos.covered(np.array([[1.0, 0, 0]])).all())


class CoverageTest(unittest.TestCase):
    def test_complete(self):
        report = coverage.check(full_set(), n=5000)
        self.assertTrue(report.complete)
        self.assertEqual(report.fraction, 1)

    def test_reports_missing_region(self):
        photos = PhotoSet([p for p in full_set().photos if p.elevation > -40])
        report = coverage.check(photos, n=5000)
        self.assertFalse(report.complete)
        self.assertLess(report.fraction, 1)
        # 缺的是下方
        self.assertTrue(all(el < 0 for _, el, _ in report.missing))
        self.assertEqual(sum(el == -90 for _, el, _ in report.missing), 12)
        self.assertIn("仰角", report.describe())


if __name__ == "__main__":
    unittest.main()
