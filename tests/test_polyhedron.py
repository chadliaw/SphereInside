import unittest

import numpy as np

from sphereinside import polyhedron


class PolyhedronTest(unittest.TestCase):
    def test_icosahedron(self):
        V, S = polyhedron.icosahedron()
        self.assertEqual((len(V), len(S)), (12, 20))
        np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1)

    def test_subdivision_counts(self):
        for level, (nv, nf) in enumerate([(12, 20), (32, 60), (92, 180), (272, 540)]):
            V, S = polyhedron.build(level)
            self.assertEqual((len(V), len(S)), (nv, nf))
            np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1)

    def test_faces_counterclockwise_from_outside(self):
        V, S = polyhedron.build(2)
        a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
        outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c)
        self.assertTrue((outward > 0).all())

    def test_convex_and_well_shaped(self):
        V, S = polyhedron.build(3)
        a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
        n = np.cross(b - a, c - a)
        # 凸：所有頂點都在每個面的內側
        self.assertLess((np.einsum("fj,vj->fv", n, V) - np.einsum("fj,fj->f", n, a)[:, None]).max(), 1e-9)
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            u, w = q - p, r - p
            cos = np.einsum("ij,ij->i", u, w) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1)
            self.assertGreater(np.degrees(np.arccos(cos)).min(), 50)

    def test_spherical_round_trip(self):
        p = np.random.default_rng(0).normal(size=(100, 3))
        np.testing.assert_allclose(polyhedron.from_spherical(*polyhedron.to_spherical(p)), p, atol=1e-12)
        r, theta, phi = polyhedron.to_spherical(np.array([[0, 0, 2.0], [1.0, 0, 0], [0, 3.0, 0]]))
        np.testing.assert_allclose(r, [2, 1, 3])
        np.testing.assert_allclose(phi, [0, np.pi / 2, np.pi / 2])
        np.testing.assert_allclose(theta[1:], [0, np.pi / 2])


if __name__ == "__main__":
    unittest.main()
