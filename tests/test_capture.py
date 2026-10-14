import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("make_polyhedra", Path(__file__).resolve().parent.parent / "capture" / "make_polyhedra.py")
make_polyhedra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_polyhedra)


class PolyhedraJsonTest(unittest.TestCase):
    def test_up_to_date(self):
        """capture/polyhedra.json 要與 sphereinside 產生的多面體一致；不一致時執行 capture/make_polyhedra.py。"""
        saved = json.loads(make_polyhedra.OUT.read_text(encoding="utf-8"))
        for level, fresh in make_polyhedra.data().items():
            self.assertEqual(saved[level]["S"], fresh["S"])
            np.testing.assert_allclose(saved[level]["V"], fresh["V"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
