"""Renders the dashboard against a stub DOM and drives its controls.

The page is one inline script built by string substitution, so a broken filter
or a detail panel whose arithmetic stops adding up shows only in a browser.
analysis/devtools/dash_drive.js exercises those states and exits non-zero.
"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "analysis" / "devtools" / "dash_drive.js"


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class DashboardRenderTest(unittest.TestCase):
    def test_the_dashboard_survives_every_control_state(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            demo = out / "demo"
            for cmd in (
                [sys.executable, str(ROOT / "analysis" / "make_demo_runs.py"),
                 "--out", str(demo)],
                [sys.executable, str(ROOT / "analysis" / "build_dashboard.py"),
                 "--dir", str(demo), "--out-dir", str(out)],
            ):
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

            r = subprocess.run(
                ["node", str(DRIVER), str(out / "dashboard_artifact.html")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
