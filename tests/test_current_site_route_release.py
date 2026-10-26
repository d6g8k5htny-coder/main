from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/current_site_route_window_release.py"


class RouteReleaseTest(unittest.TestCase):
    def test_rewrites_only_the_atlas_release_pin_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            atlas = work / "atlas.mjs"
            source = work / "dependencies.html"
            output = work / "rendered.html"
            atlas.write_bytes(b"export const routeWindow = 150;\n")
            source.write_text(
                '<script type="module" src="atlas.mjs?site-release=' + "0" * 64 + '"></script>\n'
                '<script type="module" src="current.mjs?site-release=' + "1" * 64 + '"></script>\n',
                encoding="utf-8",
            )
            subprocess.run(
                ["python3", "-B", "-S", str(SCRIPT), "--input", str(source), "--atlas", str(atlas), "--output", str(output)],
                check=True,
            )
            digest = hashlib.sha256(atlas.read_bytes()).hexdigest()
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                f'<script type="module" src="atlas.mjs?site-release={digest}"></script>\n'
                f'<script type="module" src="current.mjs?site-release={"1" * 64}"></script>\n',
            )
            subprocess.run(
                ["python3", "-B", "-S", str(SCRIPT), "--input", str(output), "--atlas", str(atlas), "--output", str(output)],
                check=True,
            )

    def test_refuses_missing_or_ambiguous_pin(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            atlas = work / "atlas.mjs"
            atlas.write_text("export {};\n", encoding="utf-8")
            for body in ["<html></html>\n", (f'atlas.mjs?site-release={"0" * 64}\n') * 2]:
                source = work / "dependencies.html"
                source.write_text(body, encoding="utf-8")
                result = subprocess.run(
                    ["python3", "-B", "-S", str(SCRIPT), "--input", str(source), "--atlas", str(atlas), "--output", str(work / "out")],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exactly one atlas release pin", result.stderr)


if __name__ == "__main__":
    unittest.main()
