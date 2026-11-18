"""Run with .venv/Scripts/python.exe -m unittest discover -s tests."""
import asyncio
import io
import unittest
from tempfile import SpooledTemporaryFile
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from PIL import Image

import app


class RegionTests(unittest.TestCase):
    def upload(self):
        image = Image.new("RGB", (100, 80), "red")
        image.paste("blue", (25, 20, 75, 60))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        stream = SpooledTemporaryFile(max_size=1024*1024)
        stream.write(encoded.getvalue())
        stream.seek(0)
        self.addCleanup(stream.close)
        return UploadFile(filename="region.png", file=stream)

    def test_only_selected_pixels_reach_inference(self):
        worker = AsyncMock(return_value={})
        with patch.object(app, "run_in_threadpool", worker):
            asyncio.run(app.identify(self.upload(), "0.25,0.25,0.75,0.75"))
        _, image, _ = worker.call_args.args
        self.assertEqual(image.size, (50, 40))
        self.assertEqual(image.getextrema(), ((0, 0), (0, 0), (255, 255)))

    def test_invalid_regions_never_reach_inference(self):
        for region in ("", "0,0,1", "nan,0,1,1", "0,0,inf,1",
                       "-0.1,0,1,1", "0.9,0,0.2,1", "0,0,1.1,1", "0,0,0.01,0.01"):
            with self.subTest(region=region), patch.object(app, "run_in_threadpool", AsyncMock()) as worker:
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(app.identify(self.upload(), region))
                self.assertEqual(caught.exception.status_code, 422)
                worker.assert_not_called()

    def test_no_region_preserves_full_image(self):
        worker = AsyncMock(return_value={})
        with patch.object(app, "run_in_threadpool", worker):
            result = asyncio.run(app.identify(self.upload(), None))
        self.assertEqual(worker.call_args.args[1].size, (100, 80))
        self.assertIsNone(result["region"])

    def test_original_inference_pipeline_is_unchanged(self):
        import ast
        import subprocess
        from pathlib import Path
        original = ast.parse(subprocess.check_output(["git", "show", "800661239e44869267828701680dc8301434796f:app.py"]).decode("utf-8"))
        current = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        before = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "compare_image")
        after = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "compare_image")
        self.assertEqual(ast.dump(before), ast.dump(after))


if __name__ == "__main__":
    unittest.main()
