import tempfile
import unittest
from pathlib import Path

from commurenew_agent.app import _resolve_selected_site_image_paths


class ResolveSelectedSiteImagePathsTest(unittest.TestCase):
    def test_resolve_by_filename_and_keep_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img1 = root / "室外活动场地-11.JPG"
            img2 = root / "社区出入口-1.jpg"
            img1.write_bytes(b"a")
            img2.write_bytes(b"b")

            selected = ["室外活动场地-11.JPG", "社区出入口-1.jpg"]
            site_images = [str(img1), str(img2)]

            resolved = _resolve_selected_site_image_paths(selected, site_images)

            self.assertEqual(resolved, [str(img1), str(img2)])

    def test_deduplicate_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img = root / "室外活动场地-11.JPG"
            img.write_bytes(b"a")
            site_images = [str(img)]

            selected = ["室外活动场地-11.JPG", "室外活动场地-11.jpg"]
            resolved = _resolve_selected_site_image_paths(selected, site_images)

            self.assertEqual(resolved, [str(img)])


if __name__ == "__main__":
    unittest.main()
